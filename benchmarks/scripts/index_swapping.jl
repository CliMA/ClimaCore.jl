#=
julia --project=.buildkite
julia -g2 --check-bounds=yes --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "index_swapping.jl"))

# Info:
This script compares the performance
of our universal index support (for
mixed DataLayout operations) against
specialized index support for uniform
DataLayout operations.

In particular,
 - `at_dot_call!` is a reference for the speed of light
                  we could achieve on the hardware, as
                  memory coalescence comes for free on
                  vectors (as opposed to arrays).
 - `custom_kernel_bc!(; swap = 0)` mimics our specialized operations
 - `custom_kernel_bc!(; swap = 1)` mimics our generalized pointwise operations
 - `custom_kernel_bc!(; swap = 2)` mimics our generalized stencil operations

# Benchmark results:

Clima A100
```
[ Info: ArrayType = CuArray
Problem size: (5443200,), N reads-writes: 2, N-reps: 1000,  Float_type = Float32, Device_bandwidth_GBs=2039
┌──────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┐
│ funcs                                                                │ time per call                    │ bw %    │ achieved bw │
├──────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┤
│ BIS.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)                 │ 34 microseconds, 738 nanoseconds │ 57.2576 │ 1167.48     │
│ BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=0, nreps=1000, bm) │ 60 microseconds, 528 nanoseconds │ 32.8605 │ 670.025     │
│ BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=1, nreps=1000, bm) │ 68 microseconds, 147 nanoseconds │ 29.1867 │ 595.118     │
│ BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=2, nreps=1000, bm) │ 60 microseconds, 524 nanoseconds │ 32.8627 │ 670.07      │
└──────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┘
```
=#

#! format: off
module IndexSwapBench

import CUDA
include("benchmark_utils.jl")

foo(x1, x2, x3) = x1
function at_dot_call!(X, Y; nreps = 1, print_info = true, bm=nothing, n_trials = 30)
    (; x1, x2, x3) = X
    (; y1) = Y
    e = Inf
    @. y1 = foo(x1, x2, x3) # compile
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps # reduce variance / impact of launch latency
                @. y1 = foo(x1, x2, x3) # 1 write, 1 read
            end
        end
        e = min(e, et)
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__),problem_size=size(X.x1),n_reads_writes=2)
    return nothing
end;

function custom_kernel_bc!(X, Y, us::UniversalSizesStatic; swap=0, printtb=false, nreps = 1, print_info = true, bm=nothing, n_trials=30)
    (; x1, x2, x3) = X
    (; y1) = Y
    bc = @lazy @. foo(x1, x2, x3)
    @assert !(y1 isa Array)
    f = if swap==0
        custom_kernel_knl_bc_0swap!
    elseif swap == 1
        custom_kernel_knl_bc_1swap!
    elseif swap == 2
        custom_kernel_knl_bc_2swap!
    else
        error("oops")
    end
    kernel =
        CUDA.@cuda always_inline = true launch = false f(
            y1,
            bc,
            us,
        )
    N = get_N(us)
    config = CUDA.launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    printtb && @show blocks, threads
    kernel(y1, bc,us; threads, blocks) # compile
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps # reduce variance / impact of launch latency
                kernel(y1, bc,us; threads, blocks)
            end
        end
        e = min(e, et)
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__),problem_size=size(X.x1),n_reads_writes=2)
    return nothing
end;

# Mimics how indexing works in generalized pointwise kernels
function custom_kernel_knl_bc_1swap!(y1, bc, us)
    @inbounds begin
        tidx = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if tidx ≤ get_N(us)
            n = (get_Nij(us), get_Nij(us), 1, get_Nv(us), get_Nh(us))
            GCI = CartesianIndices(map(x -> Base.OneTo(x), n))[tidx]
            # Perform index swap (as in `getindex(::AbstractData, ::CartesianIndex)`)
            i, j, _, v, h = GCI.I
            CI = CartesianIndex(v, i, j, 1, h)
            y1[CI] = bc[CI]
        end
    end
    return nothing
end

# Mimics how indexing works in specialized kernels
function custom_kernel_knl_bc_0swap!(y1, bc, us)
    @inbounds begin
        tidx = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if tidx ≤ get_N(us)
            n = (get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us))
            CI = CartesianIndices(map(x -> Base.OneTo(x), n))[tidx]
            y1[CI] = bc[CI] # requires special broadcasted index support
        end
    end
    return nothing
end
# Mimics how indexing works in generalized stencil kernels
function custom_kernel_knl_bc_2swap!(y1, bc, us)
    @inbounds begin
        tidx = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if tidx ≤ get_N(us)
            # We start with a VIJFH-specific CartesianIndex
            n = (get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us))
            CIK = CartesianIndices(map(x -> Base.OneTo(x), n))[tidx] # data-specific in kernel

            # Swap in `getidx`
            (v, i, j, _, h) = CIK.I
            GCI = CartesianIndex(i, j, 1, v, h)

            # Swap again (in `getindex(::AbstractData, ::CartesianIndex)`)
            (i, j, _, v, h) = GCI.I
            CI = CartesianIndex(v, i, j, 1, h)
            y1[CI] = bc[CI]
        end
    end
    return nothing
end

import Random
using Test
function test_custom_kernel_bc!(X_array, Y_array, uss; swap)
    Random.seed!(1234)
    X_array.x1 .= typeof(X_array.x1)(rand(eltype(X_array.x1), size(X_array.x1)))
    Y_array_cp = deepcopy(Y_array)
    custom_kernel_bc!(X_array, Y_array_cp, uss; swap=0, print_info = false)
    custom_kernel_bc!(X_array, Y_array, uss; swap, print_info = false)
    @test all(Y_array_cp.y1 .== Y_array.y1)
end

end # module

import .IndexSwapBench as BIS

using CUDA
device_name = CUDA.name(CUDA.device())
bm = BIS.Benchmark(;problem_size=(63,4,4,1,5400), device_name, float_type=Float32)
ArrayType = CUDA.CuArray;
# ArrayType = identity;
arr(bm, T) = T(zeros(bm.float_type, bm.problem_size...))
X_array = (;x1 = arr(bm, ArrayType),x2 = arr(bm, ArrayType),x3 = arr(bm, ArrayType));
Y_array = (;y1 = arr(bm, ArrayType),);
to_vec(ξ) = (;zip(propertynames(ξ), map(θ -> vec(θ), values(ξ)))...);
X_vector = to_vec(X_array);
Y_vector = to_vec(Y_array);
N = length(X_vector.x1)
(Nv, Nij, _, _, Nh) = size(Y_array.y1);
uss = BIS.UniversalSizesStatic(Nv, Nij, Nh);
BIS.at_dot_call!(X_vector, Y_vector; nreps=1)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=0, nreps=1)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=1, nreps=1)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=2, nreps=1)
BIS.test_custom_kernel_bc!(X_array, Y_array, uss; swap=1)
BIS.test_custom_kernel_bc!(X_array, Y_array, uss; swap=2)

BIS.at_dot_call!(X_vector, Y_vector; nreps=1000, bm)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=0, nreps=1000, bm)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=1, nreps=1000, bm)
BIS.custom_kernel_bc!(X_array, Y_array, uss; swap=2, nreps=1000, bm)

@info "ArrayType = $ArrayType"
BIS.tabulate_benchmark(bm)

#! format: on
