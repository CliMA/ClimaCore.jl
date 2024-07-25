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
at_dot_call!($X_vector, $Y_vector):
     6 milliseconds, 19 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss, swap = 0):
     6 milliseconds, 329 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss, swap = 1):
     14 milliseconds, 232 microseconds
custom_kernel_bc!($X_array, $Y_array, $uss, swap = 2):
     15 milliseconds, 960 microseconds
```
=#

#! format: off
import CUDA
using BenchmarkTools, Dates
using LazyBroadcast: @lazy
ArrayType = CUDA.CuArray;
# ArrayType = identity;

if ArrayType === identity
    macro pretty_belapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(BenchmarkTools.@belapsed(esc($expr)))
        end
    end
else
    macro pretty_belapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(
                BenchmarkTools.@belapsed(CUDA.@sync((esc($expr))))
            )
        end
    end
    macro pretty_elapsed(expr)
        return quote
            println($(string(expr)), ":")
            print("     ")
            print_time_and_units(
                BenchmarkTools.@elapsed(CUDA.@sync((esc($expr))))
            )
        end
    end
end
print_time_and_units(x) = println(time_and_units_str(x))
time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end
trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s
foo(x1, x2, x3) = x1
function at_dot_call!(X, Y)
    (; x1, x2, x3) = X
    (; y1) = Y
    for i in 1:100 # reduce variance / impact of launch latency
        @. y1 = foo(x1, x2, x3) # 3 reads, 1 write
    end
    return nothing
end;

struct UniversalSizesStatic{Nv, Nij, Nh} end

get_Nv(::UniversalSizesStatic{Nv}) where {Nv} = Nv
get_Nij(::UniversalSizesStatic{Nv, Nij}) where {Nv, Nij} = Nij
get_Nh(::UniversalSizesStatic{Nv, Nij, Nh}) where {Nv, Nij, Nh} = Nh
get_N(us::UniversalSizesStatic{Nv, Nij}) where {Nv, Nij} = prod((Nv,Nij,Nij,1,get_Nh(us)))
UniversalSizesStatic(Nv, Nij, Nh) = UniversalSizesStatic{Nv, Nij, Nh}()
using Test

function custom_kernel_bc!(X, Y, us::UniversalSizesStatic; swap=0, printtb=false)
    (; x1, x2, x3) = X
    (; y1) = Y
    bc = @lazy @. y1 = foo(x1, x2, x3)
    @assert !(y1 isa Array)
    f = if swap==0
        custom_kernel_knl_bc_no_swap!
    elseif swap == 1
        custom_kernel_knl_bc_swap!
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
    for i in 1:100 # reduce variance / impact of launch latency
        kernel(y1, bc,us; threads, blocks)
    end
    return nothing
end;

# Mimics how indexing works in generalized pointwise kernels
function custom_kernel_knl_bc_swap!(y1, bc, us)
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
function custom_kernel_knl_bc_no_swap!(y1, bc, us)
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
function test_custom_kernel_bc!(X_array, Y_array, uss; swap)
    Random.seed!(1234)
    X_array.x1 .= typeof(X_array.x1)(rand(eltype(X_array.x1), size(X_array.x1)))
    Y_array_cp = deepcopy(Y_array)
    custom_kernel_bc!(X_array, Y_array_cp, uss; swap=0)
    custom_kernel_bc!(X_array, Y_array, uss; swap)
    @test all(Y_array_cp.y1 .== Y_array.y1)
end

FT = Float32;
arr(T) = T(zeros(63,4,4,1,5400))
X_array = (;x1 = arr(ArrayType),x2 = arr(ArrayType),x3 = arr(ArrayType));
Y_array = (;y1 = arr(ArrayType),);
to_vec(ξ) = (;zip(propertynames(ξ), map(θ -> vec(θ), values(ξ)))...);
X_vector = to_vec(X_array);
Y_vector = to_vec(Y_array);
N = length(X_vector.x1)
(Nv, Nij, _, _, Nh) = size(Y_array.y1);
uss = UniversalSizesStatic(Nv, Nij, Nh);
at_dot_call!(X_vector, Y_vector)
custom_kernel_bc!(X_array, Y_array, uss; swap=0)
custom_kernel_bc!(X_array, Y_array, uss; swap=1)
custom_kernel_bc!(X_array, Y_array, uss; swap=2)
test_custom_kernel_bc!(X_array, Y_array, uss; swap=1)
test_custom_kernel_bc!(X_array, Y_array, uss; swap=2)

@pretty_belapsed at_dot_call!($X_vector, $Y_vector)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $uss, swap=0)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $uss, swap=1)
@pretty_belapsed custom_kernel_bc!($X_array, $Y_array, $uss, swap=2)

#! format: on
