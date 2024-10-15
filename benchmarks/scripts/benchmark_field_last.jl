#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "benchmark_field_last.jl"))

# Info

# Benchmark results:

Clima A100:
```
Kernel `add3(x1, x2, x3) = x1+x2+x3` and  `n_reads_writes=4`:
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 5400, 1), float_type = Float32, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                               │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 72 microseconds, 899 nanoseconds │ 54.568  │ 1112.64     │ 4              │ 100    │
│     FLD.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 56 microseconds, 259 nanoseconds │ 70.708  │ 1441.74     │ 4              │ 100    │
│     FLD.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 56 microseconds, 515 nanoseconds │ 70.3877 │ 1435.21     │ 4              │ 100    │
│     FLD.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 67 microseconds, 462 nanoseconds │ 58.9663 │ 1202.32     │ 4              │ 100    │
└─────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘

Kernel `add3(x1, x2, x3) = x1+x2+x3` and  `n_reads_writes=4`:
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 5400, 1), float_type = Float64, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                               │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 106 microseconds, 783 nanoseconds │ 74.5051 │ 1519.16     │ 4              │ 100    │
│     FLD.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 102 microseconds, 472 nanoseconds │ 77.6396 │ 1583.07     │ 4              │ 100    │
│     FLD.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 102 microseconds, 523 nanoseconds │ 77.6008 │ 1582.28     │ 4              │ 100    │
│     FLD.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 106 microseconds, 834 nanoseconds │ 74.4694 │ 1518.43     │ 4              │ 100    │
└─────────────────────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘

Kernel `add3(x1, x2, x3) = x1` and  `n_reads_writes=2`:
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 5400, 1), float_type = Float32, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                               │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 61 microseconds, 185 nanoseconds │ 32.5079 │ 662.837     │ 2              │ 100    │
│     FLD.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 31 microseconds, 376 nanoseconds │ 63.3926 │ 1292.57     │ 2              │ 100    │
│     FLD.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 31 microseconds, 120 nanoseconds │ 63.9141 │ 1303.21     │ 2              │ 100    │
│     FLD.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 44 microseconds, 53 nanoseconds  │ 45.1499 │ 920.607     │ 2              │ 100    │
└─────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```

# CPU (Mac M1)
```
[ Info: ArrayType = identity
Problem size: (63, 4, 4, 5400, 1), float_type = Float32, device_bandwidth_GBs=2039
┌─────────────────────────────────────────────────────────────────────┬───────────────────────────────────┬──────────┬─────────────┬────────────────┬────────┐
│ funcs                                                               │ time per call (CPU)               │ bw %     │ achieved bw │ n-reads/writes │ n-reps │
├─────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼──────────┼─────────────┼────────────────┼────────┤
│     FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 16 milliseconds, 494 microseconds │ 0.241171 │ 4.91747     │ 4              │ 100    │
│     FLD.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 783 microseconds, 256 nanoseconds │ 5.07871  │ 103.555     │ 4              │ 100    │
│     FLD.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 790 microseconds, 894 nanoseconds │ 5.02966  │ 102.555     │ 4              │ 100    │
│     FLD.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 12 milliseconds, 522 microseconds │ 0.317663 │ 6.47714     │ 4              │ 100    │
└─────────────────────────────────────────────────────────────────────┴───────────────────────────────────┴──────────┴─────────────┴────────────────┴────────┘
```

=#

#! format: off
module BenchmarkFieldLastIndex

using CUDA
include("benchmark_utils.jl")

@inline function const_linear_index(us::UniversalSizesStatic, I, field_index)
    n = (get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us), 1)
    i = I + prod(n)*field_index
    return i
end

@inline function const_linear_index_reference(us::UniversalSizesStatic, I, field_index)
    CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us), 1))
    LI = LinearIndices((get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us), field_index+1))
    return LI[CI[I] + CartesianIndex((0, 0, 0, 0, field_index))]
end

# add3(x1, x2, x3) = x1 + x2 + x3
add3(x1, x2, x3) = x1

function aos_cart_offset!(X, Y, us; nreps = 1, bm=nothing, n_trials = 30)
    if Y isa Array
        e = Inf
        CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us), 1))
        for t in 1:n_trials
            et = Base.@elapsed begin
                for i in 1:nreps
                    @inbounds @simd for I in 1:get_N(us)
                        CI1 = CI[I]
                        CI2 = CI1 + CartesianIndex((0, 0, 0, 0, 1))
                        CI3 = CI1 + CartesianIndex((0, 0, 0, 0, 2))
                        Y[CI1] = add3(X[CI1], X[CI2], X[CI3])
                    end
                end
            end
            e = min(e, et)
        end
    else
        e = Inf
        kernel = CUDA.@cuda always_inline = true launch = false aos_cart_offset_kernel!(X,Y,us)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(get_N(us), config.threads)
        blocks = cld(get_N(us), threads)
        for t in 1:n_trials
            et = CUDA.@elapsed begin
                for i in 1:nreps # reduce variance / impact of launch latency
                    kernel(X,Y,us; threads, blocks)
                end
            end
            e = min(e, et)
        end
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__), problem_size = size(us), n_reads_writes=4)
    return nothing
end;
function aos_cart_offset_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            n = (get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us), 1)
            CI1 = CartesianIndices(map(x -> Base.OneTo(x), n))[I]
            CI2 = CI1 + CartesianIndex((0, 0, 0, 0, 1))
            CI3 = CI1 + CartesianIndex((0, 0, 0, 0, 2))
            Y[CI1] = add3(X[CI1], X[CI2], X[CI3])
        end
    end
    return nothing
end;

function aos_lin_offset!(X, Y, us; nreps = 1, bm=nothing, n_trials = 30)
    if Y isa Array
        e = Inf
        for t in 1:n_trials
            et = Base.@elapsed begin
                for i in 1:nreps
                    @inbounds @simd for I in 1:get_N(us)
                        LY1 = const_linear_index(us, I, 0)
                        LX1 = const_linear_index(us, I, 0)
                        LX2 = const_linear_index(us, I, 1)
                        LX3 = const_linear_index(us, I, 2)
                        Y[LY1] = add3(X[LX1], X[LX2], X[LX3])
                    end
                end
            end
            e = min(e, et)
        end
    else
        e = Inf
        kernel = CUDA.@cuda always_inline = true launch = false aos_lin_offset_kernel!(X,Y,us)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(get_N(us), config.threads)
        blocks = cld(get_N(us), threads)
        for t in 1:n_trials
            et = CUDA.@elapsed begin
                for i in 1:nreps
                    kernel(X,Y,us; threads, blocks)
                end
            end
            e = min(e, et)
        end
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__), problem_size = size(us), n_reads_writes=4)
    return nothing
end;
function aos_lin_offset_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            LY1 = const_linear_index(us, I, 0)
            LX1 = const_linear_index(us, I, 0)
            LX2 = const_linear_index(us, I, 1)
            LX3 = const_linear_index(us, I, 2)
            Y[LY1] = add3(X[LX1], X[LX2], X[LX3])
        end
    end
    return nothing
end;

function soa_cart_index!(X, Y, us; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    if first(Y) isa Array
        CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us)))
        for t in 1:n_trials
            et = Base.@elapsed begin
                for i in 1:nreps
                    (y1,) = Y
                    (x1, x2, x3) = X
                    @inbounds @simd for I in 1:get_N(us)
                        y1[CI[I]] = add3(x1[CI[I]], x2[CI[I]], x3[CI[I]])
                    end
                end
            end
            e = min(e, et)
        end
    else
        kernel = CUDA.@cuda always_inline = true launch = false soa_cart_index_kernel!(X,Y,us)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(get_N(us), config.threads)
        blocks = cld(get_N(us), threads)
        for t in 1:n_trials
            et = CUDA.@elapsed begin
                for i in 1:nreps # reduce variance / impact of launch latency
                    kernel(X,Y,us; threads, blocks)
                end
            end
            e = min(e, et)
        end
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__), problem_size = size(us), n_reads_writes=4)
    return nothing
end;
function soa_cart_index_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), get_Nh(us)))
            (y1,) = Y
            (x1, x2, x3) = X
            y1[CI[I]] = add3(x1[CI[I]], x2[CI[I]], x3[CI[I]])
        end
    end
    return nothing
end;

function soa_linear_index!(X, Y, us; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    if first(Y) isa Array
        for t in 1:n_trials
            et = Base.@elapsed begin
                for i in 1:nreps
                    (y1,) = Y
                    (x1, x2, x3) = X
                    @inbounds @simd for I in 1:get_N(us)
                        y1[I] = add3(x1[I], x2[I], x3[I])
                    end
                end
            end
            e = min(e, et)
        end
    else
        kernel = CUDA.@cuda always_inline = true launch = false soa_linear_index_kernel!(X,Y,us)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(get_N(us), config.threads)
        blocks = cld(get_N(us), threads)
        for t in 1:n_trials
            et = CUDA.@elapsed begin
                for i in 1:nreps # reduce variance / impact of launch latency
                    kernel(X,Y,us; threads, blocks)
                end
            end
            e = min(e, et)
        end
    end
    push_info(bm; kernel_time_s=e/nreps, nreps, caller = @caller_name(@__FILE__), problem_size = size(us), n_reads_writes=4)
    return nothing
end;
function soa_linear_index_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            (y1,) = Y
            (x1, x2, x3) = X
            y1[I] = add3(x1[I], x2[I], x3[I])
        end
    end
    return nothing
end;

end # module

import .BenchmarkFieldLastIndex as FLD

function fill_with_rand!(arr)
    FT = eltype(arr)
    T = typeof(arr)
    s = size(arr)
    arr .= T(rand(FT, s))
end

using CUDA
using Test
@testset "Field last dim benchmark" begin
    bm = FLD.Benchmark(;problem_size=(63,4,4,5400,1), float_type=Float32) # size(problem_size, 4) == 1 to avoid double counting reads/writes
    ArrayType = CUDA.CuArray;
    # ArrayType = Base.identity;
    arr(float_type, problem_size, T) = T(zeros(float_type, problem_size...))

    s = (63,4,4,5400,3);
    sY = (63,4,4,5400,1);
    st = (63,4,4,5400);
    ndofs = prod(st);
    us = FLD.UniversalSizesStatic(s[1], s[2], s[end-1]);

    X_aos = arr(bm.float_type, s, ArrayType);
    Y_aos = arr(bm.float_type, sY, ArrayType);
    X_aos_ref = arr(bm.float_type, s, ArrayType);
    Y_aos_ref = arr(bm.float_type, sY, ArrayType);
    X_soa = ntuple(_ -> arr(bm.float_type, st, ArrayType), 3);
    Y_soa = ntuple(_ -> arr(bm.float_type, st, ArrayType), 1);
    fill_with_rand!(X_aos)
    fill_with_rand!(Y_aos)
    X_aos_ref .= X_aos
    Y_aos_ref .= Y_aos
    for i in 1:3; X_soa[i] .= X_aos[:,:,:,:, i]; end
    for i in 1:1; Y_soa[i] .= Y_aos[:,:,:,:, i]; end
    @info "ArrayType = $ArrayType"

    FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; n_trials = 1, nreps = 1)
    FLD.aos_lin_offset!(X_aos, Y_aos, us; n_trials = 1, nreps = 1)
    FLD.soa_linear_index!(X_soa, Y_soa, us; n_trials = 1, nreps = 1)

    @test all(X_aos .== X_aos_ref)
    @test all(Y_aos .== Y_aos_ref)
    for i in 1:3; @test all(X_soa[i] .== X_aos_ref[:,:,:,:,i]); end
    for i in 1:1; @test all(Y_soa[i] .== Y_aos_ref[:,:,:,:,i]); end

    FLD.soa_cart_index!(X_soa, Y_soa, us; n_trials = 1, nreps = 1)

    for i in 1:3; @test all(X_soa[i] .== X_aos_ref[:,:,:,:,i]); end
    for i in 1:1; @test all(Y_soa[i] .== Y_aos_ref[:,:,:,:,i]); end

    FLD.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100)
    FLD.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)
    FLD.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)
    FLD.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)

    FLD.tabulate_benchmark(bm)
end

# #! format: on
