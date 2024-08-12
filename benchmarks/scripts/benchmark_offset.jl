#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "benchmark_offset.jl"))

# Info

 - This benchmark demos the performance for different offset styles:
  - Array of structs with Cartesian offsets
  - Array of structs with Linear offsets
  - Struct of arrays with no offsets

# Benchmark results:

Clima A100:
```
[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                              │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     BO.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 68 microseconds, 834 nanoseconds │ 57.7908 │ 1178.35     │ 4              │ 100    │
│     BO.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 58 microseconds, 153 nanoseconds │ 68.4046 │ 1394.77     │ 4              │ 100    │
│     BO.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 56 microseconds, 576 nanoseconds │ 70.3113 │ 1433.65     │ 4              │ 100    │
│     BO.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 67 microseconds, 185 nanoseconds │ 59.2089 │ 1207.27     │ 4              │ 100    │
└────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘

[ Info: ArrayType = CuArray
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────────────────────┬──────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                              │ time per call                    │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────────────────────┼──────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     BO.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100) │ 68 microseconds, 967 nanoseconds │ 57.6793 │ 1176.08     │ 4              │ 100    │
│     BO.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)          │ 58 microseconds, 82 nanoseconds  │ 68.489  │ 1396.49     │ 4              │ 100    │
│     BO.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)        │ 56 microseconds, 597 nanoseconds │ 70.2858 │ 1433.13     │ 4              │ 100    │
│     BO.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)          │ 67 microseconds, 288 nanoseconds │ 59.1188 │ 1205.43     │ 4              │ 100    │
└────────────────────────────────────────────────────────────────────┴──────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```
=#

#! format: off
module BenchmarkOffset

include("benchmark_utils.jl")

add3(x1, x2, x3) = x1 + x2 + x3

function aos_cart_offset!(X, Y, us; nreps = 1, bm=nothing, n_trials = 30)
    if Y isa Array
        e = Inf
        CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us)))
        for t in 1:n_trials
            et = Base.@elapsed begin
                for i in 1:nreps
                    @inbounds @simd for I in 1:get_N(us)
                        CI1 = CI[I]
                        CI2 = CI1 + CartesianIndex((0, 0, 0, 1, 0))
                        CI3 = CI1 + CartesianIndex((0, 0, 0, 2, 0))
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
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=4)
    return nothing
end;
function aos_cart_offset_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            n = (get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us))
            CI1 = CartesianIndices(map(x -> Base.OneTo(x), n))[I]
            CI2 = CI1 + CartesianIndex((0, 0, 0, 1, 0))
            CI3 = CI1 + CartesianIndex((0, 0, 0, 2, 0))
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
                        CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us)))
                        LI1 = LinearIndices((get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us)))
                        LI3 = LinearIndices((get_Nv(us), get_Nij(us), get_Nij(us), 3, get_Nh(us)))
                        CI1 = CI[I]
                        CI2 = CI1 + CartesianIndex((0, 0, 0, 1, 0))
                        CI3 = CI1 + CartesianIndex((0, 0, 0, 2, 0))
                        IY1 = LI1[CI1]
                        IX1 = LI3[CI1]
                        IX2 = LI3[CI2]
                        IX3 = LI3[CI3]
                        Y[IY1] = add3(X[IX1], X[IX2], X[IX3])
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
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=4)
    return nothing
end;
function aos_lin_offset_kernel!(X, Y, us)
    @inbounds begin
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            CI = CartesianIndices((get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us)))
            LI1 = LinearIndices((get_Nv(us), get_Nij(us), get_Nij(us), 1, get_Nh(us)))
            LI3 = LinearIndices((get_Nv(us), get_Nij(us), get_Nij(us), 3, get_Nh(us)))
            CI1 = CI[I]
            CI2 = CI1 + CartesianIndex((0, 0, 0, 1, 0))
            CI3 = CI1 + CartesianIndex((0, 0, 0, 2, 0))
            IY1 = LI1[CI1]
            IX1 = LI3[CI1]
            IX2 = LI3[CI2]
            IX3 = LI3[CI3]
            Y[IY1] = add3(X[IX1], X[IX2], X[IX3])
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
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=4)
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
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=4)
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

import .BenchmarkOffset as BO

function fill_with_rand!(arr)
    FT = eltype(arr)
    T = typeof(arr)
    s = size(arr)
    arr .= T(rand(FT, s))
end

using CUDA
using Test
@testset "Offset benchmark" begin
    bm = BO.Benchmark(;problem_size=(63,4,4,1,5400), float_type=Float32) # size(problem_size, 4) == 1 to avoid double counting reads/writes
    ArrayType = CUDA.CuArray;
    # ArrayType = Base.identity;
    arr(float_type, problem_size, T) = T(zeros(float_type, problem_size...))

    FT = Float64;
    s = (63,4,4,3,5400);
    sY = (63,4,4,1,5400);
    st = (63,4,4,5400);
    ndofs = prod(st);
    us = BO.UniversalSizesStatic(s[1], s[2], s[end]);

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
    for i in 1:3; X_soa[i] .= X_aos[:,:,:,i,:]; end
    for i in 1:1; Y_soa[i] .= Y_aos[:,:,:,i,:]; end
    @info "ArrayType = $ArrayType"

    BO.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; n_trials = 1, nreps = 1)
    BO.aos_lin_offset!(X_aos, Y_aos, us; n_trials = 1, nreps = 1)
    BO.soa_linear_index!(X_soa, Y_soa, us; n_trials = 1, nreps = 1)

    @test all(X_aos .== X_aos_ref)
    @test all(Y_aos .== Y_aos_ref)
    for i in 1:3; @test all(X_soa[i] .== X_aos_ref[:,:,:,i,:]); end
    for i in 1:1; @test all(Y_soa[i] .== Y_aos_ref[:,:,:,i,:]); end

    BO.soa_cart_index!(X_soa, Y_soa, us; n_trials = 1, nreps = 1)

    for i in 1:3; @test all(X_soa[i] .== X_aos_ref[:,:,:,i,:]); end
    for i in 1:1; @test all(Y_soa[i] .== Y_aos_ref[:,:,:,i,:]); end

    BO.aos_cart_offset!(X_aos_ref, Y_aos_ref, us; bm, nreps = 100)
    BO.aos_lin_offset!(X_aos, Y_aos, us; bm, nreps = 100)
    BO.soa_linear_index!(X_soa, Y_soa, us; bm, nreps = 100)
    BO.soa_cart_index!(X_soa, Y_soa, us; bm, nreps = 100)

    BO.tabulate_benchmark(bm)
end

# #! format: on
