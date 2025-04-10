#=
On A100

[skandala@clima ClimaCore.jl]$ julia --project=.buildkite benchmarks/scripts/benchmark_IJFVH.jl 
==================================================================
Dimensions, NI = 4, NJ = 4, NF = 3, NV = 63, NH = 5400; FT = Float32
==================================================================
Test Summary:                                  | Pass  Total  Time
Sum test with IJFVH layout, cartesian indexing |    1      1  2.0s
==================================================================
Cartesian indexing benchmark for IJFVH:
==================================
min = TrialEstimate(63.210 μs)
median = TrialEstimate(68.149 μs)
mean = TrialEstimate(68.127 μs),
ntrials = 10000
---------------------------------------------
Test Summary:                               | Pass  Total  Time
Sum test with IJFVH layout, linear indexing |    1      1  0.0s
==================================================================
linear indexing benchmark for IJFVH:
==================================
min = TrialEstimate(61.919 μs)
median = TrialEstimate(68.430 μs)
mean = TrialEstimate(68.769 μs))
ntrials = 10000
---------------------------------------------
----multiaddf_linear_IJVFH, FT = Float32-----------------------------------
emin = 57.231360115110874μs; emax = 57.55904130637646μs; nreps = 100, n_trials = 30
------------------------------------------------------------------------------------------
----multiaddf_cart_IJVFH, FT = Float32-------------------------------------
emin = 57.57952108979225μs; emax = 57.95839708298445μs; nreps = 100, n_trials = 30
------------------------------------------------------------------------------------------
==================================================================
Dimensions, NI = 4, NJ = 4, NF = 3, NV = 63, NH = 5400; FT = Float64
==================================================================
Test Summary:                                  | Pass  Total  Time
Sum test with IJFVH layout, cartesian indexing |    1      1  1.2s
==================================================================
Cartesian indexing benchmark for IJFVH:
==================================
min = TrialEstimate(112.079 μs)
median = TrialEstimate(115.559 μs)
mean = TrialEstimate(115.718 μs),
ntrials = 10000
---------------------------------------------
Test Summary:                               | Pass  Total  Time
Sum test with IJFVH layout, linear indexing |    1      1  0.0s
==================================================================
linear indexing benchmark for IJFVH:
==================================
min = TrialEstimate(112.009 μs)
median = TrialEstimate(115.289 μs)
mean = TrialEstimate(118.641 μs))
ntrials = 10000
---------------------------------------------
----multiaddf_linear_IJVFH, FT = Float64-----------------------------------
emin = 103.48544456064701μs; emax = 106.06592521071434μs; nreps = 100, n_trials = 30
------------------------------------------------------------------------------------------
----multiaddf_cart_IJVFH, FT = Float64-------------------------------------
emin = 104.05887849628925μs; emax = 105.89184239506721μs; nreps = 100, n_trials = 30
------------------------------------------------------------------------------------------
=#
using CUDA
using Statistics
using Test
using BenchmarkTools

max_threads_cuda() = 256
# Cartesian indexing
function addf_cart_IJFVH_kernel!(sum_a, a)
    (i, j, vid) = threadIdx()
    (bv, bh) = blockIdx()

    nvt = blockDim().z
    nv = size(a, 4)
    v = vid + (bv - 1) * nvt

    if v ≤ nv
        @inbounds sum_a[i, j, 1, v, bh] =
            a[i, j, 1, v, bh] + a[i, j, 2, v, bh] + a[i, j, 3, v, bh]
    end
    return nothing
end

# Add (3) components of a vector field in IJFVH layout using cartesian indexing
function addf_cart_IJFVH!(sum_A_IJFVH, A_IJFVH)
    (NI, NJ, NF, NV, NH) = size(A_IJFVH)
    NVT = min(Int(fld(max_threads_cuda(), NI * NJ)), NV)
    NBV = cld(NV, NVT)
    @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_cart_IJFVH_kernel!(
        sum_A_IJFVH,
        A_IJFVH,
    )
    return nothing
end

# Linear indexing
function addf_linear_IJFVH_kernel!(sum_a, a)
    (i, j, vid) = threadIdx()
    (bv, bh) = blockIdx()

    nvt = blockDim().z
    nbv = gridDim().x
    (Ni, Nj, nf, nv, _) = size(a)
    v = vid + (bv - 1) * nvt

    st_in = Ni * (Nj * (nv * (bh - 1) + (v - 1)) + (j - 1)) + i

    st = Ni * (Nj * nf * (nv * (bh - 1) + (v - 1)) + (j - 1)) + i
    stride = Ni * Nj
    if v ≤ nv
        @inbounds sum_a[st_in] =
            a[st] + a[st + stride] + a[st + stride + stride]
    end
    return nothing
end

# Add (3) components of a vector field in IJFVH layout using linear indexing
function addf_linear_IJFVH!(sum_A_IJFVH, A_IJFVH)
    (NI, NJ, NF, NV, NH) = size(A_IJFVH)
    NVT = min(Int(fld(max_threads_cuda(), NI * NJ)), NV)
    NBV = cld(NV, NVT)
    @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_linear_IJFVH_kernel!(
        sum_A_IJFVH,
        A_IJFVH,
    )
    return nothing
end

function multiaddf_linear_IJFVH!(
    sum_A_IJFVH,
    A_IJFVH;
    nreps = 100,
    n_trials = 30,
)
    (NI, NJ, NF, NV, NH) = size(A_IJFVH)
    NVT = min(Int(fld(max_threads_cuda(), NI * NJ)), NV)
    NBV = cld(NV, NVT)
    emin, emax = typemax(Float32), typemin(Float32)
    @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_linear_IJFVH_kernel!(
        sum_A_IJFVH,
        A_IJFVH,
    )

    for j in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps
                @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_linear_IJFVH_kernel!(
                    sum_A_IJFVH,
                    A_IJFVH,
                )
            end
        end
        emin = min(emin, et)
        emax = max(emax, et)
    end
    println(
        "----multiaddf_linear_IJVFH, FT = $(eltype(A_IJFVH))-----------------------------------",
    )
    println(
        "emin = $(emin*1e6/nreps)μs; emax = $(emax*1e6/nreps)μs; nreps = $nreps, n_trials = $(n_trials)",
    )
    println(
        "------------------------------------------------------------------------------------------",
    )
    return nothing
end

function multiaddf_cart_IJFVH!(sum_A_IJFVH, A_IJFVH; nreps = 100, n_trials = 30)
    (NI, NJ, NF, NV, NH) = size(A_IJFVH)
    NVT = min(Int(fld(max_threads_cuda(), NI * NJ)), NV)
    NBV = cld(NV, NVT)
    emin, emax = typemax(Float32), typemin(Float32)
    @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_cart_IJFVH_kernel!(
        sum_A_IJFVH,
        A_IJFVH,
    )
    emin, emax = typemax(Float32), typemin(Float32)
    for j in 1:n_trials
        et = CUDA.@elapsed begin
            for i in 1:nreps
                @cuda threads = (NI, NJ, NVT) blocks = (NBV, NH) addf_cart_IJFVH_kernel!(
                    sum_A_IJFVH,
                    A_IJFVH,
                )
            end
        end
        emin = min(emin, et)
        emax = max(emax, et)
    end
    println(
        "----multiaddf_cart_IJVFH, FT = $(eltype(A_IJFVH))-------------------------------------",
    )
    println(
        "emin = $(emin*1e6/nreps)μs; emax = $(emax*1e6/nreps)μs; nreps = $nreps, n_trials = $(n_trials)",
    )
    println(
        "------------------------------------------------------------------------------------------",
    )
    return nothing
end


# generate benchmarks
function generate_datalayout_benchmarks(::Type{DA}, ::Type{FT}) where {DA, FT}
    NI = NJ = 4 # polynomial order of approximation + 1
    NV = 63 # number of vertical levels
    NF = 3 # number of components in the velocity field (not # of fields)
    NH = 5400 # number of spectral elements

    println(
        "==================================================================",
    )
    println(
        "Dimensions, NI = $NI, NJ = $NJ, NF = $NF, NV = $NV, NH = $NH; FT = $FT",
    )
    println(
        "==================================================================",
    )
    # IJFVH layout
    A_IJFVH = DA(rand(FT, NI, NJ, NF, NV, NH))
    sum_A_IJFVH = DA{FT}(undef, NI, NJ, 1, NV, NH)
    sum_A_IJFVH_ref = sum(A_IJFVH, dims = 3)
    # use cartesian indexing for IJFVH layout
    addf_cart_IJFVH!(sum_A_IJFVH, A_IJFVH)

    @testset "Sum test with IJFVH layout, cartesian indexing" begin
        @test sum_A_IJFVH ≈ sum_A_IJFVH_ref
    end
    trial_cart_IJFVH =
        @benchmark CUDA.@sync addf_cart_IJFVH!($sum_A_IJFVH, $A_IJFVH)
    println(
        "==================================================================",
    )
    println("Cartesian indexing benchmark for IJFVH:
==================================
min = $(minimum(trial_cart_IJFVH))
median = $(Statistics.median(trial_cart_IJFVH))
mean = $(Statistics.mean(trial_cart_IJFVH)),
ntrials = $(length(trial_cart_IJFVH.times))")
    println("---------------------------------------------")

    sum_A_IJFVH .= FT(0)
    # use linear indexing for IJFVH layout
    addf_linear_IJFVH!(sum_A_IJFVH, A_IJFVH)
    @testset "Sum test with IJFVH layout, linear indexing" begin
        @test sum_A_IJFVH ≈ sum_A_IJFVH_ref
    end
    trial_linear_IJFVH =
        @benchmark CUDA.@sync addf_linear_IJFVH!($sum_A_IJFVH, $A_IJFVH)
    println(
        "==================================================================",
    )
    println("linear indexing benchmark for IJFVH:
==================================
min = $(minimum(trial_linear_IJFVH))
median = $(Statistics.median(trial_linear_IJFVH))
mean = $(Statistics.mean(trial_linear_IJFVH)))
ntrials = $(length(trial_linear_IJFVH.times))")
    println("---------------------------------------------")

    multiaddf_linear_IJFVH!(sum_A_IJFVH, A_IJFVH, nreps = 100, n_trials = 30)
    multiaddf_cart_IJFVH!(sum_A_IJFVH, A_IJFVH, nreps = 100, n_trials = 30)


    return nothing
end

generate_datalayout_benchmarks(CuArray, Float32)
generate_datalayout_benchmarks(CuArray, Float64)
