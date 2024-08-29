#=
On A100

$ julia --project=.buildkite benchmarks/scripts/benchmark_IJFVH.jl 
==================================================================
Dimensions, NI = 4, NJ = 4, NF = 3, NV = 63, NH = 5300; FT = Float32
==================================================================
Test Summary:                                  | Pass  Total  Time
Sum test with IJFVH layout, cartesian indexing |    1      1  2.3s
==================================================================
Cartesian indexing benchmark for IJFVH:
==================================
min = TrialEstimate(62.149 μs)
median = TrialEstimate(67.139 μs)
mean = TrialEstimate(67.478 μs),
ntrials = 10000
---------------------------------------------
Test Summary:                               | Pass  Total  Time
Sum test with IJFVH layout, linear indexing |    1      1  0.0s
==================================================================
linear indexing benchmark for IJFVH:
==================================
min = TrialEstimate(61.269 μs)
median = TrialEstimate(67.650 μs)
mean = TrialEstimate(67.788 μs))
ntrials = 10000
---------------------------------------------
==================================================================
Dimensions, NI = 4, NJ = 4, NF = 3, NV = 63, NH = 5300; FT = Float64
==================================================================
Test Summary:                                  | Pass  Total  Time
Sum test with IJFVH layout, cartesian indexing |    1      1  1.3s
==================================================================
Cartesian indexing benchmark for IJFVH:
==================================
min = TrialEstimate(110.099 μs)
median = TrialEstimate(113.869 μs)
mean = TrialEstimate(114.574 μs),
ntrials = 10000
---------------------------------------------
Test Summary:                               | Pass  Total  Time
Sum test with IJFVH layout, linear indexing |    1      1  0.0s
==================================================================
linear indexing benchmark for IJFVH:
==================================
min = TrialEstimate(110.359 μs)
median = TrialEstimate(113.859 μs)
mean = TrialEstimate(116.783 μs))
ntrials = 10000
---------------------------------------------

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
# generate benchmarks
function generate_datalayout_benchmarks(::Type{DA}, ::Type{FT}) where {DA, FT}
    NI = NJ = 4 # polynomial order of approximation + 1
    NV = 63 # number of vertical levels
    NF = 3 # number of components in the velocity field (not # of fields)
    NH = 5300 # number of spectral elements

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
    return nothing
end

generate_datalayout_benchmarks(CuArray, Float32)
generate_datalayout_benchmarks(CuArray, Float64)
