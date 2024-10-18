#=
julia -g2 --check-bounds=yes --project=test
using Revise; include(joinpath("test", "DataLayouts", "cuda.jl"))
=#
using Test
using ClimaComms
using CUDA
ClimaComms.@import_required_backends
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: slab_index

function knl_copy!(dst, src)
    i = threadIdx().x
    j = threadIdx().y

    h = blockIdx().x

    p_dst = slab(dst, h)
    p_src = slab(src, h)

    @inbounds p_dst[slab_index(i, j)] = p_src[slab_index(i, j)]
    return nothing
end

function test_copy!(dst, src)
    CUDA.@cuda threads = (4, 4) blocks = (10,) knl_copy!(dst, src)
end

@testset "data in GPU kernels" begin

    S = Tuple{Complex{Float64}, Float64}
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    Nh = 10
    src = IJHF{S}(ArrayType{Float64}, rand; Nij = 4, Nh)
    dst = IJHF{S}(ArrayType{Float64}, zeros; Nij = 4, Nh)

    test_copy!(dst, src)

    @test getfield(dst, :array) == getfield(src, :array)
end

@testset "broadcasting" begin
    FT = Float64
    S1 = NamedTuple{(:a, :b), Tuple{Complex{Float64}, Float64}}
    S2 = Float64
    Nh = 2
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    data1 = IJHF{S1}(ArrayType{FT}, ones; Nij = 2, Nh)
    data2 = IJHF{S2}(ArrayType{FT}, ones; Nij = 2, Nh)

    f1(a1, a2) = a1.a.re * a2 + a1.b
    res = f1.(data1, data2)
    @test res isa IJHF{Float64}
    @test Array(parent(res)) == FT[2 for i in 1:2, j in 1:2, h in 1:2, f in 1:1]

    Nv = 33
    data1 = VIJHF{S1}(ArrayType{FT}, ones; Nv, Nij = 4, Nh = 2)
    data2 = VIJHF{S2}(ArrayType{FT}, ones; Nv, Nij = 4, Nh = 2)

    f2(a1, a2) = a1.a.re * a2 + a1.b
    res = f2.(data1, data2)
    @test res isa VIJHF{Float64, Nv}
    @test Array(parent(res)) ==
          FT[2 for v in 1:Nv, i in 1:4, j in 1:4, h in 1:2, f in 1:1]
end


@testset "broadcasting assignment from scalar" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    data = IJHF{S}(ArrayType{FT}; Nij = 2, Nh)
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) ==
          FT[f == 1 ? 1 : 2 for i in 1:2, j in 1:2, h in 1:3, f in 1:2]

    Nv = 33
    data = VIJHF{S}(ArrayType{FT}; Nv, Nij = 4, Nh)
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) == FT[
        f == 1 ? 1 : 2 for v in 1:Nv, i in 1:4, j in 1:4, h in 1:3, f in 1:2
    ]

    data = DataF{S}(ArrayType{FT})
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) == FT[f == 1 ? 1 : 2 for f in 1:2]
end
