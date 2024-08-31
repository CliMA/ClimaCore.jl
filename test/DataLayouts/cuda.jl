#=
julia -g2 --check-bounds=yes --project
using Revise; include(joinpath("test", "DataLayouts", "cuda.jl"))
=#
using Test
using ClimaComms
using CUDA
ClimaComms.@import_required_backends
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: field_array
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
    src = IJFH{S, 4, Nh}(field_array(ArrayType(rand(4, 4, 3, Nh)), 3))
    dst = IJFH{S, 4, Nh}(field_array(ArrayType(zeros(4, 4, 3, Nh)), 3))

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
    data_arr1 = field_array(ArrayType(ones(FT, 2, 2, 3, Nh)), 3)
    data_arr2 = field_array(ArrayType(ones(FT, 2, 2, 1, Nh)), 3)
    data1 = IJFH{S1, 2, Nh}(data_arr1)
    data2 = IJFH{S2, 2, Nh}(data_arr2)

    f1(a1, a2) = a1.a.re * a2 + a1.b
    res = f1.(data1, data2)
    @test res isa IJFH{Float64}
    @test Array(parent(res)) == FT[2 for i in 1:2, j in 1:2, f in 1:1, h in 1:2]

    Nv = 33
    data_arr1 = ArrayType(ones(FT, Nv, 4, 4, 3, 2))
    data_arr2 = ArrayType(ones(FT, Nv, 4, 4, 1, 2))
    data1 = VIJFH{S1, Nv, 4, Nh}(data_arr1)
    data2 = VIJFH{S2, Nv, 4, Nh}(data_arr2)

    f2(a1, a2) = a1.a.re * a2 + a1.b
    res = f2.(data1, data2)
    @test res isa VIJFH{Float64, Nv}
    @test Array(parent(res)) ==
          FT[2 for v in 1:Nv, i in 1:4, j in 1:4, f in 1:1, h in 1:2]
end


@testset "broadcasting assignment from scalar" begin
    FT = Float64
    S = Complex{FT}
    Nh = 3
    device = ClimaComms.device()
    ArrayType = ClimaComms.array_type(device)
    data = IJFH{S, 2, Nh}(ArrayType{FT})
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) ==
          FT[f == 1 ? 1 : 2 for i in 1:2, j in 1:2, f in 1:2, h in 1:3]

    Nv = 33
    data = VIJFH{S, Nv, 4, Nh}(ArrayType{FT}(undef, Nv, 4, 4, 2, Nh))
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) == FT[
        f == 1 ? 1 : 2 for v in 1:Nv, i in 1:4, j in 1:4, f in 1:2, h in 1:3
    ]

    data = DataF{S}(ArrayType{FT})
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) == FT[f == 1 ? 1 : 2 for f in 1:2]
end
