using Test
using CUDA
using ClimateMachineCore.DataLayouts

function knl_copy!(dst, src)
    i = threadIdx().x
    j = threadIdx().y

    h = blockIdx().x

    p_dst = slab(dst, h)
    p_src = slab(src, h)

    @inbounds p_dst[i, j] = p_src[i, j]
    return nothing
end

function test_copy!(dst, src)
    @cuda threads = (4, 4) blocks = (10,) knl_copy!(dst, src)
end

@testset "data in GPU kernels" begin

    S = Tuple{Complex{Float64}, Float64}
    src = IJFH{S, 4}(CuArray(rand(4, 4, 3, 10)))
    dst = IJFH{S, 4}(CuArray(zeros(4, 4, 3, 10)))

    test_copy!(dst, src)

    @test getfield(dst, :array) == getfield(src, :array)
end

@testset "broadcasting" begin
    FT = Float64
    S1 = NamedTuple{(:a, :b), Tuple{Complex{Float64}, Float64}}
    data1 = CuArray(ones(FT, 2, 2, 3, 2))
    S2 = Float64
    data2 = CuArray(ones(FT, 2, 2, 1, 2))
    data1 = IJFH{S1, 2}(data1)
    data2 = IJFH{S2, 2}(data2)

    f(a1, a2) = a1.a.re * a2 + a1.b
    res = f.(data1, data2)
    @test res isa IJFH{Float64}
    @test Array(parent(res)) == FT[2 for i in 1:2, j in 1:2, f in 1:1, h in 1:2]
end

@testset "broadcasting assignment from scalar" begin
    FT = Float64
    S = Complex{FT}
    data = IJFH{S, 2}(CuArray{FT}, 3)
    data .= Complex(1.0, 2.0)
    @test Array(parent(data)) ==
          FT[f == 1 ? 1 : 2 for i in 1:2, j in 1:2, f in 1:2, h in 1:3]
end
