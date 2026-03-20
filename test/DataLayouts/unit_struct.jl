using Test
using ClimaCore.DataLayouts: get_struct

struct Foo{T}
    x::T
    y::T
end

@testset "get_struct - IFH indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:24), 3, 2, 4)
    for I in CartesianIndices((3, 4))
        i = I[1] + 6 * (I[2] - 1)
        @test get_struct(a, S, I, Val(2)) == Foo{FT}(i, i + 3)
    end
end

@testset "get_struct - IJF indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:24), 3, 4, 2)
    for I in CartesianIndices((3, 4))
        i = I[1] + 3 * (I[2] - 1)
        @test get_struct(a, S, I, Val(3)) == Foo{FT}(i, i + 12)
    end
end

@testset "get_struct - VIJFH indexing" begin
    FT = Float64
    S = Foo{FT}
    a = reshape(FT.(1:32), 2, 2, 2, 2, 2)
    for I in CartesianIndices((2, 2, 2, 2))
        i = I[1] + 2 * (I[2] - 1) + 4 * (I[3] - 1) + 16 * (I[4] - 1)
        @test get_struct(a, S, I, Val(4)) == Foo{FT}(i, i + 8)
    end
end

# TODO: add set_struct!
