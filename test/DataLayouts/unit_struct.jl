using Test
using ClimaCore.DataLayouts: get_struct

struct Foo{T}
    x::T
    y::T
end

@testset "Cartesian indexing" begin
    T = Foo{Float64}

    a = reshape(Float64.(1:24), 3, 2, 4)
    for I in CartesianIndices((3, 4))
        i = I[1] + 6 * (I[2] - 1)
        @test get_struct(a, T, I, Val(2)) == T(i, i + 3)
    end
    @test_throws BoundsError get_struct(a, T, CartesianIndex(4, 3), Val(2))

    a = reshape(Float64.(1:32), 2, 2, 2, 2, 2)
    for I in CartesianIndices((2, 2, 2, 2))
        i = I[1] + 2 * (I[2] - 1) + 4 * (I[3] - 1) + 16 * (I[4] - 1)
        @test get_struct(a, T, I, Val(4)) == T(i, i + 8)
    end
    @test_throws BoundsError get_struct(a, T, CartesianIndex(1, 1, 1, 3), Val(4))
end

@testset "Linear and Cartesian indexing" begin
    T = Foo{Float64}

    a = reshape(Float64.(1:24), 3, 4, 2)
    for I in CartesianIndices((3, 4))
        i = I[1] + 3 * (I[2] - 1)
        @test get_struct(a, T, i, Val(3)) == T(i, i + 12)
        @test get_struct(a, T, I, Val(3)) == T(i, i + 12)
    end
    @test_throws BoundsError get_struct(a, T, 13, Val(3))
    @test_throws BoundsError get_struct(a, T, CartesianIndex(4, 3), Val(3))

    a = reshape(Float64.(1:32), 2, 2, 2, 2, 2)
    for I in CartesianIndices((2, 2, 2, 2))
        i = I[1] + 2 * (I[2] - 1) + 4 * (I[3] - 1) + 8 * (I[4] - 1)
        @test get_struct(a, T, i, Val(5)) == T(i, i + 16)
        @test get_struct(a, T, I, Val(5)) == T(i, i + 16)
    end
    @test_throws BoundsError get_struct(a, T, 17, Val(5))
    @test_throws BoundsError get_struct(a, T, CartesianIndex(1, 1, 1, 3), Val(5))
end

# TODO: add set_struct!
