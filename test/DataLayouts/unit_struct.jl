using Test
using ClimaCore.DataLayouts: get_struct, set_struct!, struct_field_view, check_basetype

struct Foo{T}
    x::T
    y::T
end

struct TrailingPadding
    x::Float64
    y::Float32
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

@testset "struct_field_view with padded structs" begin
    T = TrailingPadding

    # The 12 bytes of field data are padded to a size of 16 bytes, so that the
    # last of the 4 Float32 entries used to store a value is a padding byte.
    @test sizeof(T) == 16

    # The last field is not stored at the end of the Float32 entries, since the
    # size of Tuple{Float64, Float32} includes 4 bytes of trailing padding. The
    # field needs to be located using fieldoffset, which skips over the padding.
    @test sizeof(Tuple{fieldtypes(T)...}) ÷ sizeof(Float32) == 4
    @test Int(fieldoffset(T, 2)) ÷ sizeof(Float32) + 1 == 3

    a = set_struct!(zeros(Float32, 4, 2), T(1.0, 2.0f0), 1, Val(1))
    @test get_struct(struct_field_view(a, T, Val(1), Val(1)), Float64, 1, Val(1)) == 1.0
    @test get_struct(struct_field_view(a, T, Val(2), Val(1)), Float32, 1, Val(1)) == 2.0f0
end

# TODO: add set_struct!

@testset "check_basetype" begin
    @test_throws Exception check_basetype(Real, Real)
    @test_throws Exception check_basetype(Real, Float64)
    @test_throws Exception check_basetype(Float64, Real)

    @test isnothing(check_basetype(Float64, Float64))
    @test isnothing(check_basetype(Float32, Float64))
    @test_throws Exception check_basetype(Float64, Float32)

    @test isnothing(check_basetype(Tuple{}, Tuple{}))
    @test isnothing(check_basetype(Float64, Tuple{}))
    @test_throws Exception check_basetype(Tuple{}, Float64)

    S = typeof((a = ((1.0, 2.0f0), (3.0, 4.0f0)), b = (5.0, 6.0f0)))
    @test isnothing(check_basetype(Float32, S))
    @test isnothing(check_basetype(Float64, S))
    @test isnothing(check_basetype(Tuple{Float64, Float32}, S))
    @test_throws Exception check_basetype(NTuple{4, Float64}, S)

    S = typeof(((), (1.0 + 2.0im, NamedTuple()), 3.0 + 4.0im, ()))
    @test isnothing(check_basetype(Float32, S))
    @test isnothing(check_basetype(Float64, S))
    @test isnothing(check_basetype(Complex{Float64}, S))
    @test_throws Exception check_basetype(NTuple{5, Float64}, S)
end
