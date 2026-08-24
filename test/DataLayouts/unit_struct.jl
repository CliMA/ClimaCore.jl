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

@testset "Cartesian indexing [$FT]" for FT in (Float32, Float64)
    T = Foo{FT}

    a = reshape(FT.(1:24), 3, 2, 4)
    for I in CartesianIndices((3, 4))
        i = I[1] + 6 * (I[2] - 1)
        @test get_struct(a, T, I, Val(2)) == T(FT(i), FT(i + 3))
    end
    @test_throws BoundsError get_struct(a, T, CartesianIndex(4, 3), Val(2))

    a = reshape(FT.(1:32), 2, 2, 2, 2, 2)
    for I in CartesianIndices((2, 2, 2, 2))
        i = I[1] + 2 * (I[2] - 1) + 4 * (I[3] - 1) + 16 * (I[4] - 1)
        @test get_struct(a, T, I, Val(4)) == T(FT(i), FT(i + 8))
    end
    @test_throws BoundsError get_struct(a, T, CartesianIndex(1, 1, 1, 3), Val(4))
end

@testset "Linear and Cartesian indexing [$FT]" for FT in (Float32, Float64)
    T = Foo{FT}

    a = reshape(FT.(1:24), 3, 4, 2)
    for I in CartesianIndices((3, 4))
        i = I[1] + 3 * (I[2] - 1)
        @test get_struct(a, T, i, Val(3)) == T(FT(i), FT(i + 12))
        @test get_struct(a, T, I, Val(3)) == T(FT(i), FT(i + 12))
    end
    @test_throws BoundsError get_struct(a, T, 13, Val(3))
    @test_throws BoundsError get_struct(a, T, CartesianIndex(4, 3), Val(3))

    a = reshape(FT.(1:32), 2, 2, 2, 2, 2)
    for I in CartesianIndices((2, 2, 2, 2))
        i = I[1] + 2 * (I[2] - 1) + 4 * (I[3] - 1) + 8 * (I[4] - 1)
        @test get_struct(a, T, i, Val(5)) == T(FT(i), FT(i + 16))
        @test get_struct(a, T, I, Val(5)) == T(FT(i), FT(i + 16))
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

@testset "set_struct! [$FT]" for FT in (Float32, Float64)
    T = Foo{FT}

    # 1. 2D Array with F along dim 1
    a = zeros(FT, 2, 4)
    val1 = T(FT(10.0), FT(20.0))
    val2 = T(FT(30.0), FT(40.0))
    set_struct!(a, val1, CartesianIndex(1), Val(1))
    set_struct!(a, val2, CartesianIndex(3), Val(1))
    @test get_struct(a, T, CartesianIndex(1), Val(1)) == val1
    @test get_struct(a, T, CartesianIndex(3), Val(1)) == val2
    @test_throws BoundsError set_struct!(a, val1, CartesianIndex(5), Val(1))

    # 2. 3D Array with Cartesian indexing and Val(F)
    b = zeros(FT, 3, 2, 4)
    val_b = T(FT(5.0), FT(15.0))
    set_struct!(b, val_b, CartesianIndex(2, 3), Val(2))
    @test get_struct(b, T, CartesianIndex(2, 3), Val(2)) == val_b
    @test_throws BoundsError set_struct!(b, val_b, CartesianIndex(4, 3), Val(2))
    @test_throws BoundsError set_struct!(b, val_b, CartesianIndex(1, 5), Val(2))

    # 3. 5D Array with Cartesian indexing along Val(4)
    c = zeros(FT, 2, 2, 2, 2, 2)
    val_c = T(FT(7.0), FT(14.0))
    I_target = CartesianIndex(1, 2, 1, 2)
    set_struct!(c, val_c, I_target, Val(4))
    @test get_struct(c, T, I_target, Val(4)) == val_c
    @test_throws BoundsError set_struct!(c, val_c, CartesianIndex(1, 1, 1, 3), Val(4))

    # 4. Linear indexing with stride
    d = zeros(FT, 3, 4)
    val_d = T(FT(100.0), FT(200.0))
    # Stride of 3 corresponds to the first dimension size
    set_struct!(d, val_d, 2, 3)
    @test get_struct(d, T, 2, 3) == val_d
    @test_throws BoundsError set_struct!(d, val_d, 12, 3)

    # 5. Nested and composite structs roundtrip
    NestedT = typeof((a = (FT(1.0), FT(2.0)), b = (FT(3.0), FT(4.0))))
    arr_nested = zeros(FT, 4, 3)
    nested_val = (a = (FT(11.0), FT(22.0)), b = (FT(33.0), FT(44.0)))
    set_struct!(arr_nested, nested_val, CartesianIndex(2), Val(1))
    @test get_struct(arr_nested, NestedT, CartesianIndex(2), Val(1)) == nested_val

    # 6. Single value vector (no index specified)
    arr_single = zeros(Int8, 4)
    set_struct!(arr_single, Int32(42))
    @test get_struct(arr_single, Int32) == Int32(42)
end

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
