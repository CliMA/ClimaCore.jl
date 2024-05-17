#=
julia --check-bounds=yes --project
using Revise; include(joinpath("test", "DataLayouts", "unit_struct.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: get_struct
using StaticArrays

function one_to_n(a::Array)
    for i in 1:length(a)
        a[i] = i
    end
    return a
end
one_to_n(s::Tuple, ::Type{FT}) where {FT} = one_to_n(zeros(FT, s...))
ncomponents(::Type{FT}, ::Type{S}) where {FT, S} = div(sizeof(S), sizeof(FT))
field_dim_to_one(s, dim) = Tuple(map(j -> j == dim ? 1 : s[j], 1:length(s)))
CI(s) = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))

struct Foo{T}
    x::T
    y::T
end

Base.zero(::Type{Foo{T}}) where {T} = Foo{T}(0, 0)

@testset "get_struct - IFH indexing" begin
    FT = Float64
    S = Foo{FT}
    s_array = (3, 2, 4)
    @test ncomponents(FT, S) == 2
    s = field_dim_to_one(s_array, 2)
    a = one_to_n(s_array, FT)
    @test get_struct(a, S, Val(2), CI(s)[1]) == Foo{FT}(1.0, 4.0)
    @test get_struct(a, S, Val(2), CI(s)[2]) == Foo{FT}(2.0, 5.0)
    @test get_struct(a, S, Val(2), CI(s)[3]) == Foo{FT}(3.0, 6.0)
    @test get_struct(a, S, Val(2), CI(s)[4]) == Foo{FT}(7.0, 10.0)
    @test get_struct(a, S, Val(2), CI(s)[5]) == Foo{FT}(8.0, 11.0)
    @test get_struct(a, S, Val(2), CI(s)[6]) == Foo{FT}(9.0, 12.0)
    @test get_struct(a, S, Val(2), CI(s)[7]) == Foo{FT}(13.0, 16.0)
    @test get_struct(a, S, Val(2), CI(s)[8]) == Foo{FT}(14.0, 17.0)
    @test get_struct(a, S, Val(2), CI(s)[9]) == Foo{FT}(15.0, 18.0)
    @test get_struct(a, S, Val(2), CI(s)[10]) == Foo{FT}(19.0, 22.0)
    @test get_struct(a, S, Val(2), CI(s)[11]) == Foo{FT}(20.0, 23.0)
    @test get_struct(a, S, Val(2), CI(s)[12]) == Foo{FT}(21.0, 24.0)
    @test_throws BoundsError get_struct(a, S, Val(2), CI(s)[13])
end

@testset "get_struct - IJF indexing" begin
    FT = Float64
    S = Foo{FT}
    s_array = (3, 4, 2)
    @test ncomponents(FT, S) == 2
    s = field_dim_to_one(s_array, 3)
    a = one_to_n(s_array, FT)
    @test get_struct(a, S, Val(3), CI(s)[1]) == Foo{FT}(1.0, 13.0)
    @test get_struct(a, S, Val(3), CI(s)[2]) == Foo{FT}(2.0, 14.0)
    @test get_struct(a, S, Val(3), CI(s)[3]) == Foo{FT}(3.0, 15.0)
    @test get_struct(a, S, Val(3), CI(s)[4]) == Foo{FT}(4.0, 16.0)
    @test get_struct(a, S, Val(3), CI(s)[5]) == Foo{FT}(5.0, 17.0)
    @test get_struct(a, S, Val(3), CI(s)[6]) == Foo{FT}(6.0, 18.0)
    @test get_struct(a, S, Val(3), CI(s)[7]) == Foo{FT}(7.0, 19.0)
    @test get_struct(a, S, Val(3), CI(s)[8]) == Foo{FT}(8.0, 20.0)
    @test get_struct(a, S, Val(3), CI(s)[9]) == Foo{FT}(9.0, 21.0)
    @test get_struct(a, S, Val(3), CI(s)[10]) == Foo{FT}(10.0, 22.0)
    @test get_struct(a, S, Val(3), CI(s)[11]) == Foo{FT}(11.0, 23.0)
    @test get_struct(a, S, Val(3), CI(s)[12]) == Foo{FT}(12.0, 24.0)
    @test_throws BoundsError get_struct(a, S, Val(3), CI(s)[13])
end

@testset "get_struct - VIJFH indexing" begin
    FT = Float64
    S = Foo{FT}
    s = (2, 2, 2, 2, 2)
    a = one_to_n(s, FT)
    @test ncomponents(FT, S) == 2

    @test get_struct(a, S, Val(4), CI(s)[1]) == Foo{FT}(1.0, 9.0)
    @test get_struct(a, S, Val(4), CI(s)[2]) == Foo{FT}(2.0, 10.0)
    @test get_struct(a, S, Val(4), CI(s)[3]) == Foo{FT}(3.0, 11.0)
    @test get_struct(a, S, Val(4), CI(s)[4]) == Foo{FT}(4.0, 12.0)
    @test get_struct(a, S, Val(4), CI(s)[5]) == Foo{FT}(5.0, 13.0)
    @test get_struct(a, S, Val(4), CI(s)[6]) == Foo{FT}(6.0, 14.0)
    @test get_struct(a, S, Val(4), CI(s)[7]) == Foo{FT}(7.0, 15.0)
    @test get_struct(a, S, Val(4), CI(s)[8]) == Foo{FT}(8.0, 16.0)
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[9])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[10])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[11])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[12])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[13])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[14])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[15])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[16])
    @test get_struct(a, S, Val(4), CI(s)[17]) == Foo{FT}(17.0, 25.0)
    @test get_struct(a, S, Val(4), CI(s)[18]) == Foo{FT}(18.0, 26.0)
    @test get_struct(a, S, Val(4), CI(s)[19]) == Foo{FT}(19.0, 27.0)
    @test get_struct(a, S, Val(4), CI(s)[20]) == Foo{FT}(20.0, 28.0)
    @test get_struct(a, S, Val(4), CI(s)[21]) == Foo{FT}(21.0, 29.0)
    @test get_struct(a, S, Val(4), CI(s)[22]) == Foo{FT}(22.0, 30.0)
    @test get_struct(a, S, Val(4), CI(s)[23]) == Foo{FT}(23.0, 31.0)
    @test get_struct(a, S, Val(4), CI(s)[24]) == Foo{FT}(24.0, 32.0)
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[25])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[26])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[27])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[28])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[29])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[30])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[31])
    @test_throws BoundsError get_struct(a, S, Val(4), CI(s)[32])
end

# TODO: add set_struct!
