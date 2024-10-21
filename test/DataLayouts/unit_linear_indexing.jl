#=
julia --check-bounds=yes --project
using Revise; include(joinpath("test", "DataLayouts", "unit_linear_indexing.jl"))
=#
using Test
using ClimaCore.DataLayouts
using ClimaCore.DataLayouts: get_struct_linear
import ClimaCore.Geometry
using StaticArrays
import Random
Random.seed!(1234)

import ClimaCore.DataLayouts as DL
field_dim_to_one(s, dim) = Tuple(map(j -> j == dim ? 1 : s[j], 1:length(s)))

Base.@propagate_inbounds cart_ind(n::NTuple, i::Integer) =
    @inbounds CartesianIndices(map(x -> Base.OneTo(x), n))[i]
Base.@propagate_inbounds linear_ind(n::NTuple, ci::CartesianIndex) =
    @inbounds LinearIndices(map(x -> Base.OneTo(x), n))[ci]
Base.@propagate_inbounds linear_ind(n::NTuple, loc::NTuple) =
    linear_ind(n, CartesianIndex(loc))

function debug_get_struct_linear(args...; expect_test_throws = false)
    if expect_test_throws
        get_struct_linear(args...)
    else
        try
            get_struct_linear(args...)
        catch
            get_struct_linear(args...)
        end
    end
end

function one_to_n(a::Array)
    for i in 1:length(a)
        a[i] = i
    end
    return a
end
one_to_n(s::Tuple, ::Type{FT}) where {FT} = one_to_n(zeros(FT, s...))
ncomponents(::Type{FT}, ::Type{S}) where {FT, S} = div(sizeof(S), sizeof(FT))

struct Foo{T}
    x::T
    y::T
end

Base.zero(::Type{Foo{T}}) where {T} = Foo{T}(0, 0)

@testset "get_struct - IHF indexing (float)" begin
    FT = Float64
    S = FT
    s_array = (3, 4, 1)
    @test ncomponents(FT, S) == 1
    a = one_to_n(s_array, FT)
    ss = s_array
    @test debug_get_struct_linear(a, S, 1, ss) == 1.0
    @test debug_get_struct_linear(a, S, 2, ss) == 2.0
    @test debug_get_struct_linear(a, S, 3, ss) == 3.0
    @test debug_get_struct_linear(a, S, 4, ss) == 4.0
    @test debug_get_struct_linear(a, S, 5, ss) == 5.0
    @test debug_get_struct_linear(a, S, 6, ss) == 6.0
    @test debug_get_struct_linear(a, S, 7, ss) == 7.0
    @test debug_get_struct_linear(a, S, 8, ss) == 8.0
    @test debug_get_struct_linear(a, S, 9, ss) == 9.0
    @test debug_get_struct_linear(a, S, 10, ss) == 10.0
    @test debug_get_struct_linear(a, S, 11, ss) == 11.0
    @test debug_get_struct_linear(a, S, 12, ss) == 12.0
    @test_throws BoundsError debug_get_struct_linear(
        a,
        S,
        13,
        ss;
        expect_test_throws = true,
    )
end

@testset "get_struct - IHF indexing" begin
    FT = Float64
    S = Foo{FT}
    s_array = (3, 4, 2)
    @test ncomponents(FT, S) == 2
    a = one_to_n(s_array, FT)
    ss = s_array
    @test debug_get_struct_linear(a, S, 1, ss) == Foo{FT}(1.0, 13.0)
    @test debug_get_struct_linear(a, S, 2, ss) == Foo{FT}(2.0, 14.0)
    @test debug_get_struct_linear(a, S, 3, ss) == Foo{FT}(3.0, 15.0)
    @test debug_get_struct_linear(a, S, 4, ss) == Foo{FT}(4.0, 16.0)
    @test debug_get_struct_linear(a, S, 5, ss) == Foo{FT}(5.0, 17.0)
    @test debug_get_struct_linear(a, S, 6, ss) == Foo{FT}(6.0, 18.0)
    @test debug_get_struct_linear(a, S, 7, ss) == Foo{FT}(7.0, 19.0)
    @test debug_get_struct_linear(a, S, 8, ss) == Foo{FT}(8.0, 20.0)
    @test debug_get_struct_linear(a, S, 9, ss) == Foo{FT}(9.0, 21.0)
    @test debug_get_struct_linear(a, S, 10, ss) == Foo{FT}(10.0, 22.0)
    @test debug_get_struct_linear(a, S, 11, ss) == Foo{FT}(11.0, 23.0)
    @test debug_get_struct_linear(a, S, 12, ss) == Foo{FT}(12.0, 24.0)
    @test_throws BoundsError debug_get_struct_linear(
        a,
        S,
        13,
        ss;
        expect_test_throws = true,
    )
end

@testset "get_struct - IJF indexing" begin
    FT = Float64
    S = Foo{FT}
    s_array = (3, 4, 2)
    @test ncomponents(FT, S) == 2
    s = field_dim_to_one(s_array, 3)
    a = one_to_n(s_array, FT)
    ss = s_array
    @test debug_get_struct_linear(a, S, 1, ss) == Foo{FT}(1.0, 13.0)
    @test debug_get_struct_linear(a, S, 2, ss) == Foo{FT}(2.0, 14.0)
    @test debug_get_struct_linear(a, S, 3, ss) == Foo{FT}(3.0, 15.0)
    @test debug_get_struct_linear(a, S, 4, ss) == Foo{FT}(4.0, 16.0)
    @test debug_get_struct_linear(a, S, 5, ss) == Foo{FT}(5.0, 17.0)
    @test debug_get_struct_linear(a, S, 6, ss) == Foo{FT}(6.0, 18.0)
    @test debug_get_struct_linear(a, S, 7, ss) == Foo{FT}(7.0, 19.0)
    @test debug_get_struct_linear(a, S, 8, ss) == Foo{FT}(8.0, 20.0)
    @test debug_get_struct_linear(a, S, 9, ss) == Foo{FT}(9.0, 21.0)
    @test debug_get_struct_linear(a, S, 10, ss) == Foo{FT}(10.0, 22.0)
    @test debug_get_struct_linear(a, S, 11, ss) == Foo{FT}(11.0, 23.0)
    @test debug_get_struct_linear(a, S, 12, ss) == Foo{FT}(12.0, 24.0)
    @test_throws BoundsError debug_get_struct_linear(
        a,
        S,
        13,
        ss;
        expect_test_throws = true,
    )
end

@testset "get_struct - VIJHF indexing" begin
    FT = Float64
    S = Foo{FT}
    s_array = (2, 2, 2, 2, 2)
    @test ncomponents(FT, S) == 2
    s = field_dim_to_one(s_array, 5)
    a = one_to_n(s_array, FT)
    ss = s_array

    @test debug_get_struct_linear(a, S, 1, ss) == Foo{FT}(1.0, 17.0)
    @test debug_get_struct_linear(a, S, 2, ss) == Foo{FT}(2.0, 18.0)
    @test debug_get_struct_linear(a, S, 3, ss) == Foo{FT}(3.0, 19.0)
    @test debug_get_struct_linear(a, S, 4, ss) == Foo{FT}(4.0, 20.0)
    @test debug_get_struct_linear(a, S, 5, ss) == Foo{FT}(5.0, 21.0)
    @test debug_get_struct_linear(a, S, 6, ss) == Foo{FT}(6.0, 22.0)
    @test debug_get_struct_linear(a, S, 7, ss) == Foo{FT}(7.0, 23.0)
    @test debug_get_struct_linear(a, S, 8, ss) == Foo{FT}(8.0, 24.0)
    @test debug_get_struct_linear(a, S, 9, ss) == Foo{FT}(9.0, 25.0)
    @test debug_get_struct_linear(a, S, 10, ss) == Foo{FT}(10.0, 26.0)
    @test debug_get_struct_linear(a, S, 11, ss) == Foo{FT}(11.0, 27.0)
    @test debug_get_struct_linear(a, S, 12, ss) == Foo{FT}(12.0, 28.0)
    @test debug_get_struct_linear(a, S, 13, ss) == Foo{FT}(13.0, 29.0)
    @test debug_get_struct_linear(a, S, 14, ss) == Foo{FT}(14.0, 30.0)
    @test debug_get_struct_linear(a, S, 15, ss) == Foo{FT}(15.0, 31.0)
    @test debug_get_struct_linear(a, S, 16, ss) == Foo{FT}(16.0, 32.0)

    @test_throws BoundsError debug_get_struct_linear(
        a,
        S,
        17,
        ss;
        expect_test_throws = true,
    )
end

# # TODO: add set_struct!
