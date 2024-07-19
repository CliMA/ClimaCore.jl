#=
julia --check-bounds=yes --project
using Revise; include(joinpath("test", "DataLayouts", "unit_struct.jl"))
=#
using Test
using ClimaCore.DataLayouts
using StaticArrays

function one_to_n(a::Array)
    for i in 1:length(a)
        a[i] = i
    end
    return a
end
one_to_n(s::Tuple, ::Type{FT}) where {FT} = one_to_n(zeros(FT, s...))
ncomponents(::Type{FT}, ::Type{S}) where {FT, S} = div(sizeof(S), sizeof(FT))

function test_get_struct(::Type{FT}, ::Type{S}) where {FT, S}
    s = (2,)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    for (i, ci) in enumerate(CI)
        for j in 1:length(s)
            @test DataLayouts.get_struct(a, S, Val(j), ci) == FT(i)
        end
    end

    s = (2, 3)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    for (i, ci) in enumerate(CI)
        for j in 1:length(s)
            @test DataLayouts.get_struct(a, S, Val(j), ci) == FT(i)
        end
    end

    s = (2, 3, 4)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    for (i, ci) in enumerate(CI)
        for j in 1:length(s)
            @test DataLayouts.get_struct(a, S, Val(j), ci) == FT(i)
        end
    end

    s = (2, 3, 4, 5)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    for (i, ci) in enumerate(CI)
        for j in 1:length(s)
            @test DataLayouts.get_struct(a, S, Val(j), ci) == FT(i)
        end
    end
end

@testset "get_struct - Float" begin
    test_get_struct(Float64, Float64)
    test_get_struct(Float32, Float32)
end

struct Foo{T}
    x::T
    y::T
end

Base.zero(::Type{Foo{T}}) where {T} = Foo{T}(0, 0)

@testset "get_struct - flat struct 2-fields 1-dim" begin
    FT = Float64
    S = Foo{FT}
    s = (4,)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    @test ncomponents(FT, S) == 2
    @test DataLayouts.get_struct(a, S, Val(1), CI[1]) == Foo{FT}(1.0, 2.0)
    @test DataLayouts.get_struct(a, S, Val(1), CI[2]) == Foo{FT}(2.0, 3.0)
    @test DataLayouts.get_struct(a, S, Val(1), CI[3]) == Foo{FT}(3.0, 4.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(1), CI[4])
end

@testset "get_struct - flat struct 2-fields 3-dims" begin
    FT = Float64
    S = Foo{FT}
    s = (2, 3, 4)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    @test ncomponents(FT, S) == 2

    # Call get_struct, and span `a` (access elements to 24.0):
    @test DataLayouts.get_struct(a, S, Val(1), CI[1]) == Foo{FT}(1.0, 2.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(1), CI[2])

    @test DataLayouts.get_struct(a, S, Val(2), CI[1]) == Foo{FT}(1.0, 3.0)
    @test DataLayouts.get_struct(a, S, Val(2), CI[2]) == Foo{FT}(2.0, 4.0)
    @test DataLayouts.get_struct(a, S, Val(2), CI[3]) == Foo{FT}(3.0, 5.0)
    @test DataLayouts.get_struct(a, S, Val(2), CI[4]) == Foo{FT}(4.0, 6.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(2), CI[5])

    @test DataLayouts.get_struct(a, S, Val(3), CI[1]) == Foo{FT}(1.0, 7.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[2]) == Foo{FT}(2.0, 8.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[3]) == Foo{FT}(3.0, 9.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[4]) == Foo{FT}(4.0, 10.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[5]) == Foo{FT}(5.0, 11.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[6]) == Foo{FT}(6.0, 12.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[7]) == Foo{FT}(7.0, 13.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[8]) == Foo{FT}(8.0, 14.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[9]) == Foo{FT}(9.0, 15.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[10]) == Foo{FT}(10.0, 16.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[11]) == Foo{FT}(11.0, 17.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[12]) == Foo{FT}(12.0, 18.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[13]) == Foo{FT}(13.0, 19.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[14]) == Foo{FT}(14.0, 20.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[15]) == Foo{FT}(15.0, 21.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[16]) == Foo{FT}(16.0, 22.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[17]) == Foo{FT}(17.0, 23.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[18]) == Foo{FT}(18.0, 24.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(3), CI[19])
end

@testset "get_struct - flat struct 2-fields 5-dims" begin
    FT = Float64
    S = Foo{FT}
    s = (2, 2, 2, 2, 2)
    a = one_to_n(s, FT)
    CI = CartesianIndices(map(ξ -> Base.OneTo(ξ), s))
    @test ncomponents(FT, S) == 2

    # Call get_struct, and span `a` (access elements to 2^5 = 32.0):
    @test DataLayouts.get_struct(a, S, Val(1), CI[1]) == Foo{FT}(1.0, 2.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(1), CI[2])

    @test DataLayouts.get_struct(a, S, Val(2), CI[1]) == Foo{FT}(1.0, 3.0)
    @test DataLayouts.get_struct(a, S, Val(2), CI[2]) == Foo{FT}(2.0, 4.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(2), CI[3])

    @test DataLayouts.get_struct(a, S, Val(3), CI[1]) == Foo{FT}(1.0, 5.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[2]) == Foo{FT}(2.0, 6.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[3]) == Foo{FT}(3.0, 7.0)
    @test DataLayouts.get_struct(a, S, Val(3), CI[4]) == Foo{FT}(4.0, 8.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(3), CI[5])

    # VIJFH
    @test DataLayouts.get_struct(a, S, Val(4), CI[1]) == Foo{FT}(1.0, 9.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[2]) == Foo{FT}(2.0, 10.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[3]) == Foo{FT}(3.0, 11.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[4]) == Foo{FT}(4.0, 12.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[5]) == Foo{FT}(5.0, 13.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[6]) == Foo{FT}(6.0, 14.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[7]) == Foo{FT}(7.0, 15.0)
    @test DataLayouts.get_struct(a, S, Val(4), CI[8]) == Foo{FT}(8.0, 16.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(4), CI[9])

    @test DataLayouts.get_struct(a, S, Val(5), CI[1]) == Foo{FT}(1.0, 17.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[2]) == Foo{FT}(2.0, 18.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[3]) == Foo{FT}(3.0, 19.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[4]) == Foo{FT}(4.0, 20.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[5]) == Foo{FT}(5.0, 21.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[6]) == Foo{FT}(6.0, 22.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[7]) == Foo{FT}(7.0, 23.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[8]) == Foo{FT}(8.0, 24.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[9]) == Foo{FT}(9.0, 25.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[10]) == Foo{FT}(10.0, 26.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[11]) == Foo{FT}(11.0, 27.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[12]) == Foo{FT}(12.0, 28.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[13]) == Foo{FT}(13.0, 29.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[14]) == Foo{FT}(14.0, 30.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[15]) == Foo{FT}(15.0, 31.0)
    @test DataLayouts.get_struct(a, S, Val(5), CI[16]) == Foo{FT}(16.0, 32.0)
    @test_throws BoundsError DataLayouts.get_struct(a, S, Val(5), CI[17])
end

# TODO: add set_struct!
