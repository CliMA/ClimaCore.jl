# These tests require running with `--check-bounds=[auto|no]`
using Test
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf, half
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry

using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

include(joinpath(@__DIR__, "util_spaces.jl"))

# https://github.com/CliMA/ClimaCore.jl/issues/946
@testset "Allocations with broadcasting Refs" begin
    FT = Float64
    function foo!(Yx::Fields.Field)
        Yx .= Ref(1) .+ Yx
        return nothing
    end
    function foocolumn!(Yx::Fields.Field)
        Fields.bycolumn(axes(Yx)) do colidx
            Yx[colidx] .= Ref(1) .+ Yx[colidx]
            nothing
        end
        return nothing
    end
    for space in all_spaces(FT)
        bycolumnable(space) || continue
        Y = FieldFromNamedTuple(space, (; x = FT(2)))

        # Plain broadcast
        Yx = Y.x
        foo!(Yx) # compile first
        p = @allocated foo!(Yx)
        @test p == 0

        # bycolumn
        foocolumn!(Yx) # compile first
        p = @allocated foocolumn!(Yx)
        @test p == 0
    end
end

# https://github.com/CliMA/ClimaCore.jl/issues/949
@testset "Allocations with getproperty on FieldVectors" begin
    FT = Float64
    function allocs_test!(Y)
        x = Y.x
        fill!(x, 2.0)
        nothing
    end
    function callfill!(Y)
        fill!(Y, Ref((; x = 2.0)))
        nothing
    end
    for space in all_spaces(FT)
        Y = FieldFromNamedTuple(space, (; x = FT(2)))
        allocs_test!(Y)
        p = @allocated allocs_test!(Y)
        @test p == 0

        callfill!(Y)
        p = @allocated callfill!(Y)
        @test p == 0
    end
end

# https://github.com/CliMA/ClimaCore.jl/issues/963
sc(::Type{FT}) where {FT} =
    Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
function allocs_test1!(Y)
    x = Y.x
    FT = Spaces.undertype(axes(x))
    I = sc(FT)
    x .= x .+ Ref(I)
    nothing
end
function allocs_test2!(Y)
    x = Y.x
    FT = Spaces.undertype(axes(x))
    IR = Ref(sc(FT))
    @. x += IR
    nothing
end
function allocs_test1_column!(Y)
    Fields.bycolumn(axes(Y.x)) do colidx
        x = Y.x
        FT = Spaces.undertype(axes(x))
        # I = sc(FT)
        I = Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
        x[colidx] .= x[colidx] .+ Ref(I)
    end
    nothing
end
function allocs_test2_column!(Y)
    Fields.bycolumn(axes(Y.x)) do colidx
        x = Y.x
        FT = Spaces.undertype(axes(x))
        IR = Ref(sc(FT))
        @. x[colidx] += IR
    end
    nothing
end

function allocs_test3!(Y)
    Fields.bycolumn(axes(Y.x)) do colidx
        allocs_test3_column!(Y.x[colidx])
    end
    nothing
end

function allocs_test3_column!(x)
    FT = Spaces.undertype(axes(x))
    IR = Ref(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
    @. x += IR
    I = Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
    x .+= Ref(I)
    nothing
end

@testset "Allocations StencilCoefs broadcasting" begin
    FT = Float64
    for space in all_spaces(FT)
        Y = FieldFromNamedTuple(space, (; x = sc(FT)))
        allocs_test1!(Y)
        p = @allocated allocs_test1!(Y)
        @test p == 0
        allocs_test2!(Y)
        p = @allocated allocs_test2!(Y)
        @test p == 0

        bycolumnable(space) || continue

        allocs_test1_column!(Y)
        p = @allocated allocs_test1_column!(Y)
        @test p == 0

        allocs_test2_column!(Y)
        p = @allocated allocs_test2_column!(Y)
        @test p == 0

        allocs_test3!(Y)
        p = @allocated allocs_test3!(Y)
        @test p == 0
    end
end
nothing

function allocs_test_Ref_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
    Fields.bycolumn(axes(S)) do colidx
        allocs_test_Ref_with_compose_column!(
            S[colidx],
            ∂ᶠ𝕄ₜ∂ᶜρ[colidx],
            ∂ᶜρₜ∂ᶠ𝕄[colidx],
        )
    end
    nothing
end

function allocs_test_Ref_with_compose_column!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
    compose = Operators.ComposeStencils()
    FT = Spaces.undertype(axes(S))
    IR = Ref(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
    @. S = compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) - IR
    nothing
end

@testset "Allocations StencilCoefs Ref with ComposeStencils broadcasting" begin
    FT = Float64
    for space in all_spaces(FT)
        space isa Spaces.CenterExtrudedFiniteDifferenceSpace || continue
        cspace = space
        fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
        bidiag_type = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
        ∂ᶠ𝕄ₜ∂ᶜρ = Fields.Field(bidiag_type, fspace)
        ∂ᶜρₜ∂ᶠ𝕄 = Fields.Field(bidiag_type, cspace)
        tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
        S = Fields.Field(tridiag_type, fspace)

        allocs_test_Ref_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
        p = @allocated allocs_test_Ref_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
        @test_broken p == 0
    end
end
nothing
