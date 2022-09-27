# These tests require running with `--check-bounds=[auto|no]`
using Test
using StaticArrays, IntervalSets
import ClimaCore
import ClimaCore.Utilities: PlusHalf, half
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, slab, Domains, Topologies, Meshes, Operators, Spaces, Geometry

using FastBroadcast
using LinearAlgebra: norm
using Statistics: mean
using ForwardDiff

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

# https://github.com/CliMA/ClimaCore.jl/issues/946
@testset "Allocations with broadcasting Scalars" begin
    FT = Float64
    function foo!(Yx::Fields.Field)
        Yx .= (1,) .+ Yx
        return nothing
    end
    function foocolumn!(Yx::Fields.Field)
        Fields.bycolumn(axes(Yx)) do colidx
            Yx[colidx] .= (1,) .+ Yx[colidx]
            nothing
        end
        return nothing
    end
    for space in TU.all_spaces(FT)
        TU.bycolumnable(space) || continue
        Y = fill((; x = FT(2)), space)

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
@testset "Allocations with getproperty on Fields" begin
    FT = Float64
    function allocs_test!(Y)
        x = Y.x
        fill!(x, 2.0)
        nothing
    end
    function callfill!(Y)
        fill!(Y, (; x = 2.0))
        nothing
    end
    for space in TU.all_spaces(FT)
        Y = fill((; x = FT(2)), space)
        allocs_test!(Y)
        p = @allocated allocs_test!(Y)
        @test p == 0

        callfill!(Y)
        p = @allocated callfill!(Y)
        @test p == 0
    end
end

function fast_broadcast_single_field!(Y1, dt, Y2)
    x = Y1.x
    @.. x += 2.0
    nothing
end
# Removing dt from argument fixes allocations
function fast_broadcast_copyto!(Y1, dt, Y2)
    @.. Y1 = Y2
    nothing
end
# https://github.com/CliMA/ClimaCore.jl/issues/1356
@testset "Allocations in @.. broadcasting" begin
    FT = Float32
    for space in TU.all_spaces(FT)
        Y1 = fill((; x = FT(2.0), y = FT(2.0), z = FT(2.0)), space)
        Y2 = fill((; x = FT(2.0), y = FT(2.0), z = FT(2.0)), space)
        Y3 = fill((; x = FT(2.0), y = FT(2.0), z = FT(2.0)), space)
        dt = FT(2.0)
        fast_broadcast_single_field!(Y1, dt, Y2)
        p = @allocated fast_broadcast_single_field!(Y1, dt, Y2)
        @test p == 0
        fast_broadcast_copyto!(Y1, dt, Y2)
        p = @allocated fast_broadcast_copyto!(Y1, dt, Y2)
        @test_broken p == 0
    end
end

# https://github.com/CliMA/ClimaCore.jl/issues/963
sc(::Type{FT}) where {FT} =
    Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
function allocs_test1!(Y)
    x = Y.x
    FT = Spaces.undertype(axes(x))
    I = sc(FT)
    x .= x .+ (I,)
    nothing
end
function allocs_test2!(Y)
    x = Y.x
    FT = Spaces.undertype(axes(x))
    IR = (sc(FT),)
    @. x += IR
    nothing
end
function allocs_test1_column!(Y)
    Fields.bycolumn(axes(Y.x)) do colidx
        x = Y.x
        FT = Spaces.undertype(axes(x))
        # I = sc(FT)
        I = Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
        x[colidx] .= x[colidx] .+ (I,)
    end
    nothing
end
function allocs_test2_column!(Y)
    Fields.bycolumn(axes(Y.x)) do colidx
        x = Y.x
        FT = Spaces.undertype(axes(x))
        IR = (sc(FT),)
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
    IR = (Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))),)
    @. x += IR
    I = Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT)))
    x .+= (I,)
    nothing
end

@testset "Allocations StencilCoefs broadcasting" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        Y = fill((; x = sc(FT)), space)
        allocs_test1!(Y)
        p = @allocated allocs_test1!(Y)
        @test p == 0
        allocs_test2!(Y)
        p = @allocated allocs_test2!(Y)
        @test p == 0

        TU.bycolumnable(space) || continue

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

function allocs_test_scalar_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
    Fields.bycolumn(axes(S)) do colidx
        allocs_test_scalar_with_compose_column!(
            S[colidx],
            ∂ᶠ𝕄ₜ∂ᶜρ[colidx],
            ∂ᶜρₜ∂ᶠ𝕄[colidx],
        )
    end
    nothing
end

function allocs_test_scalar_with_compose_column!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
    compose = Operators.ComposeStencils()
    FT = Spaces.undertype(axes(S))
    IR = (Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))),)
    @. S = compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) - IR
    nothing
end

@testset "Allocations StencilCoefs scalar with ComposeStencils broadcasting" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        space isa Spaces.CenterExtrudedFiniteDifferenceSpace || continue
        cspace = space
        fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
        bidiag_type = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
        ∂ᶠ𝕄ₜ∂ᶜρ = Fields.Field(bidiag_type, fspace)
        ∂ᶜρₜ∂ᶠ𝕄 = Fields.Field(bidiag_type, cspace)
        tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
        S = Fields.Field(tridiag_type, fspace)

        allocs_test_scalar_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
        p = @allocated allocs_test_scalar_with_compose!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
        @test p == 0

        allocs_test_scalar_with_compose_column!(S, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄)
        p = @allocated allocs_test_scalar_with_compose_column!(
            S,
            ∂ᶠ𝕄ₜ∂ᶜρ,
            ∂ᶜρₜ∂ᶠ𝕄,
        )
        @test p == 0
    end
end

function call_zero_eltype!(Y)
    Y .= zero(eltype(Y))
    nothing
end
# https://github.com/CliMA/ClimaCore.jl/issues/983
@testset "Allocations with fill! and zero eltype broadcasting on FieldVectors" begin
    FT = Float64
    for space in TU.all_spaces(FT)
        Y = Fields.FieldVector(;
            c = fill((; x = FT(0)), space),
            f = fill((; x = FT(0)), space),
        )

        Y .= 0 # compile first
        p = @allocated begin
            Y .= 0
            nothing
        end
        @test p == 0

        call_zero_eltype!(Y) # compile first
        p = @allocated call_zero_eltype!(Y)
        @test p == 0

        fill!(Y, zero(eltype(Y))) # compile first
        p = @allocated begin
            fill!(Y, zero(eltype(Y)))
            nothing
        end
        @test p == 0
    end
end

# https://github.com/CliMA/ClimaCore.jl/issues/1062
@testset "Allocations with copyto! on FieldVectors" begin
    function toy_sphere(::Type{FT}) where {FT}
        helem = npoly = 2
        hdomain = Domains.SphereDomain(FT(1e7))
        hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
        htopology = Topologies.Topology2D(hmesh)
        quad = Spaces.Quadratures.GLL{npoly + 1}()
        hspace = Spaces.SpectralElementSpace2D(htopology, quad)
        vdomain = Domains.IntervalDomain(
            Geometry.ZPoint{FT}(zero(FT)),
            Geometry.ZPoint{FT}(FT(1e4));
            boundary_tags = (:bottom, :top),
        )
        vmesh = Meshes.IntervalMesh(vdomain, nelems = 4)
        vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)
        center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
        return (center_space, face_space)
    end
    function field_vec(center_space, face_space)
        Y = Fields.FieldVector(
            c = map(Fields.coordinate_field(center_space)) do coord
                FT = Spaces.undertype(center_space)
                (; ρ = FT(0), uₕ = Geometry.Covariant12Vector(FT(0), FT(0)))
            end,
            f = map(Fields.coordinate_field(face_space)) do coord
                FT = Spaces.undertype(face_space)
                (; w = Geometry.Covariant3Vector(FT(0)))
            end,
        )
        return Y
    end
    get_n(::Val{n}) where {n} = n
    function foo!(obj)
        @inbounds for i in 1:get_n(obj.N)
            @. obj.U[i] = obj.u
        end
        return nothing
    end
    u = field_vec(toy_sphere(Float64)...)
    n = 4
    U = map(i -> similar(u), collect(1:n))
    obj = (; u, N = Val(n), U)
    foo!(obj) # compile first

    palloc = @allocated foo!(obj)
    @test palloc == 0
end
nothing
