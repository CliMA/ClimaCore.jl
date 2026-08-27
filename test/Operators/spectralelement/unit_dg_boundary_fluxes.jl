using Test
using LinearAlgebra
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# `add_numerical_flux_boundary!`, the boundary-face counterpart of
# `add_numerical_flux_internal!`, on a channel domain (periodic in x, walls at
# y = ±Ly/2). The interior-face path is covered elsewhere, on boundary-free
# domains:
#
#   1. A zero boundary flux leaves the residual untouched.
#   2. A constant boundary flux c integrates to −c·(total boundary length):
#      the residual update is −sWJ·f per boundary node, and the GLL surface
#      weights sum to the boundary measure.
#   3. The residual is nonzero only at nodes on the domain boundary.
#   4. Boundary normals are outward: a flux f = ĵ·n̂ is −1 on the south wall
#      and +1 on the north wall, so the two walls contribute ±Lx.

function channel_space(::Type{FT}; Lx = FT(2π), Ly = FT(2), nelem = 4, Nq = 4) where {FT}
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(zero(Lx)) .. Geometry.XPoint{FT}(Lx),
        Geometry.YPoint{FT}(-Ly / 2) .. Geometry.YPoint{FT}(Ly / 2);
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discontinuous = true,
    )
end

@testset "DG boundary numerical fluxes" begin
    TU.@test_precisions FT begin
        Lx = FT(2π)
        Ly = FT(2)
        space = channel_space(FT; Lx, Ly)
        @test !Spaces.is_continuous(space)
        coords = Fields.coordinate_field(space)
        q = ones(space)

        @testset "Zero boundary flux is a no-op [$FT]" begin
            r = zeros(space)
            zero_flux(normal, (q⁻,)) = zero(q⁻)
            Operators.add_numerical_flux_boundary!(zero_flux, r, q)
            @test maximum(abs, parent(r)) == 0
        end

        @testset "Constant flux integrates to -c * boundary length [$FT]" begin
            c = FT(3)
            r = zeros(space)
            const_flux(normal, (q⁻,)) = c
            Operators.add_numerical_flux_boundary!(const_flux, r, q)
            # Boundary = south wall + north wall (x is periodic): length 2*Lx
            @test sum(parent(r)) ≈ -c * 2 * Lx rtol = sqrt(eps(FT))
        end

        @testset "Only boundary nodes are touched [$FT]" begin
            r = zeros(space)
            const_flux(normal, (q⁻,)) = one(q⁻)
            Operators.add_numerical_flux_boundary!(const_flux, r, q)
            y = Array(parent(Fields.coordinate_field(space).y))
            interior = @. abs(abs(y) - Ly / 2) > sqrt(eps(FT))
            @test all(iszero, Array(parent(r))[interior])
            @test any(!iszero, Array(parent(r))[.!interior])
        end

        @testset "Boundary normals point outward [$FT]" begin
            r = zeros(space)
            ĵ = Geometry.UVVector(FT(0), FT(1))
            normal_flux(normal, (q⁻,)) = ĵ' * normal
            Operators.add_numerical_flux_boundary!(normal_flux, r, q)
            y = Array(parent(Fields.coordinate_field(space).y))
            south = @. abs(y + Ly / 2) <= sqrt(eps(FT))
            north = @. abs(y - Ly / 2) <= sqrt(eps(FT))
            # residual = -sWJ * f: south (n̂ = -ĵ, f = -1) gains +sWJ,
            # north (n̂ = +ĵ, f = +1) gains -sWJ; each wall has length Lx.
            @test sum(Array(parent(r))[south]) ≈ Lx rtol = sqrt(eps(FT))
            @test sum(Array(parent(r))[north]) ≈ -Lx rtol = sqrt(eps(FT))
        end
    end
end
