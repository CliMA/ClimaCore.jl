using Test
using LinearAlgebra
using Random
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

# Structural properties the flux-differencing DG method (issue #2561) relies on
# for stability without artificial viscosity:
#
#   1. Summation by parts, for a *non-constant* flux.
#   2. Sign-definite interface dissipation: the jump penalty satisfies
#      ∑ q·r ≤ 0, strictly across a discontinuity and exactly zero for
#      a continuous field.
#   3. Global conservation: the single-valued interface flux is added
#      antisymmetrically, so the node sum of the residual is structurally zero.

include("utils_dg.jl")  # dg_sphere_space, dg_central_flux, dg_jump_penalty

function dg_plane_space(::Type{FT}; L = FT(2π), xelem = 4, yelem = 4, Nq = 4) where {FT}
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-L / 2) .. Geometry.XPoint{FT}(L / 2),
        Geometry.YPoint{FT}(-L / 2) .. Geometry.YPoint{FT}(L / 2);
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.RectilinearMesh(domain, xelem, yelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(
        topology,
        Quadratures.GLL{Nq}();
        discretization = Spaces.DG(),
    )
end

# A smooth, single-valued scalar and velocity for a given space. The `+ 2`
# keeps the scalar positive; the element type follows from the coordinates, so
# the state carries the space's precision.
function smooth_state(space)
    coords = Fields.coordinate_field(space)
    if :long in propertynames(coords) # sphere
        q = @. sind(coords.long) * cosd(coords.lat)^2 + 2
        uv = @. Geometry.UVVector(
            cosd(coords.long),
            -sind(coords.long) * sind(coords.lat),
        )
    else # plane
        q = @. sin(coords.x) * cos(coords.y) + 2
        uv = @. Geometry.UVVector(cos(coords.x), sin(coords.y))
    end
    return (q, uv)
end

@testset "DG stability properties" begin
    TU.@test_precisions FT begin
        for (name, space) in
            (("sphere", dg_sphere_space(FT)), ("plane", dg_plane_space(FT)))
            @test !Spaces.is_continuous(space)
            lgeom = Fields.local_geometry_field(space)
            q, uv = smooth_state(space)

            @testset "SBP identity, non-constant flux [$name, $FT]" begin
                # Divergence{WeakForm}(F)*(-WJ) + central surface flux ==
                # -Divergence(F), node-wise, for the *non-constant* flux
                # F = q*u. This is the discrete summation-by-parts identity
                # relating the weak and strong divergence operators.
                hwdiv = Operators.Divergence{Operators.WeakForm}()
                hdiv = Operators.Divergence()
                F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), q .* uv)
                y = map((qi, uvi) -> (; q = qi, uv = uvi), q, uv)

                dy_mw = @. hwdiv(F) * (-(lgeom.WJ))
                Operators.add_numerical_flux_interior!(dg_central_flux, dy_mw, y)
                dy = @. dy_mw / lgeom.WJ

                dy_strong = @. -hdiv(F)
                scale = maximum(abs, parent(dy_strong))
                tol = FT == Float32 ? 1e-3 : 1e-9
                @test maximum(abs, parent(dy) .- parent(dy_strong)) < tol * scale
            end

            @testset "Interface dissipation is sign-definite [$name, $FT]" begin
                # Energy rate ∑ q·r from the jump penalty must be ≤ 0, exactly
                # zero for a continuous (single-valued) field and strictly
                # negative across a discontinuity.
                rc = similar(q)
                fill!(parent(rc), 0)
                Operators.add_numerical_flux_interior!(dg_jump_penalty, rc, q)
                energy_continuous = sum(parent(q) .* parent(rc))
                q_scale = sum(abs2, parent(q))
                tol = FT == Float32 ? 1e-4 : 1e-10
                @test abs(energy_continuous) < tol * q_scale

                qd = copy(q)
                Random.seed!(1234)
                qd_cpu = Array(parent(qd))
                qd_cpu .+= FT(0.1) .* (rand(FT, size(qd_cpu)) .- FT(0.5))
                copyto!(parent(qd), qd_cpu)
                rd = similar(qd)
                fill!(parent(rd), 0)
                Operators.add_numerical_flux_interior!(dg_jump_penalty, rd, qd)
                energy_discontinuous = sum(parent(qd) .* parent(rd))
                @test energy_discontinuous < 0
            end

            @testset "Global conservation of the single-valued flux [$name, $FT]" begin
                # The interior flux is added antisymmetrically (−sWJ·f to the
                # minus node, +sWJ·f to the plus node), so the node sum of the
                # residual is structurally zero for any single-valued flux,
                # even across a discontinuity.
                qd = copy(q)
                Random.seed!(2024)
                qd_cpu = Array(parent(qd))
                qd_cpu .+= FT(0.1) .* (rand(FT, size(qd_cpu)) .- FT(0.5))
                copyto!(parent(qd), qd_cpu)
                uvd = copy(uv)
                y = map((qi, uvi) -> (; q = qi, uv = uvi), qd, uvd)
                r = similar(qd)
                fill!(parent(r), 0)
                Operators.add_numerical_flux_interior!(dg_central_flux, r, y)
                scale = sum(abs, parent(r))
                tol = FT == Float32 ? 1e-5 : 1e-11
                @test abs(sum(parent(r))) < tol * scale
            end
        end
    end
end
