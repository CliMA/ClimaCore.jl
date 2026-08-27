using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies
import ClimaCore.Geometry: ⊗

# End-to-end smoke test: the 2D shallow-water Bickley jet, run with continuous
# Galerkin (weak divergence completed by DSS) and with discontinuous Galerkin
# (weak divergence completed by a Rusanov interface flux), checking total mass
# conservation and numerical stability for Float32 and Float64.
#
# Both forms build their divergence from the same physical flux function
# `sw_flux` (components in the local orthonormal (U, V) basis, as in
# examples/bickleyjet/): the operators apply the metric transform to
# contravariant components internally, and the interface flux is evaluated
# against the physical unit normal, so the volume and surface terms are
# mutually consistent.

# Bickley jet: zonal jet c sech²(y) with a vortical perturbation derived from
# the streamfunction Ψ′ = exp(-(y + l/10)²/2l²) cos(kx) cos(ky), via
# (u₁′, u₂′) = (-∂Ψ′/∂y, ∂Ψ′/∂x).
struct BickleyInit{P}
    params::P
end
function (init::BickleyInit)(coord)
    x, y = coord.x, coord.y
    p = init.params
    ρ = p.ρ₀
    U₁ = p.c / Base.cosh(y)^2
    gaussian = exp(-(y + p.l / 10)^2 / (2 * p.l^2))
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)
    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    θ = sin(p.k * y)
    return (; ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

# Physical shallow-water fluxes (pressure p = g ρ²/2), in the local
# orthonormal basis. Used by the volume divergence of both forms and by the
# Rusanov interface flux of the DG form.
function sw_flux(state, p)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (;
        ρ = ρu,
        ρu = (ρu ⊗ u) + (p.g * ρ^2 / 2) * LinearAlgebra.I,
        ρθ = ρθ * u,
    )
end

# Upper bound on the normal signal speed |u ⋅ n| + √(g ρ).
sw_wavespeed(state, p) = sqrt(p.g * state.ρ) + norm(state.ρu / state.ρ)

function shallow_water_rhs_cg!(dydt, y, (space, params), t)
    wdiv = Operators.WeakDivergence()
    rparams = Ref(params)
    @. dydt = -wdiv(sw_flux(y, rparams))
    Spaces.weighted_dss!(dydt)
    return dydt
end

function shallow_water_rhs_dg!(dydt, y, (space, params, numflux), t)
    wdiv = Operators.WeakDivergence()
    lgeom = Fields.local_geometry_field(space)

    # Volume weak divergence, weighted by WJ so the interface flux
    # contributions can be accumulated directly.
    rparams = Ref(params)
    @. dydt = wdiv(sw_flux(y, rparams)) * (-lgeom.WJ)

    # Surface numerical flux across element boundaries
    Operators.add_numerical_flux_internal!(numflux, dydt, y, params)

    # Un-weight by dividing by the metric determinant WJ
    @. dydt = dydt / lgeom.WJ
    return dydt
end

@testset "Cross-Module Integration: Bickley Jet (CG & DG)" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        domain = Domains.RectangleDomain(
            Domains.IntervalDomain(
                Geometry.XPoint(-FT(2π)),
                Geometry.XPoint(FT(2π)),
                periodic = true,
            ),
            Domains.IntervalDomain(
                Geometry.YPoint(-FT(2π)),
                Geometry.YPoint(FT(2π)),
                periodic = true,
            ),
        )
        mesh = Meshes.RectilinearMesh(domain, 8, 8)
        grid_topology = Topologies.Topology2D(context, mesh)
        quad = Quadratures.GLL{4}()
        space = Spaces.SpectralElementSpace2D(grid_topology, quad)
        coords = Fields.coordinate_field(space)

        params = (
            ϵ = FT(0.1),
            l = FT(0.5),
            k = FT(0.5),
            ρ₀ = FT(1.0),
            c = FT(2.0),
            g = FT(10.0),
        )

        # Initial conditions: Bickley jet profile
        init_fn = BickleyInit(params)
        y0 = init_fn.(coords)
        mass0 = sum(y0.ρ)

        dt = FT(0.005)
        nsteps = 10

        # Two-stage SSP RK2 (Heun); whole-field broadcasts step the
        # NamedTuple-valued state (ρ, ρu, ρθ) componentwise.
        function rk_step!(step_rhs!, y, dydt, y_stage, ctx, dt)
            # Stage 1
            step_rhs!(dydt, y, ctx, FT(0))
            @. y_stage = y + dt * dydt
            # Stage 2
            step_rhs!(dydt, y_stage, ctx, dt)
            @. y = FT(0.5) * y + FT(0.5) * (y_stage + dt * dydt)
        end

        @testset "Continuous Galerkin (CG) Integration [$FT]" begin
            y = copy(y0)
            dydt = similar(y)
            y_stage = similar(y)

            # Warmup step
            shallow_water_rhs_cg!(dydt, y, (space, params), FT(0))

            # Run SSPRK steps
            for step in 1:nsteps
                rk_step!(shallow_water_rhs_cg!, y, dydt, y_stage, (space, params), dt)
            end

            # Verify mass conservation
            mass_final = sum(y.ρ)
            tol = FT == Float32 ? 1e-4 : 1e-10
            @test isapprox(mass_final, mass0, rtol = tol)
        end

        @testset "Discontinuous Galerkin (DG) Integration [$FT]" begin
            y = copy(y0)
            dydt = similar(y)
            y_stage = similar(y)
            numflux = Operators.RusanovNumericalFlux(sw_flux, sw_wavespeed)

            # Warmup step
            shallow_water_rhs_dg!(dydt, y, (space, params, numflux), FT(0))

            # Run SSPRK steps
            for step in 1:nsteps
                rk_step!(
                    shallow_water_rhs_dg!,
                    y,
                    dydt,
                    y_stage,
                    (space, params, numflux),
                    dt,
                )
            end

            # Verify mass conservation
            mass_final = sum(y.ρ)
            tol = FT == Float32 ? 1e-4 : 1e-10
            @test isapprox(mass_final, mass0, rtol = tol)
        end
    end
end
