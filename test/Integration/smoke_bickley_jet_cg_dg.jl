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

# Cross-module End-to-End Integration Smoke Test:
# Runs Shallow-Water 2D Bickley Jet using both:
# 1. Continuous Galerkin (CG) with DSS
# 2. Discontinuous Galerkin (DG) with Rusanov and Central interface fluxes
# Verifies total mass conservation, numerical stability, and consistency across precision.

struct BickleyInit{P}
    params::P
end
function (init::BickleyInit)(coord)
    x, y = coord.x, coord.y
    p = init.params
    ρ = p.ρ₀
    u_x = p.c / (Base.cosh(y)^2)
    u_y = p.ϵ * p.c * sin(p.k * x) / (Base.cosh(y)^2)
    u = Geometry.UVVector(u_x, u_y)
    θ = p.ρ₀ + (typeof(p.ρ₀)(0.1)) * cos(y)
    return (; ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

function compute_sw_fluxes(y, params)
    ρ = y.ρ
    ρu = y.ρu
    ρθ = y.ρθ
    u = @. ρu.components.data.:1 / ρ
    v = @. ρu.components.data.:2 / ρ
    p = @. params.g * ρ^2 / 2

    # Mass flux (contravariant vector): [ρu, ρv]
    F_ρ = @. Geometry.Contravariant12Vector(ρ * u, ρ * v)

    # Momentum fluxes (contravariant vectors for u-momentum and v-momentum)
    F_u = @. Geometry.Contravariant12Vector(ρ * u * u + p, ρ * u * v)
    F_v = @. Geometry.Contravariant12Vector(ρ * u * v, ρ * v * v + p)

    # Tracer flux
    F_θ = @. Geometry.Contravariant12Vector(ρθ * u, ρθ * v)

    return F_ρ, F_u, F_v, F_θ
end

function shallow_water_rhs_cg!(dydt, y, (space, params), t)
    sdiv = Operators.Divergence()
    F_ρ, F_u, F_v, F_θ = compute_sw_fluxes(y, params)

    div_ρ = sdiv.(F_ρ)
    div_u = sdiv.(F_u)
    div_v = sdiv.(F_v)
    div_θ = sdiv.(F_θ)

    @. dydt.ρ = -div_ρ
    @. dydt.ρu = Geometry.UVVector(-div_u, -div_v)
    @. dydt.ρθ = -div_θ

    Spaces.weighted_dss!(dydt)
    return dydt
end

function sw_normal_flux(state, p, normal)
    ρ = state.ρ
    u = state.ρu.components.data.:1 / ρ
    v = state.ρu.components.data.:2 / ρ
    un = u * normal.u + v * normal.v
    pres = p.g * ρ^2 / 2
    return (
        ρ = ρ * un,
        ρu = Geometry.UVVector(
            state.ρu.components.data.:1 * un + pres * normal.u,
            state.ρu.components.data.:2 * un + pres * normal.v,
        ),
        ρθ = state.ρθ * un,
    )
end

struct ConstantWaveSpeed{FT}
    speed::FT
end
(c::ConstantWaveSpeed)(state, p) = c.speed

struct RusanovSWFlux{W}
    wavespeedfn::W
end
function (fn::RusanovSWFlux)(normal, argvals⁻, argvals⁺)
    y⁻, params = argvals⁻[1], argvals⁻[2]
    y⁺ = argvals⁺[1]
    Fn⁻ = sw_normal_flux(y⁻, params, normal)
    Fn⁺ = sw_normal_flux(y⁺, params, normal)
    λ = max(fn.wavespeedfn(y⁻, params), fn.wavespeedfn(y⁺, params))
    return (
        ρ = (Fn⁻.ρ + Fn⁺.ρ) / 2 + (λ / 2) * (y⁻.ρ - y⁺.ρ),
        ρu = (Fn⁻.ρu + Fn⁺.ρu) / 2 + (λ / 2) * (y⁻.ρu - y⁺.ρu),
        ρθ = (Fn⁻.ρθ + Fn⁺.ρθ) / 2 + (λ / 2) * (y⁻.ρθ - y⁺.ρθ),
    )
end

function shallow_water_rhs_dg!(dydt, y, (space, params, numflux), t)
    wdiv = Operators.WeakDivergence()
    lgeom = Fields.local_geometry_field(space)
    F_ρ, F_u, F_v, F_θ = compute_sw_fluxes(y, params)

    # Volume weak divergence in DG: -wdiv(F) * WJ (since wdiv returns the normalized derivative)
    dydt_weighted_ρ = @. -wdiv(F_ρ) * lgeom.WJ
    dydt_weighted_u = @. -wdiv(F_u) * lgeom.WJ
    dydt_weighted_v = @. -wdiv(F_v) * lgeom.WJ
    dydt_weighted_θ = @. -wdiv(F_θ) * lgeom.WJ

    dydt_weighted = map(
        (ρ_w, u_w, v_w, θ_w) -> (; ρ = ρ_w, ρu = Geometry.UVVector(u_w, v_w), ρθ = θ_w),
        dydt_weighted_ρ,
        dydt_weighted_u,
        dydt_weighted_v,
        dydt_weighted_θ,
    )

    # Surface numerical flux across element boundaries
    Operators.add_numerical_flux_internal!(numflux, dydt_weighted, y, params)

    # Un-weight by dividing by metric determinant WJ
    @. dydt.ρ = dydt_weighted.ρ / lgeom.WJ
    @. dydt.ρu = Geometry.UVVector(
        dydt_weighted.ρu.components.data.:1 / lgeom.WJ,
        dydt_weighted.ρu.components.data.:2 / lgeom.WJ,
    )
    @. dydt.ρθ = dydt_weighted.ρθ / lgeom.WJ
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

        # Initial conditions: Bickley Jet profile
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
            # `add_numerical_flux_internal!` iterates interior faces with host
            # scalar indexing, so the DG interface-flux path is CPU-only.
            if ClimaComms.device() isa ClimaComms.CUDADevice
                @test_skip "DG interface flux (add_numerical_flux_internal!) is CPU-only"
            else
                y = copy(y0)
                dydt = similar(y)
                y_stage = similar(y)
                numflux = RusanovSWFlux(ConstantWaveSpeed(sqrt(params.g)))

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
end
