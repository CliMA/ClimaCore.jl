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
# One tendency function serves both discretizations: the CG↔DG switch is the
# space (`discretization = Spaces.DG()` at grid construction) plus the
# `Operators.tendency_completion` object built from it, which applies DSS on
# the CG space and the interface numerical flux on the DG space.
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

# The element-local weak divergence, completed across element interfaces by
# the completion object (DSS on CG, Rusanov interface flux on DG).
function shallow_water_rhs!(dydt, y, (params, completion), t)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    rparams = Ref(params)
    @. dydt = -wdiv(sw_flux(y, rparams))
    Operators.complete_tendency!(completion, dydt, y, params)
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

        params = (
            ϵ = FT(0.1),
            l = FT(0.5),
            k = FT(0.5),
            ρ₀ = FT(1.0),
            c = FT(2.0),
            g = FT(10.0),
        )

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

        # The discretization is chosen here and nowhere else: the space's
        # discretization makes `tendency_completion` return the DSS
        # completion (CG) or the numerical-flux completion (DG); the tendency
        # function and the model state are shared.
        numflux = Operators.RusanovNumericalFlux(sw_flux, sw_wavespeed)
        for (name, discretization) in (
            ("Continuous Galerkin (CG)", Spaces.CG()),
            ("Discontinuous Galerkin (DG)", Spaces.DG()),
        )
            @testset "$name Integration [$FT]" begin
                space = Spaces.SpectralElementSpace2D(
                    grid_topology,
                    quad;
                    discretization,
                )
                @test Spaces.discretization(space) === discretization
                @test Spaces.is_continuous(space) ==
                      (discretization isa Spaces.CG)
                coords = Fields.coordinate_field(space)
                y0 = BickleyInit(params).(coords)
                mass0 = sum(y0.ρ)

                y = copy(y0)
                dydt = similar(y)
                y_stage = similar(y)
                completion = Operators.tendency_completion(dydt; numflux)

                # Compiles the tendency before the stepping loop; `dydt` is
                # overwritten by stage 1 of the first `rk_step!`, so this call
                # does not affect the trajectory.
                shallow_water_rhs!(dydt, y, (params, completion), FT(0))

                for step in 1:nsteps
                    rk_step!(
                        shallow_water_rhs!,
                        y,
                        dydt,
                        y_stage,
                        (params, completion),
                        dt,
                    )
                end

                mass_final = sum(y.ρ)
                tol = FT == Float32 ? 1e-4 : 1e-10
                @test isapprox(mass_final, mass0, rtol = tol)
            end
        end
    end
end
