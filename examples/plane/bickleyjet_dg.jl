# Bickley jet (see `bickleyjet_cg.jl`), discretized with a discontinuous
# Galerkin spectral element method. Inter-element coupling is through a
# numerical flux rather than DSS, so this is where the flux choice matters:
# pass `central`, `rusanov` (default), or `roe` as the first argument, and
# optionally a boundary condition as the second.
#
# `central` adds no interface dissipation, so it does not survive the roll-up:
# the filaments the instability produces feed grid-scale energy that nothing
# removes, and the run goes to NaN. That is the standard reason a DG transport
# scheme needs an upwind-biased flux, and it is why `rusanov` is the default
# and why CI runs only `rusanov` and `roe`.
using ClimaComms
using LinearAlgebra

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

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const context = ClimaComms.SingletonCommsContext()

const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    g = 10,
)

numflux_name = get(ARGS, 1, "rusanov")
boundary_name = get(ARGS, 2, "")

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = boundary_name != "noslip",
        boundary_names = boundary_name != "noslip" ? nothing : (:south, :north),
    ),
)

n1, n2 = 16, 16
Nq = 4
Nqh = 7
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

Iquad = Quadratures.GLL{Nqh}()

function init_state(coord, p)
    x, y = coord.x, coord.y
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(y)^(-2)

    # Ψ′ = exp(-(y + p.l / 10)^2 / 2p.l^2) * cos(p.k * x) * cos(p.k * y)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)


    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * y)

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ * u)
end

function energy(state, p)
    ρ, ρu = state.ρ, state.ρu
    u = ρu / ρ
    return ρ * (u.u^2 + u.v^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end

# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    λ = sqrt(parameters⁻.g)

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5
    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)

    return map(flux(y⁻, parameters⁻), flux(y⁺, parameters⁺), Δf) do F⁻, F⁺, Δf
        ((F⁻ + F⁺) / 2)' * n + Δf
    end
end


numflux = if numflux_name == "central"
    Operators.CentralNumericalFlux(flux)
elseif numflux_name == "rusanov"
    Operators.RusanovNumericalFlux(flux, wavespeed)
elseif numflux_name == "roe"
    roeflux
else
    # Without this, an unrecognized name leaves `numflux === nothing` and the
    # run dies inside `add_numerical_flux_interior!` with a `MethodError`.
    error("Unknown numerical flux $(repr(numflux_name)): pass one of \
           \"central\", \"rusanov\" or \"roe\".")
end

function rhs!(dydt, y, (parameters, numflux), t)

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order space
    #  D = derivative operator
    #  H = suffix for higher-order space operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    wdiv = Operators.Divergence{Operators.WeakForm}()

    local_geometry_field = Fields.local_geometry_field(y)

    dydt .= wdiv.(flux.(y, Ref(parameters))) .* (.-(local_geometry_field.WJ))

    Operators.add_numerical_flux_interior!(numflux, dydt, y, parameters)
    Operators.add_numerical_flux_boundary!(
        dydt,
        y,
        parameters,
    ) do normal, (y⁻, parameters)
        y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu - dot(y⁻.ρu, normal) * normal, ρθ = y⁻.ρθ)
        numflux(normal, (y⁻, parameters), (y⁺, parameters))
    end

    # 6. Solve for final result. Both steps must land back in `dydt`:
    # `field_values` is a view, but `./` on it would build a new data layout and
    # leave `dydt` holding the unnormalized residual.
    dydt_data = Fields.field_values(dydt)
    dydt_data .= dydt_data ./ Spaces.local_geometry_data(space).WJ
    M = Quadratures.cutoff_filter_matrix(
        Float64,
        Spaces.quadrature_style(space),
        3,
    )
    Operators.tensor_product!(dydt_data, M)
    return dydt
end

dydt = Fields.Field(similar(Fields.field_values(y0)), space)
rhs!(dydt, y0, (parameters, numflux), 0.0);

# Solve the ODE operator
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs!),
    y0,
    (0.0, 200.0),
    (parameters, numflux),
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = 0.02,
    saveat = collect(0.0:1.0:200.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dg_$(numflux_name)"
if boundary_name != ""
    dir = "$(dir)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]

using Test
const is_periodic = boundary_name == ""

# On the periodic domain the numerical flux penalizes the jump across each
# interface and nothing else touches the energy, so it may only fall (measured
# drift over t = 200: -3.0e-4 for `rusanov`, -3.3e-4 for `roe` — larger than in
# the CG cases because the roll-up puts real structure on the interfaces for
# the penalty to act on). A wall does work on the domain, so that argument does
# not apply to `noslip`, which gains 4.2e-4 instead. Either way the drift must
# stay small.
if is_periodic
    @test Es[end] ≤ Es[1] * (1 + sqrt(eps()))
end
@test abs(Es[end] - Es[1]) / Es[1] < 1e-3

@testset "mass and tracer conservation" begin
    masses = [sum(y.ρ) for y in sol.u]
    tracers = [sum(y.ρθ) for y in sol.u]
    mass_drift = maximum(abs, masses .- masses[1]) / masses[1]
    tracer_drift = maximum(abs, tracers .- tracers[1])
    if is_periodic
        # A conservative numerical flux telescopes across the periodic domain,
        # so both hold to roundoff — identically for `rusanov` and `roe`, since
        # conservation is a property of the assembly and not of the flux
        # (measured: 1.4e-14 relative in ρ, 1e-13 absolute in ρθ, whose exact
        # integral is zero because θ = sin(k y) over a whole number of periods).
        @test mass_drift < 1e-12
        @test tracer_drift < 1e-10
    else
        # The wall reflects the normal momentum, which cancels the centered
        # part of the flux but not its dissipative part, so the wall leaks
        # (measured over t = 200: 3.5e-4 in mass, 0.054 in tracer mass). That
        # is a property of this boundary treatment, not of the interior
        # scheme — see the periodic branch above.
        @test mass_drift < 1e-3
        @test tracer_drift < 0.5
    end
end

@testset "jet roll-up and tracer overshoot" begin
    cross_jet_speed(y) =
        maximum(abs, Geometry.UVVector.(y.ρu ./ y.ρ).components.data.:2)
    # The shear layer is barotropically unstable, so the seeded perturbation
    # must roll the jet up here as it does in `bickleyjet_cg.jl` (measured
    # max|v| over the run: 0.050 at t = 0, peaking at 0.66 for `rusanov`, 0.71
    # for `roe` and 0.65 for `roe noslip`).
    speeds = cross_jet_speed.(sol.u)
    @test maximum(speeds) > 5 * speeds[1]
    # Rolling the jet up draws the tracer into filaments the mesh cannot
    # resolve, and there is no limiter, so θ overshoots its initial [-1, 1]
    # range (measured: ±3.7 for `rusanov`, ±2.6 for `roe`, ±2.0 with walls).
    θ_end = sol.u[end].ρθ ./ sol.u[end].ρ
    @test maximum(abs, θ_end) < 5
end
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
