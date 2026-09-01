# Bickley jet: a barotropically unstable shear layer in the shallow-water
# equations (flux form, prognostic `(ρ, ρu, ρθ)`) on a plane. A small
# perturbation seeds the instability, which rolls the jet up into vortices. The
# run tracks total energy, which the discretization should nearly conserve.
#
# This example shows how a single driver can be defined for both CG + DG 
# discretizations. The prognostic equations, the physical flux, the initial 
# condition and the diagnostics are shared; the only
# structural difference is the space's `discretization`, which selects how the
# element-local weak-form tendency is completed across element interfaces:
#
#   * `Grids.CG()` — `Spaces.weighted_dss!`, projecting onto the continuous space;
#   * `Grids.DG()` — the mass-weighted interface numerical flux (and, on a
#     bounded domain, the one-sided boundary flux).
#
# `Operators.tendency_completion` builds that completion object once from the
# space and `Operators.complete_tendency!` applies it, so `rhs!` below is
# written once and is discretization-agnostic.
#
# Usage:
#
#     julia --project=.buildkite examples/plane/bickleyjet.jl [cg|dg] [numflux] [boundary] [hyperdiffusion]
#
# `numflux` is `central`, `rusanov` (default) or `roe`, and is used only by the
# DG completion — CG passes it and ignores it. 
# `boundary` is empty (doubly periodic, the default) or `noslip`, which
# closes the y-direction with walls; walls are DG-only here, since CG imposes
# boundary conditions through the operators rather than through a flux.
# `hyperdiffusion` is empty (off, the default) or `hyperdiffusion`, which adds
# a ∇⁴ closure on the velocity and the tracer. Like the rest of the tendency it
# is written once for both discretizations: the Laplacian atoms carry the DG
# face terms themselves, and the `Spaces.weighted_dss!` between the two passes
# — which makes the intermediates continuous on CG — is a no-op on DG.
#
# `central` adds no interface dissipation, so it does not survive the roll-up:
# the filaments the instability produces feed grid-scale energy that nothing
# removes, and the run goes to NaN. That is the standard reason a DG transport
# scheme needs an upwind-biased flux, and it is why `rusanov` is the default and
# why CI runs only `rusanov` and `roe`.
#
# See `bickleyjet_cg_invariant_hypervisc.jl` for the vector-invariant form with
# hyperviscosity. This example uses potential temperature `ρθ` instead of the
# total energy `ρe` formulation enforced in the hybrid models, as `ρθ` is
# analytically conserved here and simplifies the setup.
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

discretization_name = get(ARGS, 1, "cg")
numflux_name = get(ARGS, 2, "rusanov")
boundary_name = get(ARGS, 3, "")
hyperdiffusion_name = get(ARGS, 4, "")

discretization = if discretization_name == "cg"
    Spaces.CG()
elseif discretization_name == "dg"
    Spaces.DG()
else
    error("Unknown discretization $(repr(discretization_name)): pass \"cg\" or \
           \"dg\".")
end

# Everything below the discretization is configuration, not structure. The two
# discretizations are run with the settings each was tuned with, so the drift
# numbers quoted in the checks at the bottom stay comparable to the ones the
# separate `bickleyjet_cg.jl` / `bickleyjet_dg.jl` drivers reported:
#
#   * `Nqh` — over-integration quadrature (0 disables it). The CG form
#     de-aliases the quadratic flux this way; the DG form relies on the
#     interface dissipation and the cutoff filter instead.
#   * `filter_order` — modal cutoff filter applied to the tendency (0 disables
#     it).
#   * `tspan` — the CG run stops just after the roll-up, the DG run continues
#     into the vortex-merger phase.
config = if discretization isa Spaces.CG
    (; Nqh = 7, filter_order = 0, tspan = 80.0, rollup_factor = 10)
else
    (; Nqh = 0, filter_order = 3, tspan = 200.0, rollup_factor = 5)
end

const use_hyperdiffusion = if hyperdiffusion_name in ("", "hyperdiffusion")
    hyperdiffusion_name == "hyperdiffusion"
else
    error("Unknown hyperdiffusion option $(repr(hyperdiffusion_name)): pass \
           \"hyperdiffusion\" or nothing at all.")
end

const is_periodic = boundary_name == ""
if !is_periodic && discretization isa Spaces.CG
    error("boundary $(repr(boundary_name)) is DG-only in this example: the DG \
           completion imposes it as a one-sided numerical flux, while CG would \
           impose it through the operators.")
end

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = is_periodic,
        boundary_names = is_periodic ? nothing : (:south, :north),
    ),
)

n1, n2 = 16, 16
Nq = 4
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad; discretization)

# Over-integration space, on the same discretization as the model space.
Ispace =
    iszero(config.Nqh) ? nothing :
    Spaces.SpectralElementSpace2D(
        grid_topology,
        Quadratures.GLL{config.Nqh}();
        discretization,
    )

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

# Physical shallow-water flux (pressure p = g ρ²/2), in the local orthonormal
# (U, V) basis: the operators apply the metric transform to contravariant
# components internally and the interface flux is evaluated against the
# physical unit normal, so the volume and surface terms are mutually
# consistent. This one function feeds the weak volume divergence of both
# discretizations and the DG interface flux.
function flux(state, p)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * LinearAlgebra.I), ρθ = ρθ * u)
end

function energy(state, p)
    ρ, ρu = state.ρ, state.ρu
    u = ρu / ρ
    return ρ * (u.u^2 + u.v^2) / 2 + p.g * ρ^2 / 2
end

total_energy(y, p) = sum(state -> energy(state, p), y)

# numerical fluxes
wavespeed(y, p) = sqrt(p.g)

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
    # run dies inside `tendency_completion` (DG) with a less specific error.
    error("Unknown numerical flux $(repr(numflux_name)): pass one of \
           \"central\", \"rusanov\" or \"roe\".")
end

# Zero-normal-momentum wall: the ghost state cancels the normal momentum, so
# the centered part of the interface flux carries no mass or tracer through the
# wall while the dissipative part still acts. (`Operators.ReflectingWallBC`
# reflects it instead, `ρu − 2(ρu⋅n̂)n̂`, which is the mirror-image closure; this
# example keeps the weaker one it was tuned with.) `nothing` on a periodic
# domain, where there are no boundary faces to close.
boundary_numflux =
    is_periodic ? nothing :
    (normal, (y⁻, p)) -> begin
        y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu - dot(y⁻.ρu, normal) * normal, ρθ = y⁻.ρθ)
        numflux(normal, (y⁻, p), (y⁺, p))
    end

dydt = similar(y0)

# ∇⁴ hyperdiffusion, when active: high-order numerical diffusion 
# applied to the velocity and to the tracer. `κ₄ ~ c h³` is the 
# hyperdiffusion coefficient.
hyperdiffusion = if !use_hyperdiffusion
    nothing
else
    c = sqrt(parameters.g * parameters.ρ₀)
    κ₄ = 0.0015 * c * Spaces.node_horizontal_length_scale(space)^3
    # The DG Laplacian's interface penalty, τ ~ (2Nq − 1)²/h, dominates its
    # spectrum, so its ∇⁴ is ~(2Nq − 1)⁴ stiffer than the CG one at the same
    # resolution (measured spectral radius here: 1.3e8 against 5.8e4, a factor
    # of 2160 against the 2401 that scaling predicts). Dividing κ₄ by it keeps
    # the explicit time-step limit the same as CG's while still damping the
    # grid-scale jumps hard — those are exactly the modes the penalty makes
    # stiff, so they keep a damping rate of order 1/time.
    if discretization isa Spaces.DG
        κ₄ /= (2 * Nq - 1)^4
    end
    χu = similar(y0.ρu, Geometry.Covariant12Vector{Float64})
    χθ = similar(y0.ρθ)
    (;
        κ₄,
        u = similar(χu),
        χu,
        χθ,
        ∇⁴u = similar(χu),
        ∇⁴θ = similar(χθ),
        buffer_χu = Spaces.create_dss_buffer(χu),
        buffer_χθ = Spaces.create_dss_buffer(χθ),
    )
end

# Adds `-κ₄ ∇⁴` on the velocity and the tracer to the element-local tendency,
# before it is completed: on CG the completion's DSS then covers this term too,
# and on DG the atoms have already added their own face terms.
function hyperdiffusion_tendency!(dydt, y, hyperdiffusion)
    isnothing(hyperdiffusion) && return dydt
    (; κ₄, u, χu, χθ, ∇⁴u, ∇⁴θ, buffer_χu, buffer_χθ) = hyperdiffusion
    lgeom = Fields.local_geometry_field(axes(y.ρ))
    @. u = Geometry.Covariant12Vector(y.ρu / y.ρ, lgeom)
    Operators.vector_laplacian!(χu, u)
    Operators.scalar_laplacian!(χθ, Base.broadcasted(/, y.ρθ, y.ρ))
    # Continuity of the intermediates between the passes: a DSS on CG, a no-op
    # on DG, where the first pass already coupled the elements.
    Spaces.weighted_dss!(χu => buffer_χu, χθ => buffer_χθ)
    Operators.vector_laplacian!(∇⁴u, χu)
    Operators.scalar_laplacian!(∇⁴θ, χθ; weight = y.ρ)
    # `ρu` is the prognostic, so the velocity tendency is weighted by ρ; the
    # tracer's weight is already inside its second pass.
    @. dydt.ρu -= κ₄ * y.ρ * Geometry.UVVector(∇⁴u, lgeom)
    @. dydt.ρθ -= κ₄ * ∇⁴θ
    return dydt
end

# The discretization switch: on CG this is a DSS (with its exchange buffer),
# on DG the interface- and boundary-flux completion. `numflux` is passed
# unconditionally; CG ignores it.
completion =
    Operators.tendency_completion(dydt; numflux, boundary_numflux)

# The modal cutoff filter is applied to the completed tendency, so it acts on
# the same quantity on both discretizations.
filter_matrix =
    iszero(config.filter_order) ? nothing :
    Quadratures.cutoff_filter_matrix(
        Float64,
        Spaces.quadrature_style(space),
        config.filter_order,
    )

# One tendency for both discretizations: an element-local weak-form divergence
# of the physical flux, completed across element interfaces by `completion`.
function rhs!(dydt, y, p, t)
    (; parameters, completion, Ispace, filter_matrix, hyperdiffusion) = p
    wdiv = Operators.Divergence{Operators.WeakForm}()
    rparameters = Ref(parameters)

    if isnothing(Ispace)
        @. dydt = -wdiv(flux(y, rparameters))
    else
        # Over-integration: evaluate the quadratic flux on a finer quadrature
        # and restrict the divergence back, de-aliasing the product.
        Iop = Operators.Interpolate(Ispace)
        Rop = Operators.Restrict(axes(y))
        @. dydt = -Rop(wdiv(flux(Iop(y), rparameters)))
    end

    hyperdiffusion_tendency!(dydt, y, hyperdiffusion)
    Operators.complete_tendency!(completion, dydt, y, parameters)

    isnothing(filter_matrix) ||
        Operators.tensor_product!(Fields.field_values(dydt), filter_matrix)
    return dydt
end

p = (; parameters, completion, Ispace, filter_matrix, hyperdiffusion)
rhs!(dydt, y0, p, 0.0);

# Solve the ODE operator
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs!),
    y0,
    (0.0, config.tspan),
    p,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = 0.02,
    saveat = collect(0.0:1.0:config.tspan),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = discretization isa Spaces.CG ? "cg" : "dg_$(numflux_name)"
if boundary_name != ""
    dir = "$(dir)_$(boundary_name)"
end
if use_hyperdiffusion
    dir = "$(dir)_hyperdiffusion"
end
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]

using Test

@testset "energy budget" begin
    # On a periodic domain with DG, the numerical flux penalizes the jump
    # across each interface and nothing else touches the energy, so it may only
    # fall (measured drift over t = 200: -3.0e-4 for `rusanov`, -3.3e-4 for
    # `roe` — larger than in the CG case because the roll-up puts real
    # structure on the interfaces for the penalty to act on). A wall does work
    # on the domain, so that argument does not apply to `noslip`, which gains
    # 4.2e-4 instead.
    if is_periodic && discretization isa Spaces.DG
        @test Es[end] ≤ Es[1] * (1 + sqrt(eps()))
    end
    # CG conserves total energy up to time-integration error (measured drift
    # over t = 80: +5e-5). Either way the drift must stay small — including
    # with `hyperdiffusion`, which is dissipative by construction and takes the
    # drift to -3.5e-4 (CG) and -3.0e-4 (DG `rusanov`).
    @test abs(Es[end] - Es[1]) / Es[1] < 1e-3
end

@testset "mass and tracer conservation" begin
    masses = [sum(y.ρ) for y in sol.u]
    tracers = [sum(y.ρθ) for y in sol.u]
    mass_drift = maximum(abs, masses .- masses[1]) / masses[1]
    tracer_drift = maximum(abs, tracers .- tracers[1])
    if is_periodic
        # The weak-form divergence telescopes across the periodic domain under
        # both completions — DSS on CG, a conservative single-valued interface
        # flux on DG — so both hold to roundoff, and identically for `rusanov`
        # and `roe`, since conservation is a property of the assembly and not
        # of the flux (measured: 5e-14 (CG) / 1.4e-14 (DG) relative in ρ, and
        # 2e-12 / 1e-13 absolute in ρθ, whose exact integral is zero because
        # θ = sin(k y) over a whole number of periods).
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
    # must grow as the jet rolls up into vortices. Without this the energy
    # check above is passed just as well by a flow that never destabilizes.
    # Measured max|v| over the run: 0.050 at t = 0, 0.78 at t = 80 for CG;
    # peaking at 0.66 for `rusanov`, 0.71 for `roe` and 0.65 for `roe noslip`.
    speeds = cross_jet_speed.(sol.u)
    @test maximum(speeds) > config.rollup_factor * speeds[1]
    # Rolling the jet up draws the tracer into filaments the mesh cannot
    # resolve, and there is no limiter, so θ overshoots its initial [-1, 1]
    # range (measured: [-3.1, 2.7] for CG; ±3.7 for `rusanov`, ±2.6 for `roe`,
    # ±2.0 with walls). `hyperdiffusion` removes most of that overshoot —
    # [-1.2, 1.2] for CG, [-1.1, 1.9] for DG `rusanov` — which is what it is
    # there for, though it bounds nothing: see `limiters_advection.jl` for the
    # limited transport that does hold the bounds.
    θ_end = sol.u[end].ρθ ./ sol.u[end].ρ
    @test maximum(abs, θ_end) < 5
end
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

include(joinpath(@__DIR__, "..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
