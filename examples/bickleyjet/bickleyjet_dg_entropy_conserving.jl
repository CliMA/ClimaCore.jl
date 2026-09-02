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

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
const context = ClimaComms.SingletonCommsContext()

# Compressible Euler Bickley jet, following Souza et al. (2023, JAMES).
# State variables: (ρ, ρu, ρe) — density, momentum, total energy per unit volume.
# Equation of state: p = (γ-1)(ρe - ½ρ|u|²).
# Entropy: η = -ρs/(γ-1) where s = log(p/ρ^γ) (Boltzmann entropy, Souza et al.).
const parameters = (
    ϵ = 0.1,   # perturbation size for initial condition
    l = 0.5,   # Gaussian width
    k = 0.5,   # sinusoidal wavenumber
    ρ₀ = 1.0,  # reference density
    T₀ = 1.0,  # reference temperature (with R = 1)
    γ = 1.4,   # ratio of specific heats (diatomic ideal gas)
)

boundary_name = get(ARGS, 1, "")

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

n1, n2 = 32, 32
Nq = 4
mesh = Meshes.RectilinearMesh(domain, n1, n2)
grid_topology = Topologies.Topology2D(context, mesh)
quad = Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

function init_state(coord, p)
    x, y = coord.x, coord.y
    ρ = p.ρ₀
    U₁ = cosh(y)^(-2)

    gaussian = exp(-(y + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * x)
    u₁′ += p.k * gaussian * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * gaussian * sin(p.k * x) * cos(p.k * y)

    u = Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    KE = (u.u^2 + u.v^2) / 2

    # Internal energy: cv * T₀ = T₀ / (γ-1) with R = 1
    ρe = ρ * p.T₀ / (p.γ - 1) + ρ * KE

    return (ρ = ρ, ρu = ρ * u, ρe = ρe)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    pres = (p.γ - 1) * (ρe - ρ * KE)
    return (
        ρ  = ρu,
        ρu = (ρu ⊗ u) + pres * I,
        ρe = u * (ρe + pres),
    )
end

# Souza et al. (2023) entropy function: η = -ρs/(γ-1), s = log(p/ρ^γ).
# Discrete entropy should be non-increasing for the entropy-stable method.
function mathematical_entropy(state, p)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    pres = (p.γ - 1) * (ρe - ρ * KE)
    s = log(pres) - p.γ * log(ρ)
    return -ρ * s / (p.γ - 1)
end

function total_entropy(y, parameters)
    sum(state -> mathematical_entropy(state, parameters), y)
end

# Entropy variables for compressible Euler (Harten 1983; used in Souza et al. 2023):
#   v = ∂η/∂U where η = -ρs/(γ-1), U = (ρ, ρu₁, ρu₂, ρe)
# With T_s = p/ρ (= R·T, here R = 1):
#   v₁ = (γ - s̃)/(γ-1) - KE/T_s,  v₂ = u₁/T_s,  v₃ = u₂/T_s,  v₄ = -1/T_s
# where s̃ = log(p) - γ·log(ρ).  Both log(p) and log(ρ) are well-defined for ρ,p > 0.
function entropy_variables(state, params)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    p = (params.γ - 1) * (ρe - ρ * KE)
    s̃ = log(p) - params.γ * log(ρ)
    T_s = p / ρ
    return (
        (params.γ - s̃) / (params.γ - 1) - KE / T_s,
        u.u / T_s,
        u.v / T_s,
        -1.0 / T_s,
    )
end

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

# Kennedy-Gruber KEP interface flux + Rusanov dissipation (Souza et al. Eqs 40-42)
numflux = Operators.EntropyConservingFlux(flux, entropy_variables, roe_average)
wall_bc = boundary_name == "noslip" ? Operators.ReflectingWallBC() : nothing

function rhs!(dydt, y, (parameters, numflux, wall_bc), t)
    wdiv = Operators.WeakDivergence()
    local_geometry_field = Fields.local_geometry_field(y)

    dydt .= wdiv.(flux.(y, Ref(parameters))) .* (.-(local_geometry_field.WJ))

    Operators.add_numerical_flux_interior!(numflux, dydt, y, parameters)

    if wall_bc !== nothing
        Operators.add_numerical_flux_boundary!(numflux, wall_bc, dydt, y, parameters)
    end

    dydt_data = Fields.field_values(dydt) ./ Spaces.local_geometry_data(space).WJ
    M = Quadratures.cutoff_filter_matrix(
        Float64,
        Spaces.quadrature_style(space),
        3,
    )
    Operators.tensor_product!(dydt_data, M)
    return dydt
end

dydt = Fields.Field(similar(Fields.field_values(y0)), space)
rhs!(dydt, y0, (parameters, numflux, wall_bc), 0.0);

prob = ODEProblem(rhs!, y0, (0.0, 200.0), (parameters, numflux, wall_bc))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = collect(0.0:1.0:200.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dg_entropy_conserving"
if boundary_name != ""
    dir = "$(dir)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρ, clim = (0.9, 1.1))
end
Plots.mp4(anim, joinpath(path, "density.mp4"), fps = 10)

# Total entropy η should be non-increasing for an entropy-stable scheme
Ss = [total_entropy(u, parameters) for u in sol.u]
Plots.png(
    Plots.plot(Ss, ylabel = "Total entropy η", xlabel = "Time step"),
    joinpath(path, "entropy.png"),
)

function linkfig(figpath, alt = "")
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "entropy.png"), joinpath(@__DIR__, "../..")),
    "Total Entropy",
)
