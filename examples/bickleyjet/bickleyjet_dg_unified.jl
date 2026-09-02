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

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

const context = ClimaComms.SingletonCommsContext()

# ---------------------------------------------------------------------------
# Equation set types — carry all physical parameters; also serve as dispatch
# targets so that flux, wavespeed, init_state, etc. resolve via Julia dispatch
# rather than runtime if-else on strings.
# ---------------------------------------------------------------------------

abstract type AbstractEquationSet end

struct ShallowWaterEquations <: AbstractEquationSet
    ϵ::Float64   # perturbation amplitude
    l::Float64   # Gaussian width
    k::Float64   # sinusoidal wavenumber
    ρ₀::Float64  # reference depth/density
    g::Float64   # gravitational acceleration
end

# Compressible Euler (Souza et al. 2023): state = (ρ, ρu, ρe), R = 1 (normalized).
struct CompressibleEulerEquations <: AbstractEquationSet
    ϵ::Float64
    l::Float64
    k::Float64
    ρ₀::Float64
    T₀::Float64  # reference temperature (with R = 1)
    γ::Float64   # ratio of specific heats
end

# Stratified compressible Euler in 2D (y = vertical): hydrostatic isentropic
# background + localized θ perturbation.  Shared by RTB and density current.
abstract type AbstractStratifiedEuler2D <: AbstractEquationSet end

# Rising thermal bubble (Klemp & Wilhelmson 1978 / Straka et al. 1993):
# state = (ρ, ρu, ρe), physical units.  y-axis treated as the vertical (z).
# Hydrostatic isentropic background + warm bubble perturbation in θ.
struct RisingThermalBubbleEquations <: AbstractStratifiedEuler2D
    γ::Float64      # ratio of specific heats
    Rgas::Float64   # specific gas constant (J kg⁻¹ K⁻¹)
    cₚ::Float64     # specific heat at constant pressure = γ*Rgas/(γ-1)
    g::Float64      # gravitational acceleration (m s⁻²)
    p_ref::Float64  # reference pressure (Pa)
    θ₀::Float64     # background potential temperature (K)
    θ_pert::Float64 # max potential temperature perturbation (K)
    xc::Float64     # bubble centre x (m)
    zc::Float64     # bubble centre z / y (m)
    xr::Float64     # bubble x half-width (m)
    zr::Float64     # bubble z half-width (m)
end

# Falling density current (Straka et al. 1993): cold cosine-bell θ perturbation
# on the same hydrostatic isentropic background.
struct FallingDensityCurrentEquations <: AbstractStratifiedEuler2D
    γ::Float64
    Rgas::Float64
    cₚ::Float64
    g::Float64
    p_ref::Float64
    θ₀::Float64
    θ_pert::Float64  # negative for a cold dome (e.g. -15 K)
    xc::Float64
    zc::Float64
    xr::Float64
    zr::Float64
end

# Isentropic hydrostatic background state helpers.
# The Exner pressure Π(z) = 1 - g·z/(cₚ·θ₀) satisfies dp_bg/dz = -ρ_bg·g exactly.
background_pressure(z, eq::AbstractStratifiedEuler2D) =
    eq.p_ref * (1 - eq.g * z / (eq.cₚ * eq.θ₀))^(eq.cₚ / eq.Rgas)

function background_density(z, eq::AbstractStratifiedEuler2D)
    Π = 1 - eq.g * z / (eq.cₚ * eq.θ₀)
    T_bg = eq.θ₀ * Π
    return background_pressure(z, eq) / (eq.Rgas * T_bg)
end

function theta_perturbation(x, z, eq::RisingThermalBubbleEquations)
    L = sqrt(((x - eq.xc) / eq.xr)^2 + ((z - eq.zc) / eq.zr)^2)
    return L ≤ 1 ? eq.θ_pert * (1 - L) : 0.0
end

function theta_perturbation(x, z, eq::FallingDensityCurrentEquations)
    r = sqrt(((x - eq.xc) / eq.xr)^2 + ((z - eq.zc) / eq.zr)^2)
    return r < 1.0 ? eq.θ_pert / 2 * (1 + cospi(r)) : 0.0
end

# ---------------------------------------------------------------------------
# Initial conditions — dispatched on equation set
# ---------------------------------------------------------------------------

function bickley_velocity(coord, eq::AbstractEquationSet)
    x, y = coord.x, coord.y
    U₁ = cosh(y)^(-2)
    gaussian = exp(-(y + eq.l / 10)^2 / 2eq.l^2)
    u₁′ = gaussian * (y + eq.l / 10) / eq.l^2 * cos(eq.k * x) * cos(eq.k * x)
    u₁′ += eq.k * gaussian * cos(eq.k * x) * sin(eq.k * y)
    u₂′ = -eq.k * gaussian * sin(eq.k * x) * cos(eq.k * y)
    return Geometry.UVVector(U₁ + eq.ϵ * u₁′, eq.ϵ * u₂′)
end

function init_state(coord, eq::ShallowWaterEquations)
    ρ = eq.ρ₀
    u = bickley_velocity(coord, eq)
    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * sin(eq.k * coord.y))
end

function init_state(coord, eq::CompressibleEulerEquations)
    ρ = eq.ρ₀
    u = bickley_velocity(coord, eq)
    ρe = ρ * eq.T₀ / (eq.γ - 1) + ρ * (u.u^2 + u.v^2) / 2
    return (ρ = ρ, ρu = ρ * u, ρe = ρe, ρθ = ρ * sin(eq.k * coord.y))
end

function init_state(coord, eq::AbstractStratifiedEuler2D)
    x = coord.x; z = coord.y   # y-axis is the vertical
    γ, Rgas, g, p_ref, θ₀ = eq.γ, eq.Rgas, eq.g, eq.p_ref, eq.θ₀
    cₚ = eq.cₚ
    p_bg = background_pressure(z, eq)
    Π    = (p_bg / p_ref)^(Rgas / cₚ)
    δθ   = theta_perturbation(x, z, eq)
    T_total = (θ₀ + δθ) * Π
    ρ  = p_bg / (Rgas * T_total)
    ρe = p_bg / (γ - 1) + ρ * g * z
    return (ρ = ρ, ρu = ρ * Geometry.UVVector(0.0, 0.0), ρe = ρe)
end

# ---------------------------------------------------------------------------
# Physical flux F(U) — dispatched on equation set
# ---------------------------------------------------------------------------

function flux(state, eq::ShallowWaterEquations, _)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    return (ρ = ρu, ρu = (ρu ⊗ u) + (eq.g * ρ^2 / 2) * I, ρθ = ρθ * u)
end

function flux(state, eq::CompressibleEulerEquations, _)
    ρ, ρu, ρe, ρθ = state.ρ, state.ρu, state.ρe, state.ρθ
    u = ρu / ρ
    pres = (eq.γ - 1) * (ρe - ρ * (u.u^2 + u.v^2) / 2)
    return (ρ = ρu, ρu = (ρu ⊗ u) + pres * I, ρe = u * (ρe + pres), ρθ = ρθ * u)
end

function flux(state, eq::AbstractStratifiedEuler2D, coord)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    z = coord.y
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    pres = (eq.γ - 1) * (ρe - ρ * KE - ρ * eq.g * z)
    # Perturbation pressure in momentum flux: ∇p' + (ρ-ρ_bg)g balances for
    # hydrostatic background without large ∇p vs -ρg cancellation at GLL nodes.
    p_pert = pres - background_pressure(z, eq)
    return (ρ = ρu, ρu = (ρu ⊗ u) + p_pert * I, ρe = u * (ρe + pres))
end

# ---------------------------------------------------------------------------
# Thermodynamic pressure — dispatched on equation set
# ---------------------------------------------------------------------------

function pressure(state, eq::CompressibleEulerEquations, _)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    return (eq.γ - 1) * (ρe - ρ * KE)
end

function pressure(state, eq::AbstractStratifiedEuler2D, coord)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    z = coord.y
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    return (eq.γ - 1) * (ρe - ρ * KE - ρ * eq.g * z)
end

# Perturbation pressure for K-G face momentum flux (matches volume `flux`).
function momentum_pressure(state, eq::AbstractStratifiedEuler2D, coord)
    z = coord.y
    return pressure(state, eq, coord) - background_pressure(z, eq)
end

function face_sound_speed(state, eq::AbstractStratifiedEuler2D, coord)
    ρ = state.ρ
    z = coord.y
    p_bg = background_pressure(z, eq)
    ρ_bg = background_density(z, eq)
    ρ_floor = max(ρ, ρ_bg / 10)
    c_bg = sqrt(eq.γ * p_bg / ρ_floor)
    p = pressure(state, eq, coord)
    c_phys = p > 0 ? sqrt(eq.γ * p / ρ_floor) : c_bg
    return max(c_bg, c_phys)
end

# ---------------------------------------------------------------------------
# Wave speed for Rusanov — dispatched on equation set
# ---------------------------------------------------------------------------

function wavespeed(state, eq::ShallowWaterEquations, _)
    sqrt(eq.g)
end

function wavespeed(state, eq::CompressibleEulerEquations, _)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    pres = (eq.γ - 1) * (ρe - ρ * KE)
    sqrt(eq.γ * pres / ρ)
end

function wavespeed(state, eq::AbstractStratifiedEuler2D, coord)
    ρ, ρu = state.ρ, state.ρu
    z = coord.y
    u = ρu / ρ
    u_mag = sqrt(u.u^2 + u.v^2)
    return u_mag + face_sound_speed(state, eq, coord)
end

# ---------------------------------------------------------------------------
# Entropy variables — dispatched on equation set (stored in EntropyConservingFlux)
# ---------------------------------------------------------------------------

function entropy_variables(state, eq::CompressibleEulerEquations, _)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    p = (eq.γ - 1) * (ρe - ρ * KE)
    s̃ = log(p) - eq.γ * log(ρ)  # specific entropy (Souza et al. 2023)
    T_s = p / ρ
    return (
        (eq.γ - s̃) / (eq.γ - 1) - KE / T_s,
        u.u / T_s,
        u.v / T_s,
        -1.0 / T_s,
        0.0,   # ρθ is a passive tracer; its entropy variable is zero
    )
end

function entropy_variables(state, eq::AbstractStratifiedEuler2D, coord)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    z = coord.y
    u = ρu / ρ
    KE = (u.u^2 + u.v^2) / 2
    p = (eq.γ - 1) * (ρe - ρ * KE - ρ * eq.g * z)
    s̃ = log(p) - eq.γ * log(ρ)
    T_s = p / ρ
    return (
        (eq.γ - s̃) / (eq.γ - 1) - KE / T_s,
        u.u / T_s,
        u.v / T_s,
        -1.0 / T_s,
    )
end

# Roe-averaged state: density-weighted average, same formula for all equation sets
roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

# ---------------------------------------------------------------------------
# Numerical flux factory — dispatched on (equation set type, flux name).
# Adding a new combination requires only a new make_numflux method; rhs! is
# unchanged.
# ---------------------------------------------------------------------------

make_numflux(::AbstractEquationSet, ::Val{:central}) =
    Operators.CentralNumericalFlux(flux)

make_numflux(::AbstractEquationSet, ::Val{:rusanov}) =
    Operators.RusanovNumericalFlux(flux, wavespeed)

# Shallow-water Roe: characteristic decomposition for (ρ, ρu, ρθ) system
make_numflux(::ShallowWaterEquations, ::Val{:roe}) =
    Operators.RoeNumericalFlux(flux, roe_average)

# Compressible-Euler Roe: Kennedy-Gruber KEP central + Roe dissipation (Souza et al. Eqs 40-42)
make_numflux(::CompressibleEulerEquations, ::Val{:roe}) =
    Operators.EntropyConservingFlux(flux, entropy_variables, roe_average, pressure)

make_numflux(eq::AbstractStratifiedEuler2D, ::Val{:roe}) =
    Operators.EntropyConservingFlux(
        flux,
        entropy_variables,
        roe_average;
        pressure_fn = pressure,
        # p′ in K-G momentum and Roe wave amplitudes (roe_pressure_fn defaults to this)
        momentum_pressure_fn = momentum_pressure,
        sound_speed_fn = face_sound_speed,
    )

make_numflux(eq::AbstractStratifiedEuler2D, ::Val{:rusanov}) =
    Operators.RusanovNumericalFlux(flux, wavespeed)

# ---------------------------------------------------------------------------
# Body-force source terms — dispatched on equation set.
# source_term(state, eq) returns a NamedTuple of source contributions (same
# structure as the state), to be applied as:
#   dydt .+= source_term.(y, Ref(eq)) .* lgeom.WJ
# (consistent with the weak-form convention used for the flux volume term).
# The default is a zero contribution (no body force).
# ---------------------------------------------------------------------------

source_term(state, ::ShallowWaterEquations, _) =
    (ρ = 0.0, ρu = Geometry.UVVector(0.0, 0.0), ρθ = 0.0)

source_term(state, ::CompressibleEulerEquations, _) =
    (ρ = 0.0, ρu = Geometry.UVVector(0.0, 0.0), ρe = 0.0, ρθ = 0.0)

function source_term(state, eq::AbstractStratifiedEuler2D, coord)
    ρ, ρu = state.ρ, state.ρu
    ρ_bg = background_density(coord.y, eq)
    v = ρu.v / ρ
    buoy = -(ρ - ρ_bg) * eq.g
    return (
        ρ  = 0.0,
        ρu = Geometry.UVVector(0.0, buoy),
        ρe = buoy * v,
    )
end

# ---------------------------------------------------------------------------
# Hyperdiffusion — element-local biharmonic for periodic cases; omitted for stratified
# (stabilisation from entropy-stable / Rusanov face fluxes).
# ---------------------------------------------------------------------------

function add_hyperdiffusion!(dydt, y, eq::ShallowWaterEquations, space)
    Δx = Spaces.node_horizontal_length_scale(space)
    κ₄ = 0.0015 * sqrt(eq.g * eq.ρ₀) * Δx^3
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()

    q_θ = map(s -> s.ρθ / s.ρ, y)
    χ_θ = @. wdiv(grad(q_θ))
    lap2_θ = @. wdiv(grad(χ_θ))
    @. dydt.ρθ = dydt.ρθ - κ₄ * lap2_θ

    u_vel = map(s -> s.ρu.u / s.ρ, y)
    v_vel = map(s -> s.ρu.v / s.ρ, y)
    χ_u = @. wdiv(grad(u_vel))
    χ_v = @. wdiv(grad(v_vel))
    lap2_u = @. wdiv(grad(χ_u))
    lap2_v = @. wdiv(grad(χ_v))
    @. dydt.ρu = dydt.ρu - Geometry.UVVector(κ₄ * lap2_u, κ₄ * lap2_v)
end

function add_hyperdiffusion!(dydt, y, eq::CompressibleEulerEquations, space)
    Δx = Spaces.node_horizontal_length_scale(space)
    κ₄ = 0.0015 * sqrt(eq.γ * eq.T₀) * Δx^3
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()

    e_spec = map(s -> s.ρe / s.ρ, y)
    χ_e = @. wdiv(grad(e_spec))
    lap2_e = @. wdiv(grad(χ_e))
    @. dydt.ρe = dydt.ρe - κ₄ * lap2_e

    q_θ = map(s -> s.ρθ / s.ρ, y)
    χ_θ = @. wdiv(grad(q_θ))
    lap2_θ = @. wdiv(grad(χ_θ))
    @. dydt.ρθ = dydt.ρθ - κ₄ * lap2_θ

    u_vel = map(s -> s.ρu.u / s.ρ, y)
    v_vel = map(s -> s.ρu.v / s.ρ, y)
    χ_u = @. wdiv(grad(u_vel))
    χ_v = @. wdiv(grad(v_vel))
    lap2_u = @. wdiv(grad(χ_u))
    lap2_v = @. wdiv(grad(χ_v))
    @. dydt.ρu = dydt.ρu - Geometry.UVVector(κ₄ * lap2_u, κ₄ * lap2_v)
end

add_hyperdiffusion!(dydt, y, ::AbstractStratifiedEuler2D, space) = dydt

function rhs!(dydt, y, (eq, numflux, wall_bc, space), t)
    wdiv = Operators.WeakDivergence()
    lgeom = Fields.local_geometry_field(y)
    coord = Fields.coordinate_field(y)

    dydt .= wdiv.(flux.(y, Ref(eq), coord)) .* (.-(lgeom.WJ))

    Operators.add_numerical_flux_interior!(numflux, dydt, y, eq, coord)

    if wall_bc !== nothing
        Operators.add_numerical_flux_boundary!(numflux, wall_bc, dydt, y, eq, coord)
    end

    dydt .+= source_term.(y, Ref(eq), coord) .* lgeom.WJ

    dydt_data = Fields.field_values(dydt) ./ Spaces.local_geometry_data(space).WJ
    M = Quadratures.cutoff_filter_matrix(Float64, Spaces.quadrature_style(space), 3)
    Operators.tensor_product!(dydt_data, M)
    Fields.field_values(dydt) .= dydt_data
    add_hyperdiffusion!(dydt, y, eq, space)
    return dydt
end

# ---------------------------------------------------------------------------
# Output — dispatched on equation set to choose the appropriate diagnostic
# ---------------------------------------------------------------------------

function save_output(sol, eq::ShallowWaterEquations, path)
    ρ0 = first(sol.u).ρ
    ρmin, ρmax = extrema(ρ0)
    ρ_pad = 0.05 * max(eq.ρ₀, abs(ρmax - ρmin), 1.0)
    clim_ρ = (ρmin - ρ_pad, ρmax + ρ_pad)

    anim_ρ = Plots.@animate for u in sol.u
        Plots.plot(u.ρ, clim = clim_ρ, title = "Density ρ", colorbar = true)
    end
    Plots.mp4(anim_ρ, joinpath(path, "density.mp4"), fps = 10)

    anim_θ = Plots.@animate for u in sol.u
        Plots.plot(u.ρθ, clim = (-1, 1), title = "Tracer ρθ", colorbar = true)
    end
    Plots.mp4(anim_θ, joinpath(path, "tracer.mp4"), fps = 10)

    Es = [sum(state -> (state.ρu.u^2 + state.ρu.v^2) / (2 * state.ρ) + eq.g * state.ρ^2 / 2, u) for u in sol.u]
    Plots.png(Plots.plot(Es, ylabel = "Total energy", xlabel = "Time step"), joinpath(path, "energy.png"))
    return joinpath(path, "tracer.mp4")
end

function save_output(sol, eq::CompressibleEulerEquations, path)
    anim_ρ = Plots.@animate for u in sol.u
        Plots.plot(u.ρ, clim = (0.9, 1.1))
    end
    Plots.mp4(anim_ρ, joinpath(path, "density.mp4"), fps = 10)

    anim_θ = Plots.@animate for u in sol.u
        Plots.plot(map(s -> s.ρθ / s.ρ, u), clim = (-1, 1), title = "Tracer θ")
    end
    Plots.mp4(anim_θ, joinpath(path, "tracer.mp4"), fps = 10)

    function math_entropy(state, eq)
        ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
        u = ρu / ρ; KE = (u.u^2 + u.v^2) / 2
        pres = (eq.γ - 1) * (ρe - ρ * KE)
        return -ρ * (log(pres) - eq.γ * log(ρ)) / (eq.γ - 1)
    end
    Ss = [sum(s -> math_entropy(s, eq), u) for u in sol.u]
    Plots.png(Plots.plot(Ss, ylabel = "Total entropy η", xlabel = "Time step"), joinpath(path, "entropy.png"))
    return joinpath(path, "tracer.mp4")
end

function save_output(sol, eq::AbstractStratifiedEuler2D, path)
    # Compute θ = T * (p_ref/p)^((γ-1)/γ)  from conserved variables.
    # Total energy formulation: p = (γ-1)(ρe - ρKE - ρgz).
    function potential_temperature(state, coord, eq)
        ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
        z = coord.y
        u = ρu / ρ
        KE = (u.u^2 + u.v^2) / 2
        p  = (eq.γ - 1) * (ρe - ρ * KE - ρ * eq.g * z)
        T  = p / (ρ * eq.Rgas)
        return T * (eq.p_ref / p)^((eq.γ - 1) / eq.γ)
    end

    θ_pert_field(u) =
        potential_temperature.(u, Fields.coordinate_field(u), Ref(eq)) .- eq.θ₀

    θ0_field = θ_pert_field(first(sol.u))
    θmin, θmax = extrema(θ0_field)
    pad = 0.5
    clim = (min(eq.θ_pert - pad, θmin - pad), max(pad, θmax + pad))

    anim = Plots.@animate for u in sol.u
        Plots.plot(θ_pert_field(u), clim = clim,
                   title = "θ' (K)", colorbar = true)
    end
    Plots.mp4(anim, joinpath(path, "theta_pert.mp4"), fps = 5)

    θmax = [maximum(θ_pert_field(u)) for u in sol.u]
    Plots.png(
        Plots.plot(θmax, ylabel = "max θ' (K)", xlabel = "Time step"),
        joinpath(path, "theta_max.png"),
    )
    return joinpath(path, "theta_max.png")
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

eq_name       = get(ARGS, 1, "sw")
flux_name     = get(ARGS, 2, "rusanov")
boundary_name = get(ARGS, 3, "")

eq = if eq_name == "rtb"
    # Rising thermal bubble: 20 km × 10 km domain, bubble at (10 km, 2 km), r = 2 km
    let γ = 1.4, Rgas = 287.0
        RisingThermalBubbleEquations(
            γ, Rgas, γ * Rgas / (γ - 1),   # γ, Rgas, cₚ
            9.8, 1e5, 300.0, 2.0,
            10000.0, 2000.0, 2000.0, 2000.0,
        )
    end
elseif eq_name in ("fdc", "dc", "density_current")
    # Straka et al. (1993) falling density current: 51.2 km × 6.4 km domain.
    let γ = 1.4, Rgas = 287.0
        FallingDensityCurrentEquations(
            γ, Rgas, γ * Rgas / (γ - 1),
            9.8, 1e5, 300.0, -15.0,
            25600.0, 2000.0, 4000.0, 2000.0,
        )
    end
elseif eq_name == "euler"
    CompressibleEulerEquations(0.1, 0.5, 0.5, 1.0, 1.0, 1.4)
else
    ShallowWaterEquations(0.1, 0.5, 0.5, 1.0, 10.0)
end

numflux = make_numflux(eq, Val(Symbol(flux_name)))

# Domain, mesh, and time-stepping parameters depend on the equation set.
stratified_slice_domain(Lx, Lz) = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(0.0),
        Geometry.XPoint(Lx),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(0.0),
        Geometry.YPoint(Lz),
        boundary_names = (:bottom, :top),
    ),
)

if eq_name == "rtb"
    Lx, Lz = 20000.0, 10000.0
    domain = stratified_slice_domain(Lx, Lz)
    n1, n2 = 20, 10
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    wall_bc = Operators.ReflectingWallBC()
    dt_sim   = 0.1
    t_end    = 1000.0
    saveat   = collect(0.0:10.0:t_end)
elseif eq_name in ("fdc", "dc", "density_current")
    Lx, Lz = 51200.0, 6400.0
    domain = stratified_slice_domain(Lx, Lz)
    n1, n2 = 96, 8
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    wall_bc = Operators.ReflectingWallBC()
    # CFL: dx ≈ Lx/(n1*(Nq-1)) ≈ 178 m, c_sound ≈ 347 m/s → dt ≲ 0.51 s
    dt_sim   = 0.05
    t_end    = 900.0
    saveat   = collect(0.0:30.0:t_end)
else
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
    n1, n2 = 64, 64
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    wall_bc  = boundary_name == "noslip" ? Operators.ReflectingWallBC() : nothing
    # CFL: dx halved vs 32×32 → dt halved from 0.02
    dt_sim   = 0.005
    t_end    = 200.0
    saveat   = collect(0.0:1.0:t_end)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(eq))

dydt = Fields.Field(similar(Fields.field_values(y0)), space)
rhs!(dydt, y0, (eq, numflux, wall_bc, space), 0.0)

prob = ODEProblem(rhs!, y0, (0.0, t_end), (eq, numflux, wall_bc, space))
sol = solve(
    prob,
    SSPRK33(),
    dt = dt_sim,
    saveat = saveat,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

dir = "dg_unified_$(eq_name)_$(flux_name)"
if boundary_name != ""
    dir = "$(dir)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

figpath = save_output(sol, eq, path)

function linkfig(figpath, alt = "")
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(relpath(figpath, joinpath(@__DIR__, "../..")), "Output")
