#=
Shallow-water equations on the cubed sphere with DG spectral elements
(no DSS) — the fast-turnaround companion to the CG `shallow_water.jl` and the
2D testbed for the DG face-flux machinery used by the hybrid DG-FD drivers
(`examples/hybrid/sphere/sphere_dg_fd_model.jl`).

  ∂t h = −∇·(h u)                                   (flux form + Rusanov)
  ∂t u = −∇(g (h + h_s) + |u|²/2) + u × (f + ζ k̂)   (vector-invariant)
  ζ    = k̂ · (∇ × u)

DG treatment (face quantities in the local orthonormal geographic frame):
  • h: flux-differencing (FDDG) volume terms with the Kennedy-Gruber
    two-point mass flux {h}{u}, and the same flux as the central part of the
    Rusanov-penalized interface flux (λ = |u| + √(g h)).
  • ∇E (Bernoulli) and ζ: element-local strong operators + symmetric central
    face lifting; λ-scaled jump penalties on the geographic components (u, v).
  • κ₄ biharmonic hyperdiffusion ONLY (no κ₂): two-pass element-local first
    Laplacian + SIPG (LDG-penalty) second pass, applied to h and (u, v);
    default κ₄ = explicit SIPG cap / 10 (cf. the DG-FD sphere findings — the
    CG value ν₄ h³ ≈ 5e16 is far above the explicit DG penalty limit).

Test cases (CASE env):
  • steady_state (default): Williamson et al. (1992) Test Case 2 — steady
    geostrophic zonal flow; the initial state is the exact solution, so the
    driver reports h/u error norms. 1 simulated day ≈ seconds of wall time.
    ALPHA rotates the flow relative to the cubed-sphere panels (e.g.
    ALPHA=45 exercises panel-edge crossings).
  • barotropic_instability: Galewsky et al. (2004) unstable mid-latitude jet;
    6 simulated days, minutes of wall time; plots the day-6 vorticity.

Run:
  julia --project=.buildkite examples/sphere/shallow_water_dg.jl
  CASE=barotropic_instability julia --project=.buildkite examples/sphere/shallow_water_dg.jl

Environment: CASE, ALPHA, HELEM, NPOLY, DT, T_END, KAPPA4, FILTER
=#

using LinearAlgebra: norm, norm_sqr, ×

import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies

import QuadGK
using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64
const CT3 = Geometry.Contravariant3Vector

# ---------------------------------------------------------------------------
# Parameters and case selection
# ---------------------------------------------------------------------------
const R = FT(6.37122e6)
const Ω = FT(7.292e-5)
const g = FT(9.80616)

const case = get(ENV, "CASE", "steady_state")
const α = parse(FT, get(ENV, "ALPHA", "0"))
const helem = parse(Int, get(ENV, "HELEM", "4"))
const npoly = parse(Int, get(ENV, "NPOLY", "4"))
const Δt = parse(FT, get(ENV, "DT", "200.0"))
const t_end_default = case == "steady_state" ? 86400.0 : 6 * 86400.0
const t_end = parse(FT, get(ENV, "T_END", string(t_end_default)))

# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------
context = ClimaComms.context()
domain = Domains.SphereDomain(R)
mesh = Meshes.EquiangularCubedSphere(domain, helem)
topology = Topologies.Topology2D(context, mesh)
space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{npoly + 1}())
coords = Fields.coordinate_field(space)
lgeom = Fields.local_geometry_field(space)

# Coriolis (α-rotated; in 2D W/Contravariant3/Covariant3 coincide)
const f_cor = @. CT3(
    2 * Ω * (-cosd(coords.long) * cosd(coords.lat) * sind(α) +
       sind(coords.lat) * cosd(α)),
)
const h_s = zeros(space)

# ---------------------------------------------------------------------------
# Initial conditions (copied from the CG shallow_water.jl test definitions)
# ---------------------------------------------------------------------------
function initial_state_steady_state()
    u0 = 2 * pi * R / (12 * 86400)
    h0 = FT(2.94e4) / g
    return map(lgeom) do local_geometry
        (; lat, long) = local_geometry.coordinates
        ϕ, λ = lat, long
        h =
            h0 -
            (R * Ω * u0 + u0^2 / 2) / g *
            (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))^2
        uλ = u0 * (cosd(α) * cosd(ϕ) + sind(α) * cosd(λ) * sind(ϕ))
        uϕ = -u0 * sind(α) * sind(λ)
        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλ, uϕ),
            local_geometry,
        )
        return (h = h, u = u)
    end
end

function initial_state_barotropic_instability()
    u_max = FT(80)
    αₚ = FT(19.09859)
    βₚ = FT(3.81971)
    h0 = FT(10158.18617)
    h_hat = FT(120)
    ϕ₀ = FT(25.71428)
    ϕ₁ = FT(64.28571)
    ϕ₂ = FT(45)
    eₙ = exp(-4 / (deg2rad(ϕ₁) - deg2rad(ϕ₀))^2)
    @assert α == 0 "barotropic_instability is defined for ALPHA=0"
    uλp(ϕ) =
        (u_max / eₙ) *
        exp(1 / (deg2rad(ϕ - ϕ₀) * deg2rad(ϕ - ϕ₁))) *
        (ϕ₀ < ϕ < ϕ₁)
    h_int(γ) =
        abs(γ) < 90 ? (2 * Ω * sind(γ) + uλp(γ) * tand(γ) / R) * uλp(γ) :
        zero(γ)
    return map(lgeom) do local_geometry
        (; lat, long) = local_geometry.coordinates
        ϕ, λ = lat, long
        h = h0 - (R / g) * (pi / 180) * QuadGK.quadgk(h_int, -90.0, ϕ)[1]
        if λ > 0
            λ -= 360
        end
        h += h_hat * cosd(ϕ) * exp(-(λ^2 / αₚ^2) - ((ϕ₂ - ϕ)^2 / βₚ^2))
        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(uλp(ϕ), zero(ϕ)),
            local_geometry,
        )
        return (h = h, u = u)
    end
end

initial_state() =
    case == "steady_state" ? initial_state_steady_state() :
    initial_state_barotropic_instability()

# All DG building blocks — the Kennedy-Gruber height fluxes, the central
# lifting / jump-penalty face functions, `lifting_correction`, and
# `ldg_laplacian_tendency` — come from ClimaCore's Operators module; no
# operators are defined in this driver.

const κ₄_cfl_cap = FT(
    Spaces.node_horizontal_length_scale(space)^3 / ((2 * npoly + 1)^2 * Δt),
)
const κ₄ = haskey(ENV, "KAPPA4") ? parse(FT, ENV["KAPPA4"]) : κ₄_cfl_cap / 10
const filter_Nc = parse(Int, get(ENV, "FILTER", "0"))

# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const hcurl = Operators.Curl()

function rhs!(dY, y, _, t)
    h = y.h
    u = y.u

    uv = @. Geometry.UVVector(u)
    u_sc = uv.components.data.:1
    v_sc = uv.components.data.:2
    λ_wave = @. sqrt(norm_sqr(uv)) + sqrt(g * h)

    # --- h: flux form, flux-differencing volume + KG/Rusanov interface ---
    state = map((hi, λi, uvi) -> (; h = hi, λ = λi, uv = uvi), h, λ_wave, uv)
    dh_mw = similar(h)
    dh_mw .= 0
    Operators.add_flux_differencing_divergence!(
        Operators.kennedy_gruber_height_flux,
        dh_mw,
        state,
    )
    Operators.add_numerical_flux_internal!(
        Operators.kennedy_gruber_rusanov_height,
        dh_mw,
        state,
    )
    @. dY.h = dh_mw / lgeom.WJ

    # --- u: vector-invariant ---
    ζ_sc = @. Geometry.WVector(hcurl(u)).components.data.:1
    ζ_sc .+=
        Operators.lifting_correction(Operators.central_curl3_lift, FT, u_sc, v_sc)
    ζ = @. CT3(Geometry.WVector(ζ_sc))

    E = @. g * (h + h_s) + norm_sqr(uv) / 2
    lift_E = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        E,
    )
    pen_u = Operators.lifting_correction(Operators.jump_penalty_lift, FT, u_sc, λ_wave)
    pen_v = Operators.lifting_correction(Operators.jump_penalty_lift, FT, v_sc, λ_wave)
    @. dY.u =
        -hgrad(E) + u × (f_cor + ζ) +
        Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(pen_u, pen_v) - lift_E,
        )

    # --- κ₄ hyperdiffusion (two-pass, SIPG-coupled; no κ₂) ---
    if κ₄ != 0
        τ_κ₄ = Operators.ldg_penalty_parameter(κ₄, space)
        χh = similar(h)
        @. χh = hwdiv(hgrad(h))
        χu = similar(u_sc)
        @. χu = hwdiv(hgrad(u_sc))
        χv = similar(v_sc)
        @. χv = hwdiv(hgrad(v_sc))
        @. dY.h -= $(Operators.ldg_laplacian_tendency(χh, nothing, κ₄, τ_κ₄))
        du4 = Operators.ldg_laplacian_tendency(χu, nothing, κ₄, τ_κ₄)
        dv4 = Operators.ldg_laplacian_tendency(χv, nothing, κ₄, τ_κ₄)
        @. dY.u -= Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(du4, dv4),
        )
    end

    # --- optional element-local cutoff filter ---
    if filter_Nc > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(space),
            filter_Nc,
        )
        for fld in (dY.h, dY.u)
            data = Fields.field_values(fld)
            Operators.tensor_product!(data, data, M)
        end
    end

    return dY
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
let
    h_node = Spaces.node_horizontal_length_scale(space)
    λ_max = 80 + sqrt(g * FT(10158))  # fastest jet + gravity-wave speed
    @info "DG shallow water setup" case α helem npoly Δt t_end κ₄ κ₄_cfl_cap h_node
    @info "Wave CFL estimate" horizontal = λ_max * Δt / h_node
end

Y = initial_state()
Y0 = copy(Y)
const mass_0 = sum(Y.h)

dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)
@info "Initial RHS" max_dh = maximum(abs, parent(dY.h)) max_du =
    maximum(abs, parent(dY.u))

prob = ODEProblem(rhs!, Y, (FT(0), t_end))
const dt_save = parse(FT, get(ENV, "DT_SAVE", string(min(t_end, 86400.0))))
sol = solve(prob, SSPRK33(), dt = Δt, saveat = dt_save)

Yend = sol.u[end]
@info "Conservation" mass_rel = (sum(Yend.h) - mass_0) / mass_0

if case == "steady_state"
    # The initial state is the exact steady solution: report error norms.
    herr = @. Yend.h - Y0.h
    uverr = @. Geometry.UVVector(Yend.u) - Geometry.UVVector(Y0.u)
    h_rel_l2 = norm(parent(herr)) / norm(parent(Y0.h))
    u_rel_max =
        maximum(abs, parent(uverr)) /
        maximum(abs, parent(Geometry.UVVector.(Y0.u)))
    @info "Steady-state (TC2) errors at t = $(sol.t[end])" h_rel_l2 u_rel_max
end

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
output_dir = joinpath(@__DIR__, "output", "shallow_water_dg_$case")
mkpath(output_dir)
Plots.png(Plots.plot(Yend.h), joinpath(output_dir, "h_end.png"))
ζ_end = @. Geometry.WVector(hcurl(Yend.u)).components.data.:1
Plots.png(Plots.plot(ζ_end), joinpath(output_dir, "vorticity_end.png"))
@info "Output written to $output_dir"
