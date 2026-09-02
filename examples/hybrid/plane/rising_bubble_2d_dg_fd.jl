#=
Stage B: Rising thermal bubble — DG horizontal + FD vertical
================================================================

Adapts Souza et al. (2023, JAMES) KEP Kennedy–Gruber + Roe face fluxes
(`Operators.EntropyConservingFlux`) to an Atmos-like hybrid mesh:

  • Horizontal: discontinuous Galerkin (no DSS), Souza interface flux with
    thermodynamic `p` in the energy flux and perturbation `p′` in momentum /
    Roe Δp (same stratified split as unified `rtb roe`).
  • Vertical: staggered FD matching the working CG+FD bubble /
    `density_current_2d_flux_form.jl` hydrostatics
    (`−∂z p − ρ ∂z Φ`, reflecting Neumann walls), with centered face fluxes.
  • Walls: reflecting slip — `ρw = 0`, `∇z = 0` via `SetGradient(0)`.
  • Stabilization: Nc=3 cutoff on the full residual; optional `KAPPA2` LDG
    (default 0 — Souza path needs no hyperdiffusion for pure DG).

Reference pure-DG run:
  julia --project=.buildkite examples/bickleyjet/bickleyjet_dg_unified.jl rtb roe

This hybrid:
  julia --project=.buildkite examples/hybrid/plane/rising_bubble_2d_dg_fd.jl

Environment: BOX=giraldo|rtb, HELEM, VELEM, NPOLY, DT, T_END, KAPPA2, FILTER
=#

using LinearAlgebra
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
import ClimaCore.Geometry: ⊗

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

# ---------------------------------------------------------------------------
# Mesh — Giraldo-style rising bubble box (matches CG bubble_2d_invariant_rhoe)
# Override with ENV HELEM/VELEM; domain via BOX=rtb for unified 20×10 km.
# ---------------------------------------------------------------------------
const box = get(ENV, "BOX", "rtb")  # "rtb" (unified) | "giraldo" (small CG box)
function default_domain(box)
    if box == "rtb"
        return (0.0, 20000.0), (0.0, 10000.0), 40, 40, 0.05, 2.0
    else
        # CG bubble: (−500,500)×(0,1000), θ′_max = 0.5 K
        return (-500.0, 500.0), (0.0, 1000.0), 40, 80, 0.02, 0.5
    end
end
const (xlim0, zlim0, helem0, velem0, dt0, θpert0) = default_domain(box)

function hvspace_2D(
    xlim = xlim0,
    zlim = zlim0,
    helem = helem0,
    velem = velem0,
    npoly = 3,
)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

const helem = parse(Int, get(ENV, "HELEM", string(helem0)))
const velem = parse(Int, get(ENV, "VELEM", string(velem0)))
const npoly = parse(Int, get(ENV, "NPOLY", "3"))
const Δt = parse(FT, get(ENV, "DT", string(dt0)))
hv_center_space, hv_face_space =
    hvspace_2D(xlim0, zlim0, helem, velem, npoly)

# ---------------------------------------------------------------------------
# Physics — Souza / unified RTB EOS
# ---------------------------------------------------------------------------
const MSLP = 1e5
const grav = 9.8
const R_d = 287.0
const γ = 1.4
const C_p = R_d * γ / (γ - 1)
const θ₀ = 300.0
const θ_pert = θpert0

# Equation parameters passed into EntropyConservingFlux (Souza / unified).
const eq = (; γ = γ, Rgas = R_d, cₚ = C_p, g = grav, p_ref = MSLP, θ₀ = θ₀)

Φ(z) = grav * z

background_pressure(z) = MSLP * (1 - Φ(z) / (C_p * θ₀))^(C_p / R_d)
background_density(z) =
    background_pressure(z) / (R_d * θ₀ * (1 - Φ(z) / (C_p * θ₀)))

# Thermodynamic pressure: p = (γ−1)(ρe − ρKE − ρΦ)
pressure_eos(ρ, ρe, K, z) = (γ - 1) * (ρe - ρ * K - ρ * Φ(z))

_ke_u(u) = (u.u^2) / 2

function thermo_pressure(state, eq, z)
    u = state.ρu / state.ρ
    K = _ke_u(u) + state.w^2 / 2
    return pressure_eos(state.ρ, state.ρe, K, z)
end

function momentum_pressure(state, eq, z)
    return thermo_pressure(state, eq, z) - background_pressure(z)
end

function face_sound_speed(state, eq, z)
    ρ = state.ρ
    p_bg = background_pressure(z)
    ρ_bg = background_density(z)
    ρ_floor = max(ρ, ρ_bg / 10)
    c_bg = sqrt(eq.γ * p_bg / ρ_floor)
    p = thermo_pressure(state, eq, z)
    c_phys = p > 0 ? sqrt(eq.γ * p / ρ_floor) : c_bg
    return max(c_bg, c_phys)
end

# Required by EntropyConservingFlux; Roe dissipation does not use it.
function entropy_variables(state, eq, z)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    K = _ke_u(u) + state.w^2 / 2
    p = pressure_eos(ρ, ρe, K, z)
    s̃ = log(p) - eq.γ * log(ρ)
    T_s = p / ρ
    return (
        (eq.γ - s̃) / (eq.γ - 1) - K / T_s,
        u.u / T_s,
        -1.0 / T_s,
    )
end

roe_average(ρ⁻, ρ⁺, a⁻, a⁺) =
    (sqrt(ρ⁻) * a⁻ + sqrt(ρ⁺) * a⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

# Horizontal physical flux Fˣ (Souza / unified): p′ in momentum, p in energy.
function flux_h(state, eq, z)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    K = _ke_u(u) + state.w^2 / 2
    p = pressure_eos(ρ, ρe, K, z)
    p′ = p - background_pressure(z)
    return (ρ = ρu, ρu = (ρu ⊗ u) + p′ * I, ρe = u * (ρe + p))
end

const numflux_h = Operators.EntropyConservingFlux(
    flux_h,
    entropy_variables,
    roe_average;
    pressure_fn = thermo_pressure,
    momentum_pressure_fn = momentum_pressure,
    sound_speed_fn = face_sound_speed,
)

# Face ρw: central + |uₙ| dissipation (shear / contact analog on U-faces)
function ρw_roe(normal, (ρw⁻, uₕ⁻), (ρw⁺, uₕ⁺))
    un̄ = ((uₕ⁻ + uₕ⁺) / 2)' * normal
    return un̄ * (ρw⁻ + ρw⁺) / 2 - abs(un̄) / 2 * (ρw⁺ - ρw⁻)
end

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
function init_rising_bubble_2d(x, z)
    # Always use Souza/unified resting form: p = p_bg(θ₀), ρ from total θ.
    # (Giraldo-local-θ Exner invents a large p′ vs background and is
    # hydrostatically inconsistent with the p′/buoyancy RHS.)
    if box == "rtb"
        xc, zc, xr, zr = 10000.0, 2000.0, 2000.0, 2000.0
        L = sqrt(((x - xc) / xr)^2 + ((z - zc) / zr)^2)
        δθ = L ≤ 1 ? 0.5 * θ_pert * (1 + cospi(L)) : 0.0
    else
        xc, zc, rc = 0.0, 350.0, 250.0
        r = sqrt((x - xc)^2 + (z - zc)^2)
        δθ = r < rc ? 0.5 * θ_pert * (1 + cospi(r / rc)) : 0.0
    end
    p_bg = background_pressure(z)
    Π = (p_bg / MSLP)^(R_d / C_p)
    T = (θ₀ + δθ) * Π
    ρ = p_bg / (R_d * T)
    ρe = p_bg / (γ - 1) + ρ * Φ(z)
    return (ρ = ρ, ρe = ρe)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_rising_bubble_2d(coord.x, coord.z), coords)
ρw = map(_ -> Geometry.WVector(FT(0)), face_coords)
Y = Fields.FieldVector(
    Yc = Yc,
    ρuₕ = Yc.ρ .* Ref(Geometry.UVector(FT(0))),
    ρw = ρw,
)

const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

# ---------------------------------------------------------------------------
# Optional LDG (off by default — Souza RTB uses face dissipation only)
# κ₄ defaults to ν₄_coeff × h_node³ (same h³ scaling as density_current_2d_dg_fd.jl).
# This provides minimal horizontal smoothing without second-order viscosity.
# ---------------------------------------------------------------------------
const κ₂ = parse(FT, get(ENV, "KAPPA2", "0.0"))
# Density-only LDG smoothing: separate from κ₂ so momentum/energy diffusion can be zero.
# Default is 0; set KAPPA2_RHO to add density-targeted LDG without physical viscosity.
const κ₂_ρ = parse(FT, get(ENV, "KAPPA2_RHO", string(κ₂)))
const h_node = FT(Spaces.node_horizontal_length_scale(
    Spaces.horizontal_space(hv_center_space),
))
const ν₄_coeff = parse(FT, get(ENV, "NU4", "10.5"))    # m/s; default ≈2e8 m⁴/s at HELEM=40
const κ₄_cfl_cap = FT(h_node^3 / ((2 * npoly + 1)^2 * Δt))  # SIPG explicit stability limit
const κ₄ = haskey(ENV, "KAPPA4") ? parse(FT, ENV["KAPPA4"]) :
    FT(min(ν₄_coeff * h_node^3, κ₄_cfl_cap))

# LDG / SIPG Laplacian tendency: provided by ClimaCore
# (`Operators.ldg_laplacian_tendency`).
const ldg_laplacian_tendency = Operators.ldg_laplacian_tendency

# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------
function rhs!(dY, Y, _, t)
    ρuₕ = Y.ρuₕ
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρuₕ = dY.ρuₕ
    dρw = dY.ρw
    ρ = Yc.ρ
    ρe = Yc.ρe
    z = coords.z

    hwdiv = Operators.WeakDivergence()
    lgeom_c = Fields.local_geometry_field(axes(ρ))
    lgeom_f = Fields.local_geometry_field(axes(ρw))

    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
        top = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
    )
    VanLeer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = Operators.MonotoneLocalExtrema(),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    # Reflecting wall Neumann (bubble_2d_invariant_rhoe)
    ∂f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    uₕ = @. ρuₕ / ρ
    w_c = @. Ic(ρw).components.data.:1 / ρ
    u_c = @. uₕ.components.data.:1
    K_full = @. (u_c^2 + w_c^2) / 2

    # --- Horizontal: Souza DG (default) or CG ContinuousDivergence+DSS ---
    horiz = get(ENV, "HORIZ", "dg")
    y = map(
        (ρi, ρei, ρui, wi) -> (; ρ = ρi, ρe = ρei, ρu = ρui, w = wi),
        ρ,
        ρe,
        ρuₕ,
        w_c,
    )
    if horiz == "cg"
        # Sanity check: CG-like horizontal (ContinuousDivergence + DSS).
        hdiv = Operators.Divergence()
        Fh = map((s, zi) -> flux_h(s, eq, zi), y, z)
        @. dYc.ρ = -hdiv(Fh).ρ
        @. dYc.ρe = -hdiv(Fh).ρe
        @. dρuₕ = -hdiv(Fh).ρu
        Spaces.weighted_dss!(dYc)
        Spaces.weighted_dss!(dρuₕ)
    else
        Fh = map((s, zi) -> flux_h(s, eq, zi), y, z)
        dy_mw = @. hwdiv(Fh) * (-(lgeom_c.WJ))
        Operators.add_numerical_flux_interior!(numflux_h, dy_mw, y, eq, z)
        @. dy_mw = dy_mw / lgeom_c.WJ
        @. dYc.ρ = dy_mw.ρ
        @. dYc.ρe = dy_mw.ρe
        @. dρuₕ = dy_mw.ρu
    end

    # --- Vertical FD: stratified p′ split (momentum uses p′ + buoyancy, energy full-p flux) ---
    p = @. pressure_eos(ρ, ρe, K_full, z)
    p′ = @. p - background_pressure(z)
    ρ_bg = @. background_density(z)
    h_tot = @. (ρe + p) / ρ

    Yfρ = @. If(ρ)
    w_face = @. ρw / Yfρ
    @. dYc.ρ -= vdivf2c(ρw)
    @. dYc.ρe -= vdivf2c(Yfρ * VanLeer(w_face, h_tot, Δt))
    @. dρuₕ -= Geometry.UVector(vdivf2c(Yfρ * VanLeer(w_face, u_c, Δt)))

    buoy = @. -(ρ - ρ_bg) * grav
    # Vertical self-advection ∂z(ρw²/ρ): use center density to avoid If(ρ)→0 blow-up.
    # Ic(ρw) ⊗ (Ic(ρw)/ρ) ≡ Ic(ρw)²/ρ at centers; same ∂z(ρw²/ρ) approximation.
    ρw_c = @. Ic(ρw)
    w_c_vert = @. ρw_c / ρ
    @. dρw = B(
        Geometry.project(Geometry.WAxis(), -(∂f(p′))) +
        If(Geometry.WVector(buoy)) -
        vvdivc2f(ρw_c ⊗ w_c_vert),
    )

    # Horizontal advection of ρw
    uₕf = @. If(uₕ)
    if horiz == "cg"
        hdiv = Operators.Divergence()
        @. dρw -= hdiv(uₕf ⊗ ρw)
        Spaces.weighted_dss!(dρw)
    else
        # dρw_mw/WJ is already the full weak tendency −∇·(uₕ ⊗ ρw) + face
        # dissipation (same convention as the center equations), so it is ADDED.
        dρw_mw = @. hwdiv(uₕf ⊗ ρw) * (-(lgeom_f.WJ))
        Operators.add_numerical_flux_interior!(ρw_roe, dρw_mw, ρw, uₕf)
        @. dρw += dρw_mw / lgeom_f.WJ
    end

    # Density-only LDG smoothing: applied independently of momentum/energy viscosity.
    if κ₂_ρ != 0
        τ_κ₂_ρ = Operators.ldg_penalty_parameter(κ₂_ρ, axes(ρ))
        _dρ_h = ldg_laplacian_tendency(ρ, nothing, κ₂_ρ, τ_κ₂_ρ)
        @. dYc.ρ += _dρ_h
    end
    if κ₂ != 0
        ∂c = Operators.GradientF2C()
        τ_κ₂ = Operators.ldg_penalty_parameter(κ₂, axes(ρ))
        # Materialize u_c into a proper scalar Field (similar(ρ) guarantees correct axes).
        # ldg_laplacian_tendency must be called outside @. (it's a Field-level function,
        # not element-wise); matches how the κ₄ block calls it.
        u_c_visc = similar(ρ)
        @. u_c_visc = uₕ.components.data.:1
        _du_h = ldg_laplacian_tendency(u_c_visc, ρ, κ₂, τ_κ₂)
        _de_h = ldg_laplacian_tendency(h_tot,    ρ, κ₂, τ_κ₂)
        @. dρuₕ   += Geometry.UVector(_du_h)
        @. dYc.ρe += _de_h
        # Vertical κ₂: horizontal coupling for ρuₕ and ρe; vertical FD for ρw.
        # ρw viscosity uses center-reconstructed velocity to avoid Yfρ→0 blow-up.
        @. dρuₕ += Geometry.UVector(uvdivf2c(κ₂ * (Yfρ * ∂f(uₕ))))
        w_f_safe = @. If(w_c_vert)
        @. dρw += vvdivc2f(κ₂ * (ρ * ∂c(w_f_safe)))
        @. dYc.ρe += vdivf2c(κ₂ * (Yfρ * ∂f(h_tot)))
    end

    if κ₄ != 0
        hgrad  = Operators.Gradient()
        τ_κ₄   = Operators.ldg_penalty_parameter(κ₄, axes(ρ))
        χ_htot = similar(h_tot); @. χ_htot = hwdiv(hgrad(h_tot))
        χ_u    = similar(u_c);   @. χ_u    = hwdiv(hgrad(u_c))
        χ_ρ    = similar(ρ);     @. χ_ρ    = hwdiv(hgrad(ρ))
        _de4 = ldg_laplacian_tendency(χ_htot, ρ,       κ₄, τ_κ₄)
        _du4 = ldg_laplacian_tendency(χ_u,    ρ,       κ₄, τ_κ₄)
        _dρ4 = ldg_laplacian_tendency(χ_ρ,    nothing, κ₄, τ_κ₄)
        @. dYc.ρe -= _de4
        @. dρuₕ   -= Geometry.UVector(_du4)
        @. dYc.ρ  -= _dρ4
    end

    # Optional element-local cutoff (unified RTB uses Nc=3; set FILTER=0 to disable)
    if parse(Int, get(ENV, "FILTER", "3")) > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(axes(ρ)),
            parse(Int, get(ENV, "FILTER", "3")),
        )
        for f in (dYc.ρ, dYc.ρe, dρuₕ, dρw)
            data = Fields.field_values(f)
            Operators.tensor_product!(data, data, M)
        end
        @. dρw = B(dρw)
    end

    return dY
end

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)

const t_end = parse(FT, get(ENV, "T_END", box == "rtb" ? "1000.0" : "700.0"))
prob = ODEProblem(rhs!, Y, (0.0, t_end))

# Optional solution-state cutoff filter: applies directly to Y after each step.
# Stronger than the tendency filter (removes in-element high modes from prognostics).
# Enable with FILTER_SOL=3 (or any Nc); disable with FILTER_SOL=0 (default).
const filter_sol_nc = parse(Int, get(ENV, "FILTER_SOL", "0"))
const M_sol = filter_sol_nc > 0 ? Quadratures.cutoff_filter_matrix(
    FT, Spaces.quadrature_style(axes(Y.Yc.ρ)), filter_sol_nc,
) : nothing
const B_sol = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(Geometry.WVector(FT(0))),
    top    = Operators.SetValue(Geometry.WVector(FT(0))),
)
function solution_filter!(u, integrator, p, t)
    M = M_sol
    for f in (u.Yc.ρ, u.Yc.ρe, u.ρuₕ)
        Operators.tensor_product!(Fields.field_values(f), Fields.field_values(f), M)
    end
    Operators.tensor_product!(Fields.field_values(u.ρw), Fields.field_values(u.ρw), M)
    @. u.ρw = B_sol(u.ρw)
end

algorithm = filter_sol_nc > 0 ?
    SSPRK33(step_limiter! = solution_filter!) :
    SSPRK33()

sol = solve(
    prob,
    algorithm,
    dt = Δt,
    saveat = collect(0.0:parse(FT, get(ENV, "SAVEAT", "50.0")):t_end),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# ---------------------------------------------------------------------------
# Density diagnostics (temporary: understand min ρ at each saved frame)
# ---------------------------------------------------------------------------
for (t_s, u) in zip(sol.t, sol.u)
    ρ_data = parent(Fields.field_values(u.Yc.ρ))
    @info "frame" t=t_s ρ_min=minimum(ρ_data) ρ_max=maximum(ρ_data) finite=all(isfinite, ρ_data)
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "rtb_dg_fd"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

u_ok = filter(
    u -> all(isfinite, parent(Fields.field_values(u.Yc.ρ))),
    sol.u,
)
if isempty(u_ok)
    @error "No finite solution frames to plot" retcode = sol.retcode
else
    function potential_temperature(ρ, ρe, u, w, z)
        K = (u^2 + w^2) / 2
        p = pressure_eos(ρ, ρe, K, z)
        T = p / (ρ * R_d)
        return T * (MSLP / p)^((γ - 1) / γ)
    end

    Ic_out = Operators.InterpolateF2C()
    anim = Plots.@animate for u in u_ok
        w = @. Ic_out(u.ρw).components.data.:1 / u.Yc.ρ
        us = @. (u.ρuₕ / u.Yc.ρ).components.data.:1
        θ = @. potential_temperature(u.Yc.ρ, u.Yc.ρe, us, w, coords.z)
        Plots.plot(θ .- θ₀; title = "θ′ (K)", colorbar = true)
    end
    Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 10)

    anim = Plots.@animate for u in u_ok
        Plots.plot(u.Yc.ρ; title = "ρ", colorbar = true)
    end
    Plots.mp4(anim, joinpath(path, "density.mp4"), fps = 10)

    Es = [sum(u.Yc.ρe) for u in u_ok]
    Mass = [sum(u.Yc.ρ) for u in u_ok]
    Plots.png(
        Plots.plot((Es .- energy_0) ./ energy_0; ylabel = "Δ∫ρe / ∫ρe₀"),
        joinpath(path, "energy_budget.png"),
    )
    Plots.png(
        Plots.plot((Mass .- mass_0) ./ mass_0; ylabel = "Δ∫ρ / ∫ρ₀"),
        joinpath(path, "mass_budget.png"),
    )
end

@info "Stage B rising thermal bubble (Souza hybrid) finished" path = path retcode =
    sol.retcode
