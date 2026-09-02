#=
Stage B: Straka density current on an extruded x–z plane with
  • discontinuous Galerkin (DG) horizontal spectral elements (no DSS)
  • finite-difference vertical staggering (Atmos-like)

State (total-energy / ρe form, matching working FDC Roe):
  Y.c = (ρ, ρe) on centers
  ρuₕ  = horizontal momentum (UVector) on centers
  ρw   = vertical momentum (WVector) on faces

Horizontal: WeakDivergence + 1D Kennedy–Gruber + Roe face flux
with thermodynamic p in energy and perturbation pressure p′ in momentum.
Stabilization: biharmonic hyperdiffusion (κ₄) with grid-dependent h³ scaling
following Lauritzen et al. (2018): κ₄ = ν₄_coeff × h³ where h is the nodal
horizontal length scale. Default ν₄_coeff=10.5 m/s → κ₄≈2e8 m⁴/s at
HELEM=64, npoly=3 (h≈267 m). NOTE: explicit SIPG biharmonic stability requires
DT to scale with h²/κ₄; default DT=0.0125s is tuned for HELEM=64 with κ₄=2e8.
Override via NU4 (coefficient, m/s) or KAPPA4 (absolute, m⁴/s).
Vertical FD: mass via face ρw; energy / momentum (uₕ and w) via Lin–van Leer
(`LinVanLeerC2F`) fluxes — upwinding provides implicit dissipation vertically.
Walls (top/bottom): reflecting slip — FD analog of FDC `ReflectingWallBC`:
  • ρw = 0 (no normal flow) via DivergenceF2C SetValue + SetBoundaryOperator
  • Neumann ∇z p′ = 0 (and ∇z of diffused scalars) via GradientC2F SetGradient(0)
  • tangential u free (slip); dρw re-zeroed by B after face DG / filter
Cutoff filter Nc=3 on center and face ρw tendencies (matches unified FDC Roe).

Run:
  julia --project=.buildkite examples/hybrid/plane/density_current_2d_dg_fd.jl

Environment:
  HELEM, VELEM, NPOLY, DT, T_END, KAPPA2 (m²/s), NU4 (m/s coeff), KAPPA4 (m⁴/s)
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
# Mesh
# ---------------------------------------------------------------------------
function hvspace_2D(
    xlim = (-25600.0, 25600.0),
    zlim = (0.0, 6400.0),
    helem = 32,
    velem = 32,
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

const helem = parse(Int, get(ENV, "HELEM", "64"))
const velem = parse(Int, get(ENV, "VELEM", "64"))
const npoly = parse(Int, get(ENV, "NPOLY", "3"))
const Δt = parse(FT, get(ENV, "DT", "0.1"))
hv_center_space, hv_face_space =
    hvspace_2D((-25600, 25600), (0, 6400), helem, velem, npoly)

# Grid-dependent biharmonic coefficient following Lauritzen et al. (2018):
#   κ₄ = ν₄_coeff × h³  where h = node_horizontal_length_scale
# For effective grid-scale damping we need κ₄ > U·h³ (so ν₄_coeff > U_max).
# Default ν₄_coeff=100 m/s → κ₄ ≈ 1.9e9 m⁴/s at HELEM=64, npoly=3 (h≈267 m).
# Override via KAPPA4 (absolute) or NU4 (coefficient in m/s).
const h_node = FT(Spaces.node_horizontal_length_scale(
    Spaces.horizontal_space(hv_center_space)
))
const ν₄_coeff = parse(FT, get(ENV, "NU4", "10.5"))    # m/s; ≈2e8 m⁴/s at HELEM=64

# ---------------------------------------------------------------------------
# Physics (ρe / total energy — FDC / Souza-style ideal gas)
# ---------------------------------------------------------------------------
const MSLP = 1e5
const grav = 9.8
const R_d = 287.058
const γ = 1.4
const C_p = R_d * γ / (γ - 1)
const θ₀ = 300.0

Φ(z) = grav * z

background_pressure(z) = MSLP * (1 - Φ(z) / (C_p * θ₀))^(C_p / R_d)
background_density(z) =
    background_pressure(z) / (R_d * θ₀ * (1 - Φ(z) / (C_p * θ₀)))

# p = (γ−1)(ρe − ρKE − ρΦ)
pressure(ρ, ρe, K, z) = (γ - 1) * (ρe - ρ * K - ρ * Φ(z))

function init_density_current_2d(x, z)
    x_c, z_c, r_c = 0.0, 3000.0, 1.0
    x_r, z_r = 4000.0, 2000.0
    θ_c = -15.0
    r = sqrt((x - x_c)^2 / x_r^2 + (z - z_c)^2 / z_r^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1 + cospi(r / r_c)) : 0.0
    θ = θ₀ + θ_p
    π_exn = 1 - Φ(z) / (C_p * θ)
    T = π_exn * θ
    p = MSLP * π_exn^(C_p / R_d)
    ρ = p / (R_d * T)
    # Resting: ρe = p/(γ−1) + ρΦ  (matches FDC Roe EOS)
    ρe = p / (γ - 1) + ρ * Φ(z)
    return (ρ = ρ, ρe = ρe)
end

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_density_current_2d(coord.x, coord.z), coords)
ρw = map(_ -> Geometry.WVector(0.0), face_coords)
Y = Fields.FieldVector(
    Yc = Yc,
    ρuₕ = Yc.ρ .* Ref(Geometry.UVector(0.0)),
    ρw = ρw,
)

const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

# ---------------------------------------------------------------------------
# Horizontal volume flux + 1D K-G / Roe face flux (UVector normals)
#
# Operators.EntropyConservingFlux is 2D (UV / .v + shear wave); the extruded
# plane face normal is Geometry.UVector, so we use an equivalent 1D form.
# ---------------------------------------------------------------------------

function thermo(ρ, ρe, ρu, w, z)
    u = ρu.u / ρ
    KE = (u^2 + w^2) / 2
    p = pressure(ρ, ρe, KE, z)
    p′ = p - background_pressure(z)
    return (; u = Geometry.UVector(u), KE, p, p′)
end

function flux_h(y)
    th = thermo(y.ρ, y.ρe, y.ρu, y.w, y.z)
    return (
        ρ = y.ρu,
        ρu = (y.ρu ⊗ th.u) + th.p′ * I,
        ρe = th.u * (y.ρe + th.p),
    )
end

roe_average(ρ⁻, ρ⁺, a⁻, a⁺) =
    (sqrt(ρ⁻) * a⁻ + sqrt(ρ⁺) * a⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

"""
1D horizontal Kennedy–Gruber interface + Roe dissipation for (ρ, ρu, ρe).
`normal` is a `Geometry.UVector`.
"""
function rhoe_roe_1d(normal, (y⁻,), (y⁺,))
    ρ⁻, ρu⁻, ρe⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρe
    ρ⁺, ρu⁺, ρe⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρe

    th⁻ = thermo(ρ⁻, ρe⁻, ρu⁻, y⁻.w, y⁻.z)
    th⁺ = thermo(ρ⁺, ρe⁺, ρu⁺, y⁺.w, y⁺.z)
    u⁻, u⁺ = th⁻.u, th⁺.u
    p⁻, p⁺ = th⁻.p, th⁺.p
    pm⁻, pm⁺ = th⁻.p′, th⁺.p′

    p_bg⁻ = background_pressure(y⁻.z)
    p_bg⁺ = background_pressure(y⁺.z)
    ρ_bg⁻ = background_density(y⁻.z)
    ρ_bg⁺ = background_density(y⁺.z)
    c_bg⁻ = sqrt(γ * p_bg⁻ / ρ⁻)
    c_bg⁺ = sqrt(γ * p_bg⁺ / ρ⁺)
    c⁻ = p⁻ > 0 ? max(c_bg⁻, sqrt(γ * p⁻ / ρ⁻)) : c_bg⁻
    c⁺ = p⁺ > 0 ? max(c_bg⁺, sqrt(γ * p⁺ / ρ⁺)) : c_bg⁺

    # Kennedy–Gruber central flux
    ρ̄ = (ρ⁻ + ρ⁺) / 2
    ū = (u⁻ + u⁺) / 2
    p̄ = (p⁻ + p⁺) / 2
    p̄m = (pm⁻ + pm⁺) / 2
    ē = (ρe⁻ / ρ⁻ + ρe⁺ / ρ⁺) / 2

    Fc_ρ = (ρ̄ * ū)' * normal
    Fc_ρu = (ρ̄ * (ū ⊗ ū) + p̄m * I)' * normal
    Fc_ρe = (ū * (ρ̄ * ē + p̄))' * normal

    # Roe averages — arithmetic fallback if a state density is already non-positive
    pos = ρ⁻ > 0 && ρ⁺ > 0
    ρ̃ = pos ? sqrt(ρ⁻ * ρ⁺) : abs(ρ̄)
    ũ = pos ? roe_average(ρ⁻, ρ⁺, u⁻, u⁺) : ū
    H⁻ = (ρe⁻ + p⁻) / ρ⁻
    H⁺ = (ρe⁺ + p⁺) / ρ⁺
    H̃ = pos ? roe_average(ρ⁻, ρ⁺, H⁻, H⁺) : (H⁻ + H⁺) / 2
    KE_tilde = (ũ.u^2) / 2
    c̃ = pos ? roe_average(ρ⁻, ρ⁺, c⁻, c⁺) : max(c_bg⁻, c_bg⁺)

    ũₙ = ũ' * normal
    Δρ = ρ⁺ - ρ⁻
    Δuₙ = (u⁺ - u⁻)' * normal
    Δp = pm⁺ - pm⁻  # p′ jump for stratified acoustics

    c̃⁻² = 1 / c̃^2
    α₁ = (Δp - ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²
    α₂ = Δρ - Δp * c̃⁻²
    α₄ = (Δp + ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²

    λ₁ = abs(ũₙ - c̃)
    λ₂ = abs(ũₙ)
    λ₄ = abs(ũₙ + c̃)

    diss_ρ = λ₁ * α₁ + λ₂ * α₂ + λ₄ * α₄
    diss_ρu =
        (λ₁ * α₁) * (ũ - c̃ * normal) +
        (λ₂ * α₂) * ũ +
        (λ₄ * α₄) * (ũ + c̃ * normal)
    diss_ρe =
        λ₁ * α₁ * (H̃ - c̃ * ũₙ) +
        λ₂ * α₂ * KE_tilde +
        λ₄ * α₄ * (H̃ + c̃ * ũₙ)

    return (
        ρ = Fc_ρ - diss_ρ / 2,
        ρu = Fc_ρu - diss_ρu / 2,
        ρe = Fc_ρe - diss_ρe / 2,
    )
end

# Face ρw: upwind with arithmetic-mean normal velocity
function ρw_roe(normal, (ρw⁻, uₕ⁻), (ρw⁺, uₕ⁺))
    un̄ = ((uₕ⁻ + uₕ⁺) / 2)' * normal
    ρw_up = un̄ > 0 ? ρw⁻ : ρw⁺
    return un̄ * ρw_up
end

const κ₄_cfl_cap = FT(h_node^3 / ((2 * npoly + 1)^2 * Δt))  # SIPG explicit stability limit
const κ₄ = haskey(ENV, "KAPPA4") ? parse(FT, ENV["KAPPA4"]) :
    FT(min(ν₄_coeff * h_node^3, κ₄_cfl_cap))         # m⁴/s, h³ scaling, capped
const κ₂ = parse(FT, get(ENV, "KAPPA2", "75.0"))    # m²/s; Straka (1993) reference value

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
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    # Lin–van Leer vertical reconstruction (Lin 1994); Atmos / staggered form:
    #   ∂t(ρχ) = −∇·(ρ * VL(w, χ)) with χ ∈ {h_tot, u, w}
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
    # Reflecting slip walls (FD analog of Operators.ReflectingWallBC):
    #   • ρw ≡ 0 on wall faces — no normal flow (ghost would flip w)
    #   • ∇z φ|_wall = 0 for p′, u, h — Neumann acoustics / adiabatic wall
    #   • tangential u unrestricted (slip, not no-slip)
    ∂f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
        top = Operators.SetGradient(Geometry.WVector(FT(0))),
    )
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    uₕ = @. ρuₕ / ρ
    # AxisVector Fields expose components via `.components.data`, not `.u`/`.w`.
    w_c = @. Ic(ρw).components.data.:1 / ρ
    u_c = @. uₕ.components.data.:1

    # --- Horizontal DG for (ρ, ρe, ρu) ---
    y = map(
        (ρi, ρei, ρui, zi, wi) ->
            (; ρ = ρi, ρe = ρei, ρu = ρui, z = zi, w = wi),
        ρ,
        ρe,
        ρuₕ,
        z,
        w_c,
    )
    Fh = map(flux_h, y)
    dy_mw = @. hwdiv(Fh) * (-(lgeom_c.WJ))
    Operators.add_numerical_flux_interior!(rhoe_roe_1d, dy_mw, y)
    @. dy_mw = dy_mw / lgeom_c.WJ

    @. dYc.ρ = dy_mw.ρ
    @. dYc.ρe = dy_mw.ρe
    @. dρuₕ = dy_mw.ρu

    # --- Vertical FD (mass: face ρw; energy / uₕ: Lin–van Leer) ---
    K = @. (u_c^2 + w_c^2) / 2
    p = @. pressure(ρ, ρe, K, z)
    p′ = @. p - background_pressure(z)
    ρ_bg = @. background_density(z)
    Yfρ = @. If(ρ)
    w = @. ρw / Yfρ                     # face velocity (WVector)
    h_tot = @. (ρe + p) / ρ

    # Mass: prognostic face flux ρw (centered DivergenceF2C; matches Atmos)
    @. dYc.ρ -= vdivf2c(ρw)
    # Energy: −∇·(ρw * VL_upwind(h_tot))
    @. dYc.ρe -= vdivf2c(VanLeer(ρw, h_tot, Δt))
    # Horizontal momentum: −∇·(ρw * VL_upwind(u))
    @. dρuₕ -= Geometry.UVector(vdivf2c(VanLeer(ρw, u_c, Δt)))

    # Vertical momentum: −∂z p′ + buoy − ∂z(ρw ⊗ w)  [centered]
    buoy = @. -(ρ - ρ_bg) * grav
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )
    @. dρw = B(
        Geometry.transform(Geometry.WAxis(), -(∂f(p′))) +
        If(Geometry.WVector(buoy)) -
        vvdivc2f(Ic(ρw ⊗ w)),
    )

    # Horizontal DG for ρw
    uₕf = @. If(uₕ)
    # dρw_mw/WJ is already the full weak tendency −∇·(uₕ ⊗ ρw) + face
    # dissipation (same convention as the center equations), so it is ADDED.
    dρw_mw = @. hwdiv(uₕf ⊗ ρw) * (-(lgeom_f.WJ))
    Operators.add_numerical_flux_interior!(ρw_roe, dρw_mw, ρw, uₕf)
    @. dρw += dρw_mw / lgeom_f.WJ

    # Horizontal biharmonic hyperdiffusion: ∂t q += −κ₄ ρ ∇⁴_h q
    # Two-pass form: element-local first Laplacian χ = ∇²q, then
    # SIPG-coupled second pass −κ₄ ∇·(ρ ∇χ) for inter-element damping.
    # NOTE: no κ₄/κ₂ diffusion is applied to ρ. Mass diffusion is unphysical
    # (Straka κ applies to velocity/temperature) and, against the stratified
    # background ∂z ρ_bg ≠ 0 with zero-flux walls, it pumps a spurious
    # warm-bottom / cold-top boundary-layer dipole (~3 K over 900 s at κ₂=75).
    # ρ is stabilized by the Roe contact-wave dissipation + cutoff filter only.
    if κ₄ != 0
        hgrad  = Operators.Gradient()
        τ_κ₄   = Operators.ldg_penalty_parameter(κ₄, axes(ρ))
        χ_htot = similar(h_tot); @. χ_htot = hwdiv(hgrad(h_tot))
        χ_u    = similar(u_c);   @. χ_u    = hwdiv(hgrad(u_c))
        _de4 = ldg_laplacian_tendency(χ_htot, ρ,       κ₄, τ_κ₄)
        _du4 = ldg_laplacian_tendency(χ_u,    ρ,       κ₄, τ_κ₄)
        @. dYc.ρe -= _de4
        @. dρuₕ   -= Geometry.UVector(_du4)
    end

    # Second-order diffusion (κ₂): horizontal LDG + vertical FD Laplacian.
    # Horizontal LDG is needed to control the KH rotor at the density current
    # front (Straka et al. 1993 uses κ₂ = 75 m²/s in all directions).
    # Default κ₂ = 0 (entropy-stable Roe + VanLeer without explicit viscosity).
    if κ₂ != 0
        # Horizontal κ₂ via LDG (velocity + enthalpy only; no mass diffusion)
        τ_κ₂  = Operators.ldg_penalty_parameter(κ₂, axes(ρ))
        _du_h = ldg_laplacian_tendency(u_c,   ρ,       κ₂, τ_κ₂)
        _de_h = ldg_laplacian_tendency(h_tot, ρ,       κ₂, τ_κ₂)
        @. dρuₕ   += Geometry.UVector(_du_h)
        @. dYc.ρe += _de_h

        # Vertical κ₂ via FD Laplacian (Neumann walls: ∂z φ|wall = 0)
        uvdivf2c_diff = Operators.DivergenceF2C(
            bottom = Operators.SetValue(Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0))),
            top    = Operators.SetValue(Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0))),
        )
        vvdivc2f_diff = Operators.DivergenceC2F(
            bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
            top    = Operators.SetDivergence(Geometry.WVector(FT(0))),
        )
        ∂c = Operators.GradientF2C()
        @. dρuₕ   += Geometry.UVector(uvdivf2c_diff(κ₂ * (Yfρ * ∂f(uₕ))))
        @. dρw    += Geometry.WVector(vvdivc2f_diff(κ₂ * ρ * ∂c(ρw / Yfρ)))
        @. dYc.ρe += vdivf2c(κ₂ * (Yfρ * ∂f(h_tot)))
    end

    # Element-local cutoff filter on full residual (Nc=3), matching unified FDC
    M = Quadratures.cutoff_filter_matrix(
        FT,
        Spaces.quadrature_style(axes(ρ)),
        3,
    )
    for f in (dYc.ρ, dYc.ρe, dρuₕ, dρw)
        data = Fields.field_values(f)
        Operators.tensor_product!(data, data, M)
    end
    # Re-enforce reflecting walls (ρw ≡ 0) after face DG / LDG / filter
    @. dρw = B(dρw)

    return dY
end

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)

const t_end = parse(FT, get(ENV, "T_END", "900.0"))
prob = ODEProblem(rhs!, Y, (0.0, t_end))

# Optional solution-state cutoff filter: applied to Y directly after each step.
# Enable with FILTER_SOL=3; default off (tendency filter inside rhs! is primary).
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
    saveat = collect(0.0:50.0:t_end),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# ---------------------------------------------------------------------------
# Output (skip movies if the integrator aborted with non-finite fields)
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dc_dg_fd"
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
        p = pressure(ρ, ρe, K, z)
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

@info "Stage B density current (ρe) finished" path = path retcode = sol.retcode
