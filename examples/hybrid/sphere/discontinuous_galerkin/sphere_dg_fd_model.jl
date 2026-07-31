#=
Shared setup for spherical (cubed-sphere shell) compressible-Euler DG-FD test
cases:
  • discontinuous Galerkin (DG) horizontal spectral elements (no DSS)
  • finite-difference vertical staggering (Atmos-like)

This file provides only the SHARED scaffolding consumed by the flux-form FDDG
driver (`baroclinic_wave_fddg_fluxform.jl`): the cubed-sphere shell spaces,
physical constants, the balanced/perturbed initial state, the vertical FD
operators, and the cutoff-filter / Rayleigh-sponge / κ₄-hyperdiffusion
parameters. The prognostic tendency, time stepping and HEVI Jacobian live in
the including driver. (The earlier vector-invariant tendency/driver and its
`sphere_dg_fd_jacobian.jl` have been removed; only the flux-form FDDG pathway
remains.)

Initial state (returned by `initial_state`, in geographic/covariant form for
the driver to convert): Yc = (ρ, ρe) centers; uₕ Covariant12Vector centers;
w Covariant3Vector faces. ρe is total energy density (internal + kinetic +
geopotential). Face quantities are taken in the local orthonormal geographic
frame, which is single-valued at shared nodes — including across cubed-sphere
panel edges, where covariant components are not.

Vertical FD: mass via face mass flux; energy via Lin–van Leer upwind;
w = 0 and ∂z(·) = 0 at top/bottom (CG-model boundary conditions).
Stabilization: κ₄ biharmonic hyperdiffusion ONLY (no κ₂), two-pass:
element-local first Laplacian, then SIPG (LDG penalty) second pass for
inter-element damping. Optional element-local cutoff filter on the state.

The including driver must define (before `include`):
  const FT                  # floating-point type
  const apply_held_suarez   # Bool: add Held–Suarez forcing
  const is_balanced_flow    # Bool: disable the baroclinic-wave perturbation
  const t_end_default       # default simulation length [s]

Environment overrides: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, KAPPA4, FILTER,
STEPPER
=#

using LinearAlgebra: ×, norm, norm_sqr, dot

import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies

using OrdinaryDiffEqSSPRK: ODEProblem, solve, SSPRK33
import SciMLBase
import ClimaTimeSteppers as CTS
import StaticArrays: SVector
import Printf

# DiffEqBase's default internal norm reduces a FieldVector state by iterating
# it element-by-element — disallowed scalar indexing on GPU backing arrays.
# Pass this to `solve` via `internalnorm`: it reduces over each component's
# contiguous backing array instead (with fixed-dt SSP-RK3 it is only
# evaluated once, at solver init).
fieldvector_norm(u::Fields.FieldVector, t) = sqrt(
    sum(x -> sum(abs2, Fields.backing_array(x)), Tuple(Fields._values(u))) /
    length(u),
)
fieldvector_norm(u, t) = abs(u)

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const C3 = Geometry.Covariant3Vector
const C12 = Geometry.Covariant12Vector
const C123 = Geometry.Covariant123Vector
const CT3 = Geometry.Contravariant3Vector
const CT12 = Geometry.Contravariant12Vector

# ---------------------------------------------------------------------------
# Physical constants (match the CG sphere examples)
# ---------------------------------------------------------------------------
const p_0 = FT(1.0e5)
const R_d = FT(287.0)
const κ_gas = FT(2 / 7)
const T_tri = FT(273.16)
const grav = FT(9.80616)
const Ω = FT(7.29212e-5)
const cp_d = R_d / κ_gas
const cv_d = cp_d - R_d
const γ = cp_d / cv_d

pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)

# ---------------------------------------------------------------------------
# Grid (equiangular cubed sphere × uniform vertical levels)
# ---------------------------------------------------------------------------
const R = FT(6.371229e6)
const helem = parse(Int, get(ENV, "HELEM", "4"))
const npoly = parse(Int, get(ENV, "NPOLY", "4"))
const zelem = parse(Int, get(ENV, "ZELEM", "10"))
const zmax = parse(FT, get(ENV, "ZMAX", "30e3"))
const t_end = parse(FT, get(ENV, "T_END", string(t_end_default)))
# STEPPER selects the time integrator: "explicit" (fully explicit SSP-RK3) or
# "hevi" (IMEX ARK: horizontal DG terms explicit, vertical acoustics implicit
# with a column-wise Newton solve).
const stepper = lowercase(get(ENV, "STEPPER", "explicit"))
stepper in ("explicit", "hevi") || error("STEPPER must be explicit or hevi")

function sphere_hv_spaces()
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(zmax);
        boundary_names = (:bottom, :top),
    )
    # ZSTRETCH = <dz_bottom>,<dz_top> [m] selects a generalized-exponential
    # stretched vertical grid (canonical practice, e.g. JW06's 26 stretched
    # levels): fine cells resolve the troposphere while the coarse upper
    # cells cannot support the ρ^{-1/2}-amplifying wave field — the
    # dissipation-free substitute for what uniform fine levels + a rigid
    # lid otherwise demand of a sponge or ∇⁴.
    vertmesh = if haskey(ENV, "ZSTRETCH")
        dzb, dzt = parse.(FT, split(ENV["ZSTRETCH"], ","))
        Meshes.IntervalMesh(
            vertdomain,
            Meshes.GeneralizedExponentialStretching(dzb, dzt);
            nelems = zelem,
        )
    else
        Meshes.IntervalMesh(vertdomain, nelems = zelem)
    end
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_center_space, hv_face_space)
end

horzspace, hv_center_space, hv_face_space = sphere_hv_spaces()
ccoords = Fields.coordinate_field(hv_center_space)
fcoords = Fields.coordinate_field(hv_face_space)

# Explicit stepping is limited by the vertical acoustic CFL: SSP-RK3 needs
# c·π·Δt/Δz ≲ √3 (Δz = 3 km, c ≈ 350 m/s ⇒ Δt ≲ 4.7 s); the horizontal DG
# limit is far looser at these resolutions. HEVI removes the vertical limit,
# so its default Δt is set by the horizontal DG acoustic CFL instead,
# h_node / (c (2p + 1)).
const Δt = if haskey(ENV, "DT")
    parse(FT, ENV["DT"])
elseif stepper == "hevi"
    FT(max(
        1,
        floor(
            Spaces.node_horizontal_length_scale(horzspace) /
            (350 * (2 * npoly + 1)),
        ),
    ))
else
    FT(4.0)
end

const ᶜΦ = @. grav * ccoords.z
const ᶜf_cor = @. CT3(Geometry.WVector(2 * Ω * sind(ccoords.lat)))

# Cartesian basis fields (centers): ê_E, ê_N, r̂ from lat/long (degrees).
# Used by the flux-form FDDG driver — velocity components advected as scalars
# must live in a globally constant frame.
const eE1 = @. -sind(ccoords.long)
const eE2 = @. cosd(ccoords.long)
const eE3 = map(_ -> FT(0), eE1)
const eN1 = @. -sind(ccoords.lat) * cosd(ccoords.long)
const eN2 = @. -sind(ccoords.lat) * sind(ccoords.long)
const eN3 = @. cosd(ccoords.lat)
const eR1 = @. cosd(ccoords.lat) * cosd(ccoords.long)
const eR2 = @. cosd(ccoords.lat) * sind(ccoords.long)
const eR3 = @. sind(ccoords.lat)

# Rayleigh sponge over the top z_sponge of the domain: absorbing layer for
# the rigid lid (reflected/amplifying waves otherwise accumulate in near-lid
# modes; at fine vertical resolution the wave field genuinely propagates
# there and breaks). β profile as in the CG examples
# (baroclinic_wave_utilities.jl) with a Δt-INDEPENDENT peak rate 1/τ
# (SPONGE_TAU [s], default 1200 — matches the validated dt = 60 runs, which
# used the CG 1/(20Δt) convention). Applied to ρw/w always; SPONGE_UH=1
# additionally damps horizontal momentum (the CG reference damps both — at
# 22.5–30 km the JW jet is weak, and resolved upper-level wave breaking
# otherwise deposits momentum with nothing to remove it).
const z_sponge = FT(7.5e3)
const sponge_τ = parse(FT, get(ENV, "SPONGE_TAU", "1200"))
const sponge_uh = get(ENV, "SPONGE_UH", "0") == "1"
const ᶠβ_sponge = @. ifelse(
    fcoords.z > zmax - z_sponge,
    1 / sponge_τ * sin(FT(π) / 2 * (fcoords.z - (zmax - z_sponge)) / z_sponge)^2,
    FT(0),
)
const ᶜβ_sponge = @. ifelse(
    ccoords.z > zmax - z_sponge,
    1 / sponge_τ * sin(FT(π) / 2 * (ccoords.z - (zmax - z_sponge)) / z_sponge)^2,
    FT(0),
)

# ---------------------------------------------------------------------------
# Initial conditions (Ullrich et al. moist-free baroclinic wave base state;
# copied from baroclinic_wave_utilities.jl, which cannot be included here
# without pulling in the CG implicit model)
# ---------------------------------------------------------------------------
const kb = 3
const T_e = FT(310)
const T_p = FT(240)
const T_0 = FT(0.5) * (T_e + T_p)
const Γ = FT(0.005)
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = FT(0.5) * (kb + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = FT(15e3)
const λ_c = FT(20)
const ϕ_c = FT(40)
const d_0 = R / 6
const V_p = FT(1)

τ_z_1(z) = exp(Γ * z / T_0)
τ_z_2(z) = 1 - 2 * (z / b / H)^2
τ_z_3(z) = exp(-(z / b / H)^2)
τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
τ_int_2(z) = C * z * τ_z_3(z)
F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
I_T(ϕ) = cosd(ϕ)^kb - kb * (cosd(ϕ))^(kb + 2) / (kb + 2)
temp(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
pres(ϕ, z) = p_0 * exp(-grav / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
r_gc(λ, ϕ) =
    R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
U(ϕ, z) =
    grav * kb / R *
    τ_int_2(z) *
    temp(ϕ, z) *
    (cosd(ϕ)^(kb - 1) - cosd(ϕ)^(kb + 1))
u_base(ϕ, z) =
    -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
c3(λ, ϕ) = cos(FT(π) * r_gc(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(FT(π) * r_gc(λ, ϕ) / 2 / d_0)
cond(λ, ϕ) = (0 < r_gc(λ, ϕ) < d_0) * (r_gc(λ, ϕ) != R * pi)
δu(λ, ϕ, z) =
    -16 * V_p / 3 / sqrt(FT(3)) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
    sin(r_gc(λ, ϕ) / R) * cond(λ, ϕ)
δv(λ, ϕ, z) =
    16 * V_p / 3 / sqrt(FT(3)) *
    F_z(z) *
    c3(λ, ϕ) *
    s1(λ, ϕ) *
    cosd(ϕ_c) *
    sind(λ - λ_c) / sin(r_gc(λ, ϕ) / R) * cond(λ, ϕ)

function initial_state(ᶜlocal_geometry, ᶠlocal_geometry)
    (; lat, long, z) = ᶜlocal_geometry.coordinates
    ᶜρ = @. pres(lat, z) / R_d / temp(lat, z)
    u₀ = @. u_base(lat, z)
    v₀ = @. 0 * z
    if !is_balanced_flow
        @. u₀ += δu(long, lat, z)
        @. v₀ += δv(long, lat, z)
    end
    ᶜuₕ_local = @. Geometry.UVVector(u₀, v₀)
    ᶜuₕ = @. C12(ᶜuₕ_local, ᶜlocal_geometry)

    # Discrete hydrostatic balance (column-wise, cf. solid_body_rotation_3d):
    # the analytic state satisfies ∂z p = −ρg only in the continuum; on the
    # staggered FD grid the residual of ᶠgradᵥ(p)/ᶠinterp(ρ) + g projects
    # onto gravity modes and drives O(10 m/s) spurious w. Keep the analytic
    # p at cell centers and correct ρ so the centered face balance
    # (p[v+1] − p[v])/Δz = −g (ρ[v] + ρ[v+1])/2 holds exactly, then set ρe
    # such that the diagnosed pressure is exactly the analytic p.
    ᶜp_ana = @. pres(lat, z)
    ρ_par = parent(ᶜρ)
    p_par = parent(ᶜp_ana)
    # per-interface Δz from the actual center heights (supports ZSTRETCH;
    # a uniform zmax/zelem here silently corrupts ρ on stretched grids)
    z_par = parent(z)
    for v in 1:(size(ρ_par, 1) - 1)
        @views @. ρ_par[v + 1, :, :, :, :] =
            -ρ_par[v, :, :, :, :] -
            2 * (p_par[v + 1, :, :, :, :] - p_par[v, :, :, :, :]) /
            (z_par[v + 1, :, :, :, :] - z_par[v, :, :, :, :]) / grav
    end
    ᶜK = @. norm_sqr(ᶜuₕ_local) / 2
    ᶜρe = @. cv_d * ᶜp_ana / R_d + ᶜρ * (ᶜK + grav * z - cv_d * T_tri)

    ᶠw = map(_ -> C3(FT(0)), ᶠlocal_geometry)
    Yc = map((ρi, ρei) -> (; ρ = ρi, ρe = ρei), ᶜρ, ᶜρe)
    return Fields.FieldVector(Yc = Yc, uₕ = ᶜuₕ, w = ᶠw)
end

# ---------------------------------------------------------------------------
# Held–Suarez forcing constants (Held & Suarez 1994)
# ---------------------------------------------------------------------------
const day = FT(3600 * 24)
const k_a = 1 / (40 * day)
const k_f = 1 / day
const k_s = 1 / (4 * day)
const ΔT_y = FT(60)
const Δθ_z = FT(10)
const T_equator = FT(315)
const T_min = FT(200)
const σ_b = FT(7 / 10)

# All DG building blocks — the Kennedy-Gruber two-point/interface fluxes,
# the central lifting / jump-penalty face functions, `lifting_correction`,
# and `ldg_laplacian_tendency` — come from ClimaCore's Operators module;
# no operators are defined in this driver.

# Explicit SIPG biharmonic stability cap (validated on the plane DG-FD
# cases): the CG value 2e17 is only stable there because DSS makes the
# first-pass Laplacian continuous; the DG penalty at 2e17 exceeds this cap
# ~400× at the default resolution and blows up within a few steps.
const κ₄_cfl_cap = FT(
    Spaces.node_horizontal_length_scale(horzspace)^3 /
    ((2 * npoly + 1)^2 * Δt),
)
# Default κ₄ = cap/10: the SIPG penalty acts on the O(truncation) face jumps
# of the element-local first-pass Laplacian, so cap-level κ₄ produces a
# measurable spurious forcing of smooth balanced states (~4 m/s of inertial
# v-oscillation per hour at the cap at the default resolution); cap/10 keeps
# that near the truncation floor while still damping grid modes.
const κ₄ = haskey(ENV, "KAPPA4") ? parse(FT, ENV["KAPPA4"]) :
    min(FT(2e17), κ₄_cfl_cap / 10)
κ₄ > κ₄_cfl_cap &&
    @warn "κ₄ exceeds the explicit SIPG stability cap" κ₄ κ₄_cfl_cap
const filter_Nc = parse(Int, get(ENV, "FILTER", string(npoly)))

# Optional per-step exponential filter on the VELOCITY state (uₕ, w).
# The tendency cutoff above starves the top modes of forcing, but the HEVI
# implicit update bypasses it and nonlinear products regenerate top-mode
# content in the state, so noise still accumulates (helem=4 FILTER=3 GPU run
# crashed at t = 560,400 s). Filtering the state is the classical SEM cure:
# a modal projection, unconditionally stable (not Δt-limited like κ₄/SIPG),
# and conservation-neutral since ρ and ρe are untouched. Mode multipliers
# σ(m) = 1 for m ≤ kc, exp(−α((m−kc)/(npoly−kc))^(2s)) above; off when α = 0.
const state_filter_α = parse(FT, get(ENV, "STATE_FILTER_ALPHA", "0"))
const state_filter_kc = parse(Int, get(ENV, "STATE_FILTER_KC", "2"))
const state_filter_s = parse(Int, get(ENV, "STATE_FILTER_S", "2"))
const state_filter_M = let
    Nq = npoly + 1
    Σ = SVector{Nq, FT}(ntuple(Nq) do i
        m = i - 1
        m <= state_filter_kc ? FT(1) :
        exp(
            -state_filter_α *
            ((m - state_filter_kc) / (npoly - state_filter_kc))^(2 * state_filter_s),
        )
    end)
    Quadratures.spectral_filter_matrix(
        Spaces.quadrature_style(hv_center_space),
        Σ,
    )
end

function state_filter!(Y)
    state_filter_α == 0 && return Y
    # level-wise horizontal projection: does not mix levels, so the zero
    # boundary faces of w stay exactly zero
    for f in (Y.uₕ, Y.w)
        data = Fields.field_values(f)
        Operators.tensor_product!(data, data, state_filter_M)
    end
    return Y
end

# ---------------------------------------------------------------------------
# Operators (shared with the including driver)
# ---------------------------------------------------------------------------
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()

const Ic = Operators.InterpolateF2C()
const If = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const vdivf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(Geometry.WVector(FT(0))),
    top = Operators.SetValue(Geometry.WVector(FT(0))),
)
const VanLeer = Operators.LinVanLeerC2F(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
    constraint = Operators.MonotoneLocalExtrema(),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(FT(0))),
    top = Operators.SetGradient(C3(FT(0))),
)
const Bw = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)

# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------
let
    h_node = Spaces.node_horizontal_length_scale(horzspace)
    # true minimum level spacing (≠ zmax/zelem on ZSTRETCH grids)
    zf = vec(Array(parent(fcoords.z))[:, 1, 1, 1, 1])
    Δz_min = minimum(diff(zf))
    c_max = sqrt(γ * R_d * T_e)
    @info "DG-FD sphere setup" stepper helem npoly zelem Δt t_end κ₄ κ₄_cfl_cap filter_Nc h_node Δz_min
    @info "Acoustic CFL estimates" vertical = c_max * Δt / Δz_min horizontal =
        c_max * Δt / h_node
end
