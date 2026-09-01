#=
Shared setup for spherical (cubed-sphere shell) compressible-Euler DG-FD test
cases:
  • discontinuous Galerkin (DG) horizontal spectral elements (no DSS)
  • finite-difference vertical staggering (Atmos-like)

This file provides only the SHARED scaffolding consumed by the flux-form FDDG
driver (`baroclinic_wave_fddg_fluxform.jl`): the cubed-sphere shell spaces,
physical constants, the balanced/perturbed initial state, the vertical FD
operators, and the cutoff-filter / Rayleigh-sponge parameters (κ₄
hyperdiffusion has been removed). The prognostic tendency, time stepping and HEVI Jacobian live in
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
Stabilization: interface (Roe/Rusanov) numerical-flux dissipation, plus an
optional element-local cutoff filter and velocity-state spectral filter. (κ₄
biharmonic hyperdiffusion has been removed from this setup.)

The including driver must define (before `include`):
  const FT                  # floating-point type
  const apply_held_suarez   # Bool: add Held–Suarez forcing
  const is_balanced_flow    # Bool: disable the baroclinic-wave perturbation
  const t_end_default       # default simulation length [s]

Environment overrides: HELEM, NPOLY, ZELEM, ZMAX, DT, T_END, FILTER,
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
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies

import ClimaTimeSteppers as CTS
import StaticArrays: SVector
import Printf

# Moist thermodynamics (saturation adjustment) + 0-moment microphysics.
import ClimaParams as CP
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Microphysics0M as M0M

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
const is_moist = true    # moist core (q_tot + 0-moment microphysics)

pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)

# ---------------------------------------------------------------------------
# Moist thermodynamics + 0-moment microphysics parameters
# ---------------------------------------------------------------------------
# Thermodynamics.jl saturation-adjustment param set and CloudMicrophysics.jl
# 0-moment (τ_precip, qc_0) params, from ClimaParams defaults. The TD constants
# match the model's dry constants (R_d = 287, cv_d = 717.5).
const thermo_params = TP.ThermodynamicsParameters(FT)
const cm_params = CMP.Microphysics0MParams(FT)
# Fixed saturation-adjustment iterations (GPU branch-safe; ample near
# equilibrium). SAT_MAXITER overrides for experimentation.
const sat_maxiter = parse(Int, get(ENV, "SAT_MAXITER", "3"))
# Moist latent-heat / phase constants (for the precipitation energy sink).
const cv_l = TP.cv_l(thermo_params)
const cv_i = TP.cv_i(thermo_params)
const e_int_i0 = TP.e_int_i0(thermo_params)
const T_0_td = TP.T_0(thermo_params)
# Vapor gas constant / heat capacity / reference internal energy — for the moist
# HEVI Jacobian ∂p derivatives (κ_m-based; see fddg_fluxform_jacobian.jl).
const R_v = TP.R_v(thermo_params)
const cv_v = TP.cv_v(thermo_params)
const e_int_v0 = TP.e_int_v0(thermo_params)

"""
    moist_state(ρ, e_int, q_tot) -> (; T, p, q_liq, q_ice)

Saturation-adjusted moist thermodynamic diagnostics from density `ρ`, internal
energy per mass `e_int = ρe/ρ − K − Φ`, and total specific humidity `q_tot`:
the equilibrium temperature `T`, moist pressure `p = ρ R_m T`, and the condensate
partition `(q_liq, q_ice)`. Broadcastable over ClimaCore Fields (returns an
isbits NamedTuple). This replaces the dry `pressure_ρe` in the moist driver.
"""
@inline function moist_state(ρ, e_int, q_tot)
    sol = TD.saturation_adjustment(
        thermo_params,
        TD.ρe(),
        ρ,
        e_int,
        q_tot;
        maxiter = sat_maxiter,
    )
    p = TD.air_pressure(thermo_params, sol.T, ρ, q_tot, sol.q_liq, sol.q_ice)
    return (; T = sol.T, p = p, q_liq = sol.q_liq, q_ice = sol.q_ice)
end

# Robustness temperature floor for the DYNAMICS pressure (ClimaAtmos T_min_sgs
# mechanism). The implicit IMEX solver probes transient intermediate iterates
# where saturation_adjustment (exp/log/Newton) throws on T<0; ClimaAtmos avoids
# this by NEVER calling saturation adjustment inside the Newton loop — it uses a
# frozen/floored thermo state per stage. We mirror that with a NON-THROWING
# closed-form dynamics pressure used in BOTH the implicit and explicit tendencies.
const T_min_rob = parse(FT, get(ENV, "T_MIN", "150"))

"""
    moist_p_dyn(ρ, e_int, q_tot) -> (; T, p)

Robust (NON-throwing) moist pressure for the dynamics (h_tot, fluxes, PGF), used
identically in the implicit and explicit tendencies so the HEVI split stays exact.
Unlike `moist_state`, it uses the CLOSED-FORM `TD.air_temperature` (linear in
`e_int`, no Newton iteration) with a `T_min_rob` floor, so it cannot DomainError
on the implicit solver's transient iterates. Condensate is treated as vapor for
the pressure — retaining the dominant virtual-temperature effect (q_tot in R_m)
while dropping only the small condensate loading/latent term (the accurate
saturation-adjusted pressure and condensate are still used by the microphysics /
diagnostics via `moist_state`).
"""
@inline function moist_p_dyn(ρ, e_int, q_tot)
    # T from TD's CLOSED-FORM internal-energy inverse (air_temperature; the ρe
    # IndepVars branch, air_temperatures.jl:18) — non-iterating and non-throwing,
    # and the exact inverse of the TD.internal_energy used to build ρe (so it is
    # convention-consistent, unlike a hand-rolled cv_d(T−T_tri) form). Condensate
    # treated as vapor for the dynamics pressure (retains the vapor virtual effect;
    # the accurate saturated p / condensate come from moist_state for microphysics).
    # T floored to T_min_rob (ClimaAtmos T_min_sgs mechanism) ONLY to keep the
    # downstream √(γp/ρ) safe on the implicit solver's transient iterates — this
    # guards a diagnostic on solver iterates, it does not clamp the prognostic state.
    z = zero(q_tot)
    T = max(T_min_rob, TD.air_temperature(thermo_params, e_int, q_tot, z, z))
    p = TD.air_pressure(thermo_params, T, ρ, q_tot, z, z)
    return (; T = T, p = p)
end

# Condensate partition (q_liq, q_ice) at a given (T, ρ, q_tot) as a broadcastable
# NamedTuple. Closed-form (saturation vapor pressure + partition), non-throwing
# for T > 0 — so with the floored `moist_p_dyn` temperature it is safe on the
# implicit iterates. Used for the 0-moment microphysics condensate.
@inline function condensate_partition_tuple(T, ρ, q_tot)
    (q_liq, q_ice) = TD.condensate_partition(thermo_params, T, ρ, q_tot)
    return (; q_liq = q_liq, q_ice = q_ice)
end

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

# ---------------------------------------------------------------------------
# Topography (TOPO = flat | hj). Hughes & Jablonowski (2023, GMD 16, 6805)
# double-mountain: two ridges, super-Gaussian (exponent 6) in latitude ×
# Gaussian in longitude, centred at (72°E, 45°N) and (140°E, 45°N), peak
# h₀ = 2000 m. Warp applied via Gal–Chen `Hypsography.LinearAdaption`.
# ---------------------------------------------------------------------------
const topo = lowercase(get(ENV, "TOPO", "flat"))
topo in ("flat", "hj", "earth") || error("TOPO must be flat, hj, or earth")
# TOPO=earth pulls in ClimaUtilities/Interpolations/NCDatasets + the ETOPO2022
# artifact; isolated in earth_topography.jl and loaded only when requested so
# flat/hj (and GPU production) runs stay lean.
topo == "earth" && include(joinpath(@__DIR__, "earth_topography.jl"))
const hj_h0 = parse(FT, get(ENV, "MTN_HEIGHT", "2000"))  # peak elevation [m]
# WARP = linear (Gal–Chen, default) | sleve (Schär 2002 exponential decay).
# SLEVE decays the warp below η = SLEVE_ETAH with decay scale SLEVE_S (must
# satisfy s·z_top > max z_surface, enforced by Hypsography). NOTE: SLEVE
# concentrates ∂η(slope) in thinner near-surface cells — check the printed
# min Δz against the vertical acoustic CFL before keeping DT.
const warp_type = lowercase(get(ENV, "WARP", "linear"))
warp_type in ("linear", "sleve") || error("WARP must be linear or sleve")
const sleve_ηₕ = parse(FT, get(ENV, "SLEVE_ETAH", "0.7"))
const sleve_s = parse(FT, get(ENV, "SLEVE_S", "0.8"))
const hj_dφ = FT(16)   # meridional width [deg] (super-Gaussian, exponent 6)
const hj_dλ = FT(7)    # zonal width [deg] (Gaussian, exponent 2)

# shortest signed longitude difference [deg] (periodic wrap to [−180, 180])
_dlon(λ, λc) = mod(λ - λc + FT(180), FT(360)) - FT(180)
_hj_ridge(λ, φ, λc, φc) =
    exp(-((φ - φc) / hj_dφ)^6 - (_dlon(λ, λc) / hj_dλ)^2)
# z_s(λ, φ): analytic, continuous (single-valued at shared nodes) — GPU-safe.
function warp_hj(coord)
    λ = coord.long
    φ = coord.lat
    return hj_h0 *
           (_hj_ridge(λ, φ, FT(72), FT(45)) + _hj_ridge(λ, φ, FT(140), FT(45)))
end

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
    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    if topo == "flat"
        vert_center_space =
            Spaces.CenterFiniteDifferenceSpace(device, vertmesh)
        hv_center_space =
            Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
        hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    else
        # Terrain-following (hj = analytic Hughes–Jablonowski ridges; earth =
        # ETOPO2022 via earth_z_surface). Build from faces so the surface
        # coincides with the lowest face, then warp with the Gal–Chen
        # LinearAdaption of z_s(λ,φ).
        vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)
        z_surface =
            topo == "earth" ?
            Geometry.ZPoint.(earth_z_surface(horzspace)) :
            Geometry.ZPoint.(warp_hj.(Fields.coordinate_field(horzspace)))
        adaption =
            warp_type == "sleve" ?
            Hypsography.SLEVEAdaption(z_surface, sleve_ηₕ, sleve_s) :
            Hypsography.LinearAdaption(z_surface)
        hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
            adaption,
        )
        hv_center_space =
            Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
        # Grid health over terrain: the warp thins the near-surface cells
        # (SLEVE especially), and min Δz sets the explicit vertical acoustic
        # CFL — surface Δz below the flat-grid value means DT must shrink
        # proportionally, independent of any discretization property.
        Δz_f = Fields.Δz_field(hv_center_space)
        @info "warped grid health" warp_type min_Δz = minimum(Δz_f) max_Δz =
            maximum(Δz_f)
    end
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

# ---------------------------------------------------------------------------
# Hydrostatic reference state for the Exner-perturbation pressure-gradient
# force (Yatunin et al. 2026). An isothermal reference T_ref is analytic and
# satisfies dp_ref/dz = −ρ_ref g in the continuum:
#     Π_ref(z) = (p_ref/p_0)^κ = exp(−κ g z / (R_d T_ref)),
#     θ_ref(z) = T_ref / Π_ref(z),   ρ_ref = p_ref / (R_d T_ref).
# The momentum pressure-gradient + gravity is written as the deviation
#     −ρ c_pd (θ ∇Π' + θ' ∇Π_ref),   Π' = Π − Π_ref,  θ' = θ − θ_ref,
# which is the identically-zero field at rest (θ'=Π'=0).
const T_ref = parse(FT, get(ENV, "REF_TEMP", "250"))
const ᶜΠ_ref = @. exp(-κ_gas * grav * ccoords.z / (R_d * T_ref))
const ᶜθ_ref = @. T_ref / ᶜΠ_ref
# Pressure/density of the isothermal reference (for the stratified conservative
# PGF, PGF=conservative_pert: p' = p − p_ref carried in the momentum flux,
# buoyancy −(ρ−ρ_ref)g in ρw). Same reference as the Exner form.
const ᶜp_ref = @. p_0 * exp(-grav * ccoords.z / (R_d * T_ref))
const ᶜρ_ref = @. ᶜp_ref / (R_d * T_ref)
# IC = baroclinic (default, Ullrich et al. jet) | resting (quiescent
# isothermal = the reference; the sphere C-property / well-balancedness witness)
const ic_mode = lowercase(get(ENV, "IC", "baroclinic"))
ic_mode in ("baroclinic", "resting") ||
    error("IC must be baroclinic or resting")

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

# Initial total specific humidity. Two modes (MOISTURE_IC):
#   "rh"    (default): q_tot = RH0 · q_sat(T_v, ρ). Subsaturated everywhere, so no
#           immediate condensation, but the absolute moisture inherits the cool
#           base-state temperature (⇒ low surface q_tot, e.g. ~3 g/kg at RH0=0.1).
#   "dcmip": the DCMIP-2016 (Ullrich et al. 2016, test 4) explicit specific-humidity
#           profile q_0·exp[−(z/z_q1)²]·exp[−(z/z_q2)⁴], which targets ~q_0 (18 g/kg)
#           at the surface, but CAPPED at RH_MAX·q_sat so it stays subsaturated: a
#           bare fixed profile is uniform in latitude and would supersaturate the cold
#           polar columns (q_sat there ~1 g/kg ⇒ ~18× supersaturation ⇒ latent-heat
#           shock at t=0). The cap gives ~q_0 where it is warm enough to hold it and
#           RH_MAX·q_sat where it is not.
# All capped at the tropopause z_t.
const q_rh0 = parse(FT, get(ENV, "RH0", "0.8"))
const moisture_ic = get(ENV, "MOISTURE_IC", "rh")
const q_sfc0 = parse(FT, get(ENV, "QT0", "0.018"))   # DCMIP q_0 [kg/kg]
const z_q1 = parse(FT, get(ENV, "ZQ1", "3000"))      # DCMIP vertical scale 1 [m]
const z_q2 = parse(FT, get(ENV, "ZQ2", "8000"))      # DCMIP vertical scale 2 [m]
const rh_max = parse(FT, get(ENV, "RH_MAX", "1.0"))  # saturation cap for "dcmip"
moisture_ic in ("rh", "dcmip") ||
    error("MOISTURE_IC must be \"rh\" or \"dcmip\"")

# Internal energy per mass for the moist IC at (ρ, T, q_tot): equilibrium
# condensate partition then the moisture-weighted internal energy (referenced to
# T_0, consistent with the RHS saturation adjustment). Reduces to cv_d(T−T_0)
# when q_tot = 0, matching the dry driver.
@inline function ic_eint(ρ, T, q_tot)
    (q_liq, q_ice) = TD.condensate_partition(thermo_params, T, ρ, q_tot)
    return TD.internal_energy(thermo_params, T, q_tot, q_liq, q_ice)
end

function initial_state(ᶜlocal_geometry, ᶠlocal_geometry)
    (; lat, long, z) = ᶜlocal_geometry.coordinates

    # Quiescent isothermal atmosphere = the Exner reference,
    # so Π'≡0, θ'≡0 and the Exner-perturbation PGF is exactly zero.
    # Well-balanced rest-state on the sphere
    if ic_mode == "resting"
        ᶜp = @. p_0 * exp(-grav * z / (R_d * T_ref))
        ᶜρr = @. ᶜp / (R_d * T_ref)
        ᶜρer = @. cv_d * ᶜp / R_d + ᶜρr * (grav * z - cv_d * T_tri)
        ᶜuₕr = @. C12(Geometry.UVVector(FT(0), FT(0)), ᶜlocal_geometry)
        ᶠwr = map(_ -> C3(FT(0)), ᶠlocal_geometry)
        # Dry quiescent rest state (q_tot = 0): the well-balancedness witness.
        Ycr = map(
            (ρi, ρei) -> (; ρ = ρi, ρe = ρei, ρq_tot = zero(ρi)),
            ᶜρr,
            ᶜρer,
        )
        return Fields.FieldVector(Yc = Ycr, uₕ = ᶜuₕr, w = ᶠwr)
    end

    # MOIST baroclinic wave in GEOSTROPHIC + hydrostatic balance (DCMIP2016-style
    # virtual-temperature construction): the Ullrich analytic temperature is taken
    # as the VIRTUAL temperature T_v, so density ρ = p/(R_d T_v), pressure p, and the
    # wind u_base remain EXACTLY the balanced dry fields — adding moisture does not
    # perturb the momentum balance (this removes the geostrophic IC imbalance that
    # otherwise spins up spurious O(50–100 m/s) meridional wind). The actual
    # temperature follows from p = ρ R_m T = ρ R_d T_v ⇒ T = T_v · R_d/R_m, and the
    # diagnosed moist pressure ρ R_m T = ρ R_d T_v ≡ p is the analytic (balanced) p.
    # RH0=0 recovers R_m=R_d, T=T_v, ρ=ρ_dry — i.e. exactly the dry balanced state.
    ᶜTv = @. temp(lat, z)                                   # analytic T ≡ virtual temp
    ᶜρ = @. pres(lat, z) / (R_d * ᶜTv)                      # = dry-balanced density
    ᶜq_sat = @. TD.q_vap_saturation(thermo_params, ᶜTv, ᶜρ)
    ᶜq_tot =
        moisture_ic == "dcmip" ?
        (@. min(
            q_sfc0 * exp(-(z / z_q1)^2) * exp(-(z / z_q2)^4),
            rh_max * ᶜq_sat,
        ) * (z ≤ z_t)) :
        (@. q_rh0 * ᶜq_sat * (z ≤ z_t))
    ᶜR_m = @. TD.gas_constant_air(thermo_params, ᶜq_tot, FT(0), FT(0))
    ᶜT = @. ᶜTv * R_d / ᶜR_m                                # actual temperature
    u₀ = @. u_base(lat, z)
    v₀ = @. 0 * z
    if !is_balanced_flow
        @. u₀ += δu(long, lat, z)
        @. v₀ += δv(long, lat, z)
    end
    ᶜuₕ_local = @. Geometry.UVVector(u₀, v₀)
    ᶜuₕ = @. C12(ᶜuₕ_local, ᶜlocal_geometry)

    # Discrete hydrostatic balance (column-wise), MOISTURE-CONSISTENT: enforce the
    # centered face balance on the pressure the DYNAMICS actually diagnose,
    # p = ρ·R_m·T (moist_p_dyn), NOT the dry analytic p. The analytic state satisfies
    # ∂z p = −ρg only in the continuum; on the staggered FD grid the residual projects
    # onto gravity modes and drives spurious w. Holding T, q_tot, R_m at their analytic
    # values and writing aₖ ≡ R_m[k]·T[k] (so p = ρ a), the centered balance
    #   (ρa)[v+1] − (ρa)[v] = −gΔz (ρ[v]+ρ[v+1])/2
    # solves column-upward (bottom ρ = analytic ⇒ surface p = analytic p) to
    #   ρ[v+1] = ρ[v]·(2 a[v] − gΔz)/(2 a[v+1] + gΔz).
    # Using the moist RT (a = R_m·T) removes the O(q_tot) virtual-temperature
    # imbalance the earlier dry-p rebalance left in the moist column — the seed of
    # the fast initial adjustment. REBALANCE=0 uses the raw analytic state (residual
    # truncation-level under the Exner reference-subtracted PGF; generalizes to
    # data-derived ERA5/sounding ICs).
    if get(ENV, "REBALANCE", "1") == "1"
        ᶜa = @. ᶜR_m * ᶜT
        ρ_par = parent(ᶜρ)
        a_par = parent(ᶜa)
        # per-interface Δz from the actual center heights (supports ZSTRETCH;
        # a uniform zmax/zelem here silently corrupts ρ on stretched grids)
        z_par = parent(z)
        for v in 1:(size(ρ_par, 1) - 1)
            @views @. ρ_par[v + 1, :, :, :, :] =
                ρ_par[v, :, :, :, :] *
                (2 * a_par[v, :, :, :, :] -
                 grav * (z_par[v + 1, :, :, :, :] - z_par[v, :, :, :, :])) /
                (2 * a_par[v + 1, :, :, :, :] +
                 grav * (z_par[v + 1, :, :, :, :] - z_par[v, :, :, :, :]))
        end
    end
    ᶜK = @. norm_sqr(ᶜuₕ_local) / 2
    # Moist total energy: internal (with latent-heat reference) + kinetic +
    # geopotential; internal energy from the moist thermodynamics at the analytic
    # temperature and the RH-based q_tot (both set above, with the rebalanced ρ).
    ᶜeint = @. ic_eint(ᶜρ, ᶜT, ᶜq_tot)
    ᶜρe = @. ᶜρ * (ᶜeint + ᶜK + grav * z)

    ᶠw = map(_ -> C3(FT(0)), ᶠlocal_geometry)
    Yc = map(
        (ρi, ρei, qti) -> (; ρ = ρi, ρe = ρei, ρq_tot = ρi * qti),
        ᶜρ,
        ᶜρe,
        ᶜq_tot,
    )
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

# All DG building blocks — the Kennedy-Gruber two-point/interface fluxes and
# the central lifting / jump-penalty face functions (`lifting_correction`) —
# come from ClimaCore's Operators module; no operators are defined here.

# κ₄ biharmonic (SIPG) hyperdiffusion has been REMOVED from this setup: there is
# no KAPPA4 knob and no `ldg_laplacian_tendency` call. Grid-scale dissipation, if
# needed, comes from the interface (Roe/Rusanov) numerical flux and the optional
# velocity-state spectral filter below.

# Default OFF: the cutoff filter voids the KEP property of the flux-differencing
# scheme (the driver @warns on FILTER>0), and its tensor_product! has no GPU
# dispatch for the extruded layout (scalar-indexing error on CUDA). Prefer the
# interface dissipation / velocity-state filter.
const filter_Nc = parse(Int, get(ENV, "FILTER", "0"))

# Optional per-step exponential filter on the VELOCITY state (uₕ, w).
# The tendency cutoff above starves the top modes of forcing, but the HEVI
# implicit update bypasses it and nonlinear products regenerate top-mode
# content in the state, so noise still accumulates (helem=4 FILTER=3 GPU run
# crashed at t = 560,400 s). Filtering the state is the classical SEM cure:
# a modal projection, unconditionally stable (not Δt-limited like an explicit SIPG hyperdiffusion),
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
    # True global minimum cell spacing over ALL columns (an earlier version
    # read one hard-coded column, so it reported the unwarped Δz even with
    # topography — misleading). Over terrain the near-surface cells under the
    # ridges are compressed, so this is < zmax/zelem there.
    Δz_min = minimum(parent(Fields.Δz_field(hv_center_space)))
    c_max = sqrt(γ * R_d * T_e)
    # Peak surface elevation (bottom face physical z): 0 for flat, ≈h₀ for hj.
    max_mtn = maximum(parent(Fields.level(fcoords.z, ClimaCore.Utilities.half)))
    @info "DG-FD sphere setup" stepper topo max_mtn helem npoly zelem Δt t_end filter_Nc h_node Δz_min
    @info "Acoustic CFL estimates" vertical = c_max * Δt / Δz_min horizontal =
        c_max * Δt / h_node
end
