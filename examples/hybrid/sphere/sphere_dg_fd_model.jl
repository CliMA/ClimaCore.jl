#=
Shared model for spherical (cubed-sphere shell) compressible-Euler test cases
with
  • discontinuous Galerkin (DG) horizontal spectral elements (no DSS)
  • finite-difference vertical staggering (Atmos-like)

Sphere analog of the plane drivers `plane/vector_invariant_2d_dg_fd.jl`:
horizontal momentum in vector-invariant form, scalars (ρ, ρe) in flux form.

State: Yc = (ρ, ρe) centers; uₕ Covariant12Vector centers; w Covariant3Vector
faces. ρe is total energy density (internal + kinetic + geopotential).

Horizontal DG treatment (all face quantities in the local orthonormal
geographic frame, which is single-valued at shared nodes — including across
cubed-sphere panel edges, where covariant components are not):
  • (ρ, ρe): flux-differencing (FDDG) volume terms with the Kennedy-Gruber
    two-point flux (KEP property; Souza et al. 2023, JAMES), and the same KG
    flux as the central part of the Rusanov-penalized interface flux.
  • Non-conservative gradients ∇ₕp, ∇ₕK and the curls feeding ω³ / ω¹²:
    element-local strong operators completed by symmetric central face
    lifting (`add_lifting_flux_internal!`), the DG analog of CG grad + DSS.
  • Velocity jumps [[u]], [[v]], [[w]]: λ-scaled interface penalties
    (λ = |uₕ| + c) through the same lifting.
Vertical FD: mass via face mass flux; energy via Lin–van Leer upwind;
w = 0 and ∂z(·) = 0 at top/bottom (CG-model boundary conditions).
Time stepping (STEPPER): "explicit" = fully explicit SSP-RK3 (Δt limited by
the vertical acoustic CFL); "hevi" = IMEX ARK with the vertical acoustic
terms implicit (column-wise Newton solve with the analytic Jacobian from
`sphere_dg_fd_jacobian.jl`; central implicit vertical energy flux instead
of Lin–van Leer; Δt limited by the horizontal DG acoustic CFL).
Stabilization: κ₄ biharmonic hyperdiffusion ONLY (no κ₂), two-pass:
element-local first Laplacian, then SIPG (LDG penalty) second pass for
inter-element damping; applied to h_tot = (ρe+p)/ρ and the geographic
velocity components (u, v) — never to ρ or w. Optional element-local
cutoff filter on the tendencies.

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
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
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
c3(λ, ϕ) = cos(π * r_gc(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(π * r_gc(λ, ϕ) / 2 / d_0)
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
    dz = zmax / zelem
    for v in 1:(size(ρ_par, 1) - 1)
        @views @. ρ_par[v + 1, :, :, :, :] =
            -ρ_par[v, :, :, :, :] -
            2 * (p_par[v + 1, :, :, :, :] - p_par[v, :, :, :, :]) / dz / grav
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

# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const hcurl = Operators.Curl()

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
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(FT(0), FT(0))),
    top = Operators.SetCurl(CT12(FT(0), FT(0))),
)
const Bw = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)

# Shared tendency core. With `vertical_transport = true` this is the full
# (explicit) tendency; with `false` it is the HEVI explicit part, i.e.
# everything except the vertical acoustic terms handled by
# `implicit_tendency!` (vertical (ρ, ρe) transport and the pressure-gradient
# + buoyancy terms of the w equation).
function compute_tendency!(dY, Y, t, vertical_transport::Bool)
    ρ = Y.Yc.ρ
    ρe = Y.Yc.ρe
    uₕ = Y.uₕ
    w = Y.w
    dYc = dY.Yc
    duₕ = dY.uₕ
    dw = dY.w

    lgeom_c = Fields.local_geometry_field(hv_center_space)
    lgeom_f = Fields.local_geometry_field(hv_face_space)

    # --- Diagnostics ---
    w_c = @. Ic(Geometry.WVector(w))
    uv = @. Geometry.UVVector(uₕ)          # geographic components
    u_sc = uv.components.data.:1
    v_sc = uv.components.data.:2
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p = @. pressure_ρe(ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p) / ρ
    c_snd = @. sqrt(γ * p / ρ)
    λ_c = @. sqrt(norm_sqr(uv)) + c_snd
    λ_f = @. If(λ_c)
    ρ_f = @. If(ρ)
    w_sc = @. Geometry.WVector(w).components.data.:1

    # --- (ρ, ρe): horizontal flux-form DG ---
    y = map(
        (ρi, ρei, pi, λi, uvi, ei) ->
            (; ρ = ρi, ρe = ρei, p = pi, λ = λi, uv = uvi, e = ei),
        ρ,
        ρe,
        p,
        λ_c,
        uv,
        ρe ./ ρ,
    )
    # Flux-differencing (FDDG) volume terms with the Kennedy-Gruber
    # two-point flux (KEP property; Souza et al. 2023), and the same KG
    # flux as the central part of the interface flux.
    dy_mw = map(_ -> (ρ = FT(0), ρe = FT(0)), ρ)
    Operators.add_flux_differencing_divergence!(
        Operators.kennedy_gruber_scalars_flux,
        dy_mw,
        y,
    )
    Operators.add_numerical_flux_internal!(
        Operators.kennedy_gruber_rusanov_scalars,
        dy_mw,
        y,
    )
    @. dYc.ρ = dy_mw.ρ / lgeom_c.WJ
    @. dYc.ρe = dy_mw.ρe / lgeom_c.WJ

    # --- (ρ, ρe): vertical FD (implicit under HEVI) ---
    if vertical_transport
        w_vec = @. Geometry.WVector(w)
        @. dYc.ρ -= vdivf2c(ρ_f * w_vec)
        @. dYc.ρe -= vdivf2c(ρ_f * VanLeer(w_vec, h_tot, Δt))
    end

    # --- Vorticities (element-local strong curl + central face lifting) ---
    ω³_sc = @. Geometry.WVector(hcurl(uₕ)).components.data.:1
    ω³_sc .+=
        Operators.lifting_correction(Operators.central_curl3_lift, FT, u_sc, v_sc)
    ω³ = @. CT3(Geometry.WVector(ω³_sc))

    ᶠω¹² = @. hcurl(w)
    ᶠω¹² .+= Geometry.transform.(
        Ref(Geometry.Contravariant12Axis()),
        Operators.lifting_correction(
            Operators.central_curl12_lift,
            Geometry.UVVector{FT},
            w_sc,
        ),
    )
    @. ᶠω¹² += ᶠcurlᵥ(uₕ)

    ᶠu¹² = @. CT12(If(uₕ))
    ᶠu³ = @. CT3(w)

    # --- Horizontal momentum (vector-invariant) ---
    @. duₕ = -(Ic(ᶠω¹² × ᶠu³) + (ᶜf_cor + ω³) × CT12(uₕ))
    @. duₕ -= hgrad(p) / ρ + hgrad(K + ᶜΦ)
    # DG lifting corrections for the strong gradients (Φ is continuous)
    lift_p = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        p,
    )
    lift_K = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        K,
    )
    @. duₕ -= Geometry.transform(
        Geometry.Covariant12Axis(),
        lift_p / ρ + lift_K,
    )
    # λ-scaled jump penalties on the geographic velocity components
    pen_u = Operators.lifting_correction(Operators.jump_penalty_lift, FT, u_sc, λ_c)
    pen_v = Operators.lifting_correction(Operators.jump_penalty_lift, FT, v_sc, λ_c)
    @. duₕ += Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.UVVector(pen_u, pen_v),
    )

    # --- Vertical momentum (acoustic terms implicit under HEVI) ---
    if vertical_transport
        @. dw = -(ᶠgradᵥ(p) / If(ρ) + ᶠgradᵥ(K + ᶜΦ))
        @. dw -= ᶠω¹² × ᶠu¹²
    else
        @. dw = -(ᶠω¹² × ᶠu¹²)
    end
    pen_w = Operators.lifting_correction(Operators.jump_penalty_lift, FT, w_sc, λ_f)
    @. dw += C3(Geometry.WVector(pen_w), lgeom_f)
    @. dw = Bw(dw)

    # --- κ₄ hyperdiffusion (two-pass, SIPG-coupled; h_tot and (u, v) only;
    # no diffusion of ρ or w; no κ₂ anywhere) ---
    if κ₄ != 0
        τ_κ₄ = Operators.ldg_penalty_parameter(κ₄, hv_center_space)
        # Element-local first Laplacian (as in the plane DG-FD drivers). A
        # DG-consistent (lifted-gradient) first pass was tested and is
        # unstable at this τ/κ₄; the element-local form is the validated one.
        χe = similar(h_tot)
        @. χe = hwdiv(hgrad(h_tot))
        χu = similar(u_sc)
        @. χu = hwdiv(hgrad(u_sc))
        χv = similar(v_sc)
        @. χv = hwdiv(hgrad(v_sc))
        de4 = Operators.ldg_laplacian_tendency(χe, ρ, κ₄, τ_κ₄)
        du4 = Operators.ldg_laplacian_tendency(χu, nothing, κ₄, τ_κ₄)
        dv4 = Operators.ldg_laplacian_tendency(χv, nothing, κ₄, τ_κ₄)
        @. dYc.ρe -= de4
        @. duₕ -= Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(du4, dv4),
        )
    end

    # --- Held–Suarez forcing (Rayleigh low-level drag + Newtonian T relaxation) ---
    if apply_held_suarez
        φ = @. deg2rad(ccoords.lat)
        σ = @. p / p_0
        height_factor = @. max(0, (σ - σ_b) / (1 - σ_b))
        ΔρT = @. (k_a + (k_s - k_a) * height_factor * cos(φ)^4) *
           ρ *
           (
               p / (ρ * R_d) - max(
                   T_min,
                   (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * σ^(R_d / cp_d),
               )
           )
        @. duₕ -= (k_f * height_factor) * uₕ
        @. dYc.ρe -= ΔρT * cv_d
    end

    # --- Element-local cutoff filter on the tendencies ---
    if filter_Nc > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(hv_center_space),
            filter_Nc,
        )
        for f in (dYc.ρ, dYc.ρe, duₕ, dw)
            data = Fields.field_values(f)
            Operators.tensor_product!(data, data, M)
        end
        @. dw = Bw(dw)
    end

    return dY
end

# Fully explicit RHS (SSP-RK3 path) and the HEVI explicit part.
rhs!(dY, Y, p, t) = compute_tendency!(dY, Y, t, true)
remaining_tendency!(dY, Y, p, t) = compute_tendency!(dY, Y, t, false)

# HEVI implicit part: vertical acoustics (column-local, no DG coupling).
# The implicit vertical energy flux is the central If(ρe + p)·w of the CG
# staggered model — the TVD VanLeer flux of the fully explicit path is
# nonlinear in w and cannot be used inside the linearized Newton solve.
function implicit_tendency!(dY, Y, p, t)
    ρ = Y.Yc.ρ
    ρe = Y.Yc.ρe
    uₕ = Y.uₕ
    w = Y.w

    uv = @. Geometry.UVVector(uₕ)
    w_c = @. Ic(Geometry.WVector(w))
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p_thermo = @. pressure_ρe(ρe, K, ᶜΦ, ρ)

    w_vec = @. Geometry.WVector(w)
    @. dY.Yc.ρ = -vdivf2c(If(ρ) * w_vec)
    @. dY.Yc.ρe = -vdivf2c(If(ρe + p_thermo) * w_vec)
    dY.uₕ .= (zero(eltype(dY.uₕ)),)
    # ᶠgradᵥ's SetGradient(0) boundary conditions zero the boundary-face rows,
    # consistent with the Bw treatment of the explicit path.
    @. dY.w = -(ᶠgradᵥ(p_thermo) / If(ρ) + ᶠgradᵥ(K + ᶜΦ))
    return dY
end

include("sphere_dg_fd_jacobian.jl")

# ---------------------------------------------------------------------------
# Time integration (STEPPER = explicit | hevi)
# ---------------------------------------------------------------------------
function run_simulation(Y; dt_save)
    if stepper == "hevi"
        jacobian = DGImplicitEquationJacobian(Y)
        prob = SciMLBase.ODEProblem(
            CTS.ClimaODEFunction(;
                T_imp! = SciMLBase.ODEFunction(
                    implicit_tendency!;
                    jac_prototype = jacobian,
                    Wfact = implicit_equation_jacobian!,
                ),
                T_exp! = remaining_tendency!,
            ),
            Y,
            (FT(0), t_end),
            nothing,
        )
        ode_algo =
            CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
        return SciMLBase.solve(
            prob,
            ode_algo;
            dt = Δt,
            saveat = collect(FT(0):dt_save:t_end),
        )
    else
        prob = ODEProblem(rhs!, Y, (FT(0), t_end))
        return solve(
            prob,
            SSPRK33(),
            dt = Δt,
            saveat = dt_save,
            internalnorm = fieldvector_norm,
        )
    end
end

# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------
let
    h_node = Spaces.node_horizontal_length_scale(horzspace)
    Δz = zmax / zelem
    c_max = sqrt(γ * R_d * T_e)
    @info "DG-FD sphere setup" stepper helem npoly zelem Δt t_end κ₄ κ₄_cfl_cap filter_Nc h_node
    @info "Acoustic CFL estimates" vertical = c_max * Δt / Δz horizontal =
        c_max * Δt / h_node
end
