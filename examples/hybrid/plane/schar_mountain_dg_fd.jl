#=
Well-balancedness over topography — DG horizontal + FD vertical
================================================================

A numerical test of the discrete well-balancedness property for the
flux-form hybrid DG(horizontal) + FD(vertical) compressible-Euler
solver over the Schär (2002) terrain-following coordinate.

The topography warp is the Schär ridge  h(x) = h₀ exp(−(x/a)²) cos²(πx/λ)  (h₀=250 m,
a=5 km, λ=4 km). 

Reference (resting) state: the constant-N stratified atmosphere of Schär et al.
(2002) §3(b) — θ(z)=θ₀exp(N²z/g), θ₀=280 K, N=0.01 s⁻¹ — which satisfies the
*continuous* hydrostatic relation dp/dz = −ρg analytically. Sampling it at the
warped node heights makes the momentum-driving deviation fields the exact zero
field:  p′ = p − p_bg(z) ≡ 0  and  (ρ − ρ_bg(z)) ≡ 0.  

This driver initializes u ≡ 0 (WELL_BALANCED test, default) and reports, at
t=0, the max normalized tendency of every prognostic, an empirical test of the proof. 
It then integrates and tracks max|w|, which must stay at roundoff.

  julia --project=.buildkite examples/hybrid/plane/schar_mountain_dg_fd.jl

Environment: HELEM, VELEM, NPOLY, DT, T_END, U0 (default 0), FILTER (default 0),
             WARP (default schar; set flat for the sanity control).
=#

using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

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
import ClimaCore.Geometry: ⊗

import ClimaTimeSteppers as CTS

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

# ---------------------------------------------------------------------------
# Terrain — Schär (2002) §3(b) sinusoidally modulated Gaussian ridge
# ---------------------------------------------------------------------------
function warp_schar(coord)
    x = Geometry.component(coord, 1)
    a = FT(5000)
    λ = FT(4000)
    h₀ = FT(250)
    return abs(x) <= a ? h₀ * exp(-(x / a)^2) * (cos(π * x / λ))^2 : FT(0)
end

# ---------------------------------------------------------------------------
# Mesh — periodic x, extruded z, optional terrain-following warp
# ---------------------------------------------------------------------------
const xlim0 = (-60000.0, 60000.0)   # 120 km periodic (Schär domain)
const zlim0 = (0.0, 25000.0)        # 25 km lid
const helem0 = 32
const velem0 = 40
const npoly0 = 4

function hvspace_2D(
    xlim = xlim0,
    zlim = zlim0,
    helem = helem0,
    velem = velem0,
    npoly = npoly0;
    warp_fn = warp_schar,
)
    context = ClimaComms.context()
    device = ClimaComms.device(context)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)

    if isnothing(warp_fn)
        vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)
        center_space =
            Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    else
        # Terrain-following: build from faces so the surface coincides with the
        # lowest face, then wrap with the Gal–Chen LinearAdaption.
        vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)
        z_surface =
            Geometry.ZPoint.(warp_fn.(Fields.coordinate_field(horzspace)))
        face_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
            Hypsography.LinearAdaption(z_surface),
        )
        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)
    end
    return (center_space, face_space)
end

const helem = parse(Int, get(ENV, "HELEM", string(helem0)))
const velem = parse(Int, get(ENV, "VELEM", string(velem0)))
const npoly = parse(Int, get(ENV, "NPOLY", string(npoly0)))
const warp_mode = get(ENV, "WARP", "schar")
const warp_fn = warp_mode == "flat" ? nothing : warp_schar

hv_center_space, hv_face_space =
    hvspace_2D(xlim0, zlim0, helem, velem, npoly; warp_fn = warp_fn)

# ---------------------------------------------------------------------------
# Physics — dry compressible Euler, total energy; constant-N reference
# ---------------------------------------------------------------------------
const MSLP = 1e5
const grav = 9.8
const R_d = 287.058
const γ = 1.4
const C_p = R_d * γ / (γ - 1)
const C_v = R_d / (γ - 1)
const θ₀ = 280.0          # surface potential temperature (Schär §3b)
const N = 0.01            # Brunt–Väisälä frequency (s⁻¹)
const u₀ = parse(FT, get(ENV, "U0", "0.0"))   # resting by default (C-property)

const eq = (; γ = γ, Rgas = R_d, cₚ = C_p, g = grav, p_ref = MSLP, θ₀ = θ₀)

Φ(z) = grav * z

# Constant-N stratified hydrostatic reference. Verified analytically to satisfy
# dp_bg/dz = −ρ_bg g exactly, so the discrete IC has p′ ≡ 0, buoyancy ≡ 0.
θ_ref(z) = θ₀ * exp(N^2 * z / grav)
π_exner_ref(z) = 1 + grav^2 / (N^2 * C_p * θ₀) * (exp(-N^2 * z / grav) - 1)
background_pressure(z) = MSLP * π_exner_ref(z)^(C_p / R_d)
background_density(z) =
    MSLP / (R_d * θ_ref(z)) * π_exner_ref(z)^(C_v / R_d)

# Thermodynamic pressure: p = (γ−1)(ρe − ρK − ρΦ)  (T₀ = 0 energy convention)
pressure_eos(ρ, ρe, K, z) = (γ - 1) * (ρe - ρ * K - ρ * Φ(z))

_ke_u(u) = (u.u^2) / 2

function thermo_pressure(state, eq, z)
    u = state.ρu / state.ρ
    K = _ke_u(u) + state.w^2 / 2
    return pressure_eos(state.ρ, state.ρe, K, z)
end

momentum_pressure(state, eq, z) =
    thermo_pressure(state, eq, z) - background_pressure(z)

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

function entropy_variables(state, eq, z)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    K = _ke_u(u) + state.w^2 / 2
    p = pressure_eos(ρ, ρe, K, z)
    s̃ = log(p) - eq.γ * log(ρ)
    T_s = p / ρ
    return ((eq.γ - s̃) / (eq.γ - 1) - K / T_s, u.u / T_s, -1.0 / T_s)
end

roe_average(ρ⁻, ρ⁺, a⁻, a⁺) =
    (sqrt(ρ⁻) * a⁻ + sqrt(ρ⁺) * a⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

# Horizontal physical flux: p′ in momentum, full p in energy.
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

function ρw_roe(normal, (ρw⁻, uₕ⁻), (ρw⁺, uₕ⁺))
    un̄ = ((uₕ⁻ + uₕ⁺) / 2)' * normal
    return un̄ * (ρw⁻ + ρw⁺) / 2 - abs(un̄) / 2 * (ρw⁺ - ρw⁻)
end

# ---------------------------------------------------------------------------
# Init — resting (u₀=0) constant-N reference sampled at warped node heights
# ---------------------------------------------------------------------------
function init_schar(x, z)
    p_bg = background_pressure(z)
    ρ = background_density(z)
    K = u₀^2 / 2
    # ρe from EOS inverse at rest/uniform-u: p = (γ−1)(ρe − ρK − ρΦ)
    ρe = p_bg / (γ - 1) + ρ * (K + Φ(z))
    return (ρ = ρ, ρe = ρe)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_schar(coord.x, coord.z), coords)
Y = Fields.FieldVector(
    Yc = Yc,
    ρuₕ = Yc.ρ .* Ref(Geometry.UVector(u₀)),
    ρw = map(_ -> Geometry.WVector(FT(0)), face_coords),
)

const mass_0 = sum(Y.Yc.ρ)
const energy_0 = sum(Y.Yc.ρe)

const Δt = parse(FT, get(ENV, "DT", "0.4"))

# ---------------------------------------------------------------------------
# RHS (flux-form, orthonormal momentum) — identical structure to
# rising_bubble_2d_dg_fd.jl; only the reference state and mesh differ.
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
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.Contravariant3Vector(FT(0))),
        top = Operators.SetDivergence(Geometry.Contravariant3Vector(FT(0))),
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
    ∂f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    uₕ = @. ρuₕ / ρ
    w_c = @. Ic(ρw).components.data.:1 / ρ
    u_c = @. uₕ.components.data.:1
    K_full = @. (u_c^2 + w_c^2) / 2

    # --- Horizontal DG (no DSS): Souza KG + Roe ---
    y = map(
        (ρi, ρei, ρui, wi) -> (; ρ = ρi, ρe = ρei, ρu = ρui, w = wi),
        ρ,
        ρe,
        ρuₕ,
        w_c,
    )
    Fh = map((s, zi) -> flux_h(s, eq, zi), y, z)
    dy_mw = @. hwdiv(Fh) * (-(lgeom_c.WJ))
    Operators.add_numerical_flux_internal!(numflux_h, dy_mw, y, eq, z)
    @. dy_mw = dy_mw / lgeom_c.WJ
    @. dYc.ρ = dy_mw.ρ
    @. dYc.ρe = dy_mw.ρe
    @. dρuₕ = dy_mw.ρu

    # --- Vertical FD: stratified p′ split ---
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
    ρw_c = @. Ic(ρw)
    w_c_vert = @. ρw_c / ρ
    @. dρw = B(
        Geometry.project(Geometry.WAxis(), -(∂f(p′))) +
        If(Geometry.WVector(buoy)) -
        vvdivc2f(ρw_c ⊗ w_c_vert),
    )

    # Horizontal advection of ρw
    uₕf = @. If(uₕ)
    dρw_mw = @. hwdiv(uₕf ⊗ ρw) * (-(lgeom_f.WJ))
    Operators.add_numerical_flux_internal!(ρw_roe, dρw_mw, ρw, uₕf)
    @. dρw += dρw_mw / lgeom_f.WJ

    # Optional element-local cutoff filter (off by default for a clean witness)
    nc = parse(Int, get(ENV, "FILTER", "0"))
    if nc > 0
        M = Quadratures.cutoff_filter_matrix(
            FT,
            Spaces.quadrature_style(axes(ρ)),
            nc,
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
# t = 0 C-property witness — the numerical proof of well-balancedness.
# For a balanced resting reference, every tendency must be at machine precision.
# ---------------------------------------------------------------------------
dY = similar(Y)
rhs!(dY, Y, nothing, 0.0)

maxabs(f) = maximum(abs, parent(Fields.field_values(f)))
ρg = grav * maximum(abs, parent(Fields.field_values(Y.Yc.ρ)))  # PGF scale (N/m³)

@info "C-property witness at t=0 (warp=$(warp_mode), u₀=$(u₀))" begin
    dρw_max = maxabs(dY.ρw)
    dρuₕ_max = maxabs(dY.ρuₕ)
    (;
        max_dρw = dρw_max,
        max_dρw_over_ρg = dρw_max / ρg,
        max_dρuₕ = dρuₕ_max,
        max_dρuₕ_over_ρg = dρuₕ_max / ρg,
        max_dρ = maxabs(dY.Yc.ρ),
        max_dρe = maxabs(dY.Yc.ρe),
    )
end

# ---------------------------------------------------------------------------
# Integration — a resting balanced state must stay at rest (max|w| ≈ roundoff)
# ---------------------------------------------------------------------------
const t_end = parse(FT, get(ENV, "T_END", "3600.0"))
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs!),
    Y,
    (0.0, t_end),
    nothing,
)
integrator = CTS.init(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:parse(FT, get(ENV, "SAVEAT", "300.0")):t_end),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)
sol = CTS.solve!(integrator)

Ic_out = Operators.InterpolateF2C()
for (t_s, u) in zip(sol.t, sol.u)
    w = @. Ic_out(u.ρw).components.data.:1 / u.Yc.ρ
    us = @. (u.ρuₕ / u.Yc.ρ).components.data.:1
    @info "frame" t = t_s max_w = maximum(abs, parent(Fields.field_values(w))) max_u_dev =
        maximum(abs, parent(Fields.field_values(us)) .- u₀) Δmass =
        (sum(u.Yc.ρ) - mass_0) / mass_0 Δenergy = (sum(u.Yc.ρe) - energy_0) / energy_0
end

@info "Schär DG-FD well-balanced test finished" n_frames = length(sol.u) t_final =
    sol.t[end]
