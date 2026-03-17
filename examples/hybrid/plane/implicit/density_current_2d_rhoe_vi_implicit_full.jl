#=
Density current 2D (vector-invariant form, total energy) with fully implicit
timestepping — ClimaAtmos-matching version.

This file replicates ALL the complexities from ClimaAtmos's fully implicit solver:
  - J-factor metric terms (ᶜJ/ᶠJ) in mass/energy tendencies
  - Contravariant velocity ᶠu³ = CT3(u₃) for advective fluxes
  - ᶜadvdivᵥ operator with CT3 boundary conditions
  - Density-weighted vorticity advection for horizontal momentum
  - vertical_transport pattern for energy: ᶠinterp(ρ*ᶜJ)/ᶠJ * ᶠu³ * ᶠinterp(h_tot)
  - Dual PGF Jacobian for ∂ᶠu₃/∂ᶜρ (ρ-expansion + Exner/θ_v gravity term)
  - Full ∂K/∂u₃ including horizontal velocity coupling (CT3(uₕ) term)
  - (f.u₃, c.uₕ) Jacobian block through KE coupling
  - ᶜadvdivᵥ_matrix with J-factors for mass/energy Jacobian blocks
  - BlockArrowheadSolve with BlockLowerTriangularSolve for velocity
  - ARS222 timestepper (matching ClimaAtmos config)
  - Newton rtol = 1e-4 (matching ClimaAtmos config)

Compare with density_current_2d_rhoe_vi_implicit.jl (the "simple" version) to
isolate which ClimaAtmos complexity causes accuracy issues.

State vector: Y.c = (ρ, ρe, uₕ), Y.f = (u₃,)
  - ρ:   density (center, scalar)
  - ρe:  total energy density (center, scalar)
  - uₕ:  horizontal velocity (center, Covariant1Vector)
  - u₃:  vertical velocity (face, Covariant3Vector)

Run:
    julia --project=. examples/hybrid/plane/implicit/density_current_2d_rhoe_vi_implicit_full.jl

Environment variables:
    DT    — timestep in seconds (default 3.0)
    T_END — end time in seconds (default 900.0)
=#

using Test
using LinearAlgebra
import LinearAlgebra: ldiv!

import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    ClimaCore,
    slab,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators

import ClimaCore.Geometry: ⊗

import ClimaTimeSteppers as CTS
import SciMLBase
using ClimaCore.MatrixFields
using ClimaCore.MatrixFields: @name

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

# Geometry type aliases
const C1 = Geometry.Covariant1Vector
const C3 = Geometry.Covariant3Vector
const C12 = Geometry.Covariant12Vector
const C13 = Geometry.Covariant13Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT3 = Geometry.Contravariant3Vector
const CT12 = Geometry.Contravariant12Vector

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
const MSLP = 1e5      # mean sea level pressure [Pa]
const grav = 9.80616   # gravitational acceleration [m/s²]
const R_d = 287.058    # gas constant for dry air [J/(kg·K)]
const γ_ratio = 1.4    # heat capacity ratio
const C_p = R_d * γ_ratio / (γ_ratio - 1)  # heat capacity at constant pressure
const C_v = R_d / (γ_ratio - 1)            # heat capacity at constant volume
const T_0 = 273.16    # triple point temperature [K]
const κ_d = R_d / C_v  # = γ for dry air (matching ClimaAtmos ᶜkappa_m)

Φ(z) = grav * z

# ---------------------------------------------------------------------------
# Thermodynamic functions for total energy formulation
# ---------------------------------------------------------------------------
function pressure_from_ρe(ρ, ρe, K, z)
    e = ρe / ρ
    I = e - Φ(z) - K
    T = I / C_v + T_0
    return ρ * R_d * T
end

function temperature_from_ρe(ρ, ρe, K, z)
    e = ρe / ρ
    I = e - Φ(z) - K
    return I / C_v + T_0
end

exner_from_p(p) = (p / MSLP)^(R_d / C_p)

# ---------------------------------------------------------------------------
# Reference state profiles (matching ClimaAtmos refstate_thermodynamics.jl)
# ---------------------------------------------------------------------------
const s_ref = 7
const T_min = 220.0  # K
const T_sfc = 290.0  # K

T_ref(Π) = T_min + (T_sfc - T_min) * Π^s_ref
θ_ref(Π) = T_ref(Π) / Π
Φ_ref(Π) = -C_p * (T_min * log(Π) + (T_sfc - T_min) / s_ref * (Π^s_ref - 1))

# ---------------------------------------------------------------------------
# Function space setup (density current domain: 51200m × 6400m)
# ---------------------------------------------------------------------------
function hvspace_2D(
    xlim = (0.0, 51200.0),
    zlim = (0.0, 6400.0),
    helem = 128,
    velem = 32,
    npoly = 4,
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
        Geometry.XPoint{FT}(xlim[2]),
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

hv_center_space, hv_face_space = hvspace_2D()

# ---------------------------------------------------------------------------
# Initial conditions: Dry density current (cold dome)
# ---------------------------------------------------------------------------
function init_density_current_2d(x, z)
    θ_b = 300.0
    x_c = 25600.0
    z_c = 2000.0
    x_r = 4000.0
    z_r = 2000.0

    r = sqrt(((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2)
    θ_p = r < 1.0 ? -15.0 / 2.0 * (1.0 + cospi(r)) : 0.0

    θ = θ_b + θ_p
    π_exn = 1.0 - grav * z / C_p / θ
    T = π_exn * θ
    p = MSLP * π_exn^(C_p / R_d)
    ρ = p / R_d / T
    e = C_v * (T - T_0) + grav * z  # total specific energy (K=0 at t=0)
    ρe = ρ * e
    return (ρ = ρ, ρe = ρe)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    ic = init_density_current_2d(x, z)
    uₕ = C1(FT(0))
    return (; ρ = ic.ρ, ρe = ic.ρe, uₕ)
end

function face_initial_condition(local_geometry)
    return (; u₃ = C3(FT(0)))
end

ᶜlocal_geometry = Fields.local_geometry_field(hv_center_space)
ᶠlocal_geometry = Fields.local_geometry_field(hv_face_space)

Y = Fields.FieldVector(
    c = center_initial_condition.(ᶜlocal_geometry),
    f = face_initial_condition.(ᶠlocal_geometry),
)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
energy_0 = sum(Y.c.ρe)
mass_0 = sum(Y.c.ρ)

# ---------------------------------------------------------------------------
# Vertical FD operators
# ---------------------------------------------------------------------------
const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

# C3-boundary divergence (used for simple divergence)
const ᶜdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)

# CT3-boundary divergence (matching ClimaAtmos ᶜadvdivᵥ for advective fluxes)
const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(FT(0))),
    top = Operators.SetValue(CT3(FT(0))),
)

const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(FT(0))),
    top = Operators.SetGradient(C3(FT(0))),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT2(FT(0))),
    top = Operators.SetCurl(CT2(FT(0))),
)

# Operator matrices for Jacobian
const ᶜadvdivᵥ_matrix = MatrixFields.operator_matrix(ᶜadvdivᵥ)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)

# Horizontal operators
const hdiv = Operators.Divergence()
const hgrad = Operators.Gradient()
const hwdiv = Operators.WeakDivergence()
const hwgrad = Operators.WeakGradient()
const hcurl = Operators.Curl()

# Additional vertical operators
const If = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const Ic = Operators.InterpolateF2C()

const vdivf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)

# ---------------------------------------------------------------------------
# Pre-allocated cache
# ---------------------------------------------------------------------------
function build_cache(Y)
    ᶜspace = axes(Y.c.ρ)
    ᶠspace = axes(Y.f.u₃)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    return (;
        p = similar(Y.c.ρ),         # pressure
        K = similar(Y.c.ρ),         # kinetic energy
        h_tot = similar(Y.c.ρ),     # total enthalpy
        T_field = similar(Y.c.ρ),   # temperature
        θ = similar(Y.c.ρ),         # potential temperature
        Π = similar(Y.c.ρ),         # Exner pressure
        θ_r = similar(Y.c.ρ),       # reference θ
        Φ_r = similar(Y.c.ρ),       # reference geopotential
        Δθ = similar(Y.c.ρ),        # perturbation θ
        # Face fields
        ᶠω¹² = similar(Y.f.u₃, CT2{FT}),  # horizontal vorticity on faces
        ᶠu_ct1 = similar(Y.f.u₃, CT1{FT}), # contravariant1 horiz vel on faces
        ᶠu_ct3 = similar(Y.f.u₃, CT3{FT}), # contravariant3 vert vel on faces
        # Metric fields (matching ClimaAtmos J-factor usage)
        ᶜJ = Fields.local_geometry_field(ᶜspace).J,
        ᶠJ = Fields.local_geometry_field(ᶠspace).J,
        # Constant fields
        ᶜΦ = Φ.(ᶜcoords.z),
    )
end

# ---------------------------------------------------------------------------
# RHS function (ClimaAtmos-matching tendency forms)
# ---------------------------------------------------------------------------
function rhs!(dY, Y, cache, t)
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    u₃ = Y.f.u₃

    z = coords.z

    (; p, K, h_tot, T_field, θ, Π, θ_r, Φ_r, Δθ,
       ᶠω¹², ᶠu_ct1, ᶠu_ct3, ᶜJ, ᶠJ, ᶜΦ) = cache

    # --- Diagnostics ---
    @. K = (norm(uₕ)^2 + norm(Ic(u₃))^2) / 2
    @. p = pressure_from_ρe(ρ, ρe, K, z)
    @. T_field = temperature_from_ρe(ρ, ρe, K, z)
    @. h_tot = (ρe + p) / ρ
    @. Π = exner_from_p(p)
    @. θ = T_field / Π

    # Reference state (functions of Exner pressure)
    @. θ_r = θ_ref(Π)
    @. Φ_r = Φ_ref(Π)
    @. Δθ = θ - θ_r

    # --- Vorticity (CT2 in 2D XZ) ---
    @. ᶠω¹² = hcurl(u₃)
    @. ᶠω¹² += ᶠcurlᵥ(uₕ)

    # --- Contravariant velocities ---
    # ᶠu³ = CT3(u₃) matching ClimaAtmos precomputed ᶠu³
    @. ᶠu_ct3 = CT3(u₃)
    # CT1 horizontal velocity on faces (for vertical momentum cross product)
    @. ᶠu_ct1 = CT1(C13(If(uₕ)))

    # === HYPERVISCOSITY (biharmonic on energy and horizontal momentum) ===
    @. dY.c.ρe = hwdiv(hgrad(h_tot))
    @. dY.c.uₕ = hwgrad(hdiv(uₕ))
    Spaces.weighted_dss!(dY.c)

    κ₄ = 1e7 # m⁴/s
    @. dY.c.ρe = -κ₄ * hwdiv(ρ * hgrad(dY.c.ρe))
    @. dY.c.uₕ = -κ₄ * hwgrad(hdiv(dY.c.uₕ))

    # === Mass conservation (ClimaAtmos form with J-factors) ===
    # ∂ρ/∂t = -ᶜadvdivᵥ(ᶠinterp(ρ·ᶜJ)/ᶠJ · ᶠu³) - divₕ(ρ·u)
    # Matches ClimaAtmos implicit_tendency.jl line 203
    @. dY.c.ρ = -(ᶜadvdivᵥ(ᶠinterp(ρ * ᶜJ) / ᶠJ * ᶠu_ct3))
    @. dY.c.ρ -= hdiv(ρ * uₕ)

    # === Total energy conservation (ClimaAtmos vertical_transport pattern) ===
    # vertical_transport(ρ, ᶠu³, h_tot, dt, Val(:none)):
    #   -(ᶜadvdivᵥ(ᶠinterp(ρ·ᶜJ)/ᶠJ · ᶠu³ · ᶠinterp(h_tot)))
    # Matches ClimaAtmos implicit_tendency.jl line 206
    @. dY.c.ρe += -(ᶜadvdivᵥ(ᶠinterp(ρ * ᶜJ) / ᶠJ * ᶠu_ct3 * ᶠinterp(h_tot)))
    @. dY.c.ρe -= hdiv(uₕ * (ρe + p))

    # === Horizontal momentum (ClimaAtmos density-weighted vorticity form) ===
    # Matches ClimaAtmos advection.jl line 292 (shallow atmosphere, no Coriolis)
    # Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × (ᶠinterp(ρ·ᶜJ) · ᶠu³)) / (ρ·ᶜJ)
    @. dY.c.uₕ += -(Ic(ᶠω¹² × (ᶠinterp(ρ * ᶜJ) * ᶠu_ct3)) / (ρ * ᶜJ))

    # Bernoulli gradient + split-form pressure gradient with reference state
    @. dY.c.uₕ -= hgrad(K + ᶜΦ - Φ_r) +
        C_p / 2 * (Δθ * hgrad(Π) + hgrad(Δθ * Π) - Π * hgrad(Δθ))

    # === Vertical momentum (ClimaAtmos form) ===
    # Matches ClimaAtmos advection.jl line 295 (shallow atmosphere)
    # Yₜ.f.u₃ -= ᶠω¹² × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
    # In 2D XZ: CT12(ᶜu) ≈ CT1(ᶜuₕ), but using ᶠu_ct1 which goes through C13
    @. dY.f.u₃ = -(ᶠω¹² × ᶠu_ct1)
    @. dY.f.u₃ -= ᶠgradᵥ(K)

    # Pressure gradient with reference state (matches ClimaAtmos implicit_tendency.jl line 289)
    @. dY.f.u₃ -= ᶠgradᵥ(ᶜΦ) - ᶠgradᵥ(Φ_r) +
        C_p * ᶠinterp(Δθ) * ᶠgradᵥ(Π)

    # === DIFFUSION ===
    κ₂ = 0.0  # m²/s
    @. dY.c.uₕ += hwdiv(κ₂ * hgrad(uₕ))
    @. dY.c.ρe += hwdiv(κ₂ * (ρ * hgrad(h_tot)))
    @. dY.c.ρe += vdivf2c(κ₂ * (ᶠinterp(ρ) * ᶠgradᵥ(h_tot)))

    Spaces.weighted_dss!(dY.c)
    Spaces.weighted_dss!(dY.f)
    return dY
end

# Build cache and verify rhs! works
const rhs_cache = build_cache(Y)
dYdt = similar(Y);
rhs!(dYdt, Y, rhs_cache, 0.0);

# ---------------------------------------------------------------------------
# Implicit Equation Jacobian (ClimaAtmos-matching preconditioner)
# ---------------------------------------------------------------------------
struct ImplicitEquationJacobian{TJ, RJ}
    ∂Yₜ∂Y::TJ
    ∂R∂Y::RJ
    transform::Bool
end

function ImplicitEquationJacobian(Y, transform)
    ᶜρ_name = @name(c.ρ)
    ᶜρe_name = @name(c.ρe)
    ᶜuₕ_name = @name(c.uₕ)
    ᶠu₃_name = @name(f.u₃)

    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}

    # NEW: (f.u₃, c.uₕ) block type — maps C1 → C3 through KE coupling
    # Matches ClimaAtmos BidiagonalRow_C3xACT12, but CT1 for 2D
    BidiagonalRow_C3xACT1 =
        BidiagonalMatrixRow{typeof(C3(FT(0)) * CT1(FT(0))')}

    ᶜspace = axes(Y.c.ρ)
    ᶠspace = axes(Y.f.u₃)

    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        # ∂ᶜρₜ/∂ᶠu₃: mass block (ClimaAtmos uses ᶜadvdivᵥ_matrix with J-factors)
        (ᶜρ_name, ᶠu₃_name) => Fields.zeros(BidiagonalRow_ACT3, ᶜspace),
        # ∂ᶜρeₜ/∂ᶠu₃: energy block (ClimaAtmos uses ᶜadvdivᵥ_matrix with J-factors & h_tot)
        (ᶜρe_name, ᶠu₃_name) => Fields.zeros(BidiagonalRow_ACT3, ᶜspace),
        # ∂ᶠu₃ₜ/∂ᶜρ: dual PGF (ClimaAtmos ρ-expansion + Exner/θ_v gravity)
        (ᶠu₃_name, ᶜρ_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠu₃ₜ/∂ᶜρe: pressure gradient, ∂p/∂ρe = κ = R_d/Cv
        (ᶠu₃_name, ᶜρe_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠu₃ₜ/∂ᶜuₕ: NEW — KE coupling through horizontal velocity
        # Matches ClimaAtmos manual_sparse_jacobian.jl line 590-591
        (ᶠu₃_name, ᶜuₕ_name) => Fields.zeros(BidiagonalRow_C3xACT1, ᶠspace),
        # ∂ᶠu₃ₜ/∂ᶠu₃: KE coupling via full ∂K/∂u₃
        (ᶠu₃_name, ᶠu₃_name) => Fields.zeros(TridiagonalRow_C3xACT3, ᶠspace),
    )

    I = MatrixFields.identity_field_matrix(Y)
    δtγ = FT(1)
    ∂R∂Y = transform ? I ./ δtγ .- ∂Yₜ∂Y : δtγ .* ∂Yₜ∂Y .- I

    # ClimaAtmos solver structure:
    # BlockArrowheadSolve with scalar thin variables (ρ, ρe)
    # and BlockLowerTriangularSolve for velocity block (uₕ, u₃)
    # This handles the (f.u₃, c.uₕ) coupling correctly.
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 354-406
    velocity_alg = MatrixFields.BlockLowerTriangularSolve(ᶜuₕ_name)
    alg = MatrixFields.BlockArrowheadSolve(
        ᶜρ_name, ᶜρe_name;
        alg₂ = velocity_alg,
    )

    return ImplicitEquationJacobian(
        ∂Yₜ∂Y,
        MatrixFields.FieldMatrixWithSolver(∂R∂Y, Y, alg),
        transform,
    )
end

Base.similar(j::ImplicitEquationJacobian) = ImplicitEquationJacobian(
    similar(j.∂Yₜ∂Y),
    similar(j.∂R∂Y),
    j.transform,
)

Base.zero(j::ImplicitEquationJacobian) = ImplicitEquationJacobian(
    zero(j.∂Yₜ∂Y),
    zero(j.∂R∂Y),
    j.transform,
)

ldiv!(
    δY::Fields.FieldVector,
    j::ImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

# ---------------------------------------------------------------------------
# wfact! — update the preconditioner Jacobian (ClimaAtmos-matching blocks)
# ---------------------------------------------------------------------------
function wfact!(j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, transform) = j
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃

    ᶜρ_name = @name(c.ρ)
    ᶜρe_name = @name(c.ρe)
    ᶜuₕ_name = @name(c.uₕ)
    ᶠu₃_name = @name(f.u₃)

    ∂ᶜρₜ∂ᶠu₃ = ∂Yₜ∂Y[ᶜρ_name, ᶠu₃_name]
    ∂ᶜρeₜ∂ᶠu₃ = ∂Yₜ∂Y[ᶜρe_name, ᶠu₃_name]
    ∂ᶠu₃ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠu₃_name, ᶜρ_name]
    ∂ᶠu₃ₜ∂ᶜρe = ∂Yₜ∂Y[ᶠu₃_name, ᶜρe_name]
    ∂ᶠu₃ₜ∂ᶜuₕ = ∂Yₜ∂Y[ᶠu₃_name, ᶜuₕ_name]
    ∂ᶠu₃ₜ∂ᶠu₃ = ∂Yₜ∂Y[ᶠu₃_name, ᶠu₃_name]

    # Metric factors
    ᶠgⁱʲ = Fields.local_geometry_field(ᶠu₃).gⁱʲ
    g³³(gⁱʲ) = Geometry.AxisTensor(
        (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
        Geometry.components(gⁱʲ)[end],
    )

    z = coords.z
    ᶜJ = p.ᶜJ
    ᶠJ = p.ᶠJ

    # --- Compute thermodynamic state ---
    @. p.K = (norm(ᶜuₕ)^2 + norm(Ic(ᶠu₃))^2) / 2
    @. p.p = pressure_from_ρe(ᶜρ, ᶜρe, p.K, z)
    @. p.T_field = temperature_from_ρe(ᶜρ, ᶜρe, p.K, z)
    @. p.h_tot = (ᶜρe + p.p) / ᶜρ
    @. p.Π = exner_from_p(p.p)
    @. p.θ = p.T_field / p.Π  # θ_v = θ for dry air

    # =====================================================================
    # ∂K/∂u₃ (ClimaAtmos form with horizontal velocity coupling)
    # Matches manual_sparse_jacobian.jl lines 491-493
    # ∂ᶜK/∂ᶠu₃ = ᶜinterp_matrix ⋅ diag(CT3(ᶠu₃)') + diag(CT3(ᶜuₕ)') ⋅ ᶜinterp_matrix
    # The CT3(ᶜuₕ) term couples horizontal velocity to KE through g³¹
    # (vanishes on flat Cartesian plane, but included for ClimaAtmos fidelity)
    # =====================================================================
    ∂ᶜK∂ᶠu₃ = @. DiagonalMatrixRow(adjoint(CT3(ᶜinterp(ᶠu₃)))) *
        ᶜinterp_matrix() +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) * ᶜinterp_matrix()

    # --- Block (c.ρ, f.u₃): mass block ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 506-507
    # Uses ᶜadvdivᵥ_matrix with J-factors (ClimaAtmos form)
    @. ∂ᶜρₜ∂ᶠu₃ =
        -(ᶜadvdivᵥ_matrix()) *
        DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * g³³(ᶠgⁱʲ))

    # --- Block (c.ρe, f.u₃): energy block ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 522-524
    @. ∂ᶜρeₜ∂ᶠu₃ =
        -(ᶜadvdivᵥ_matrix()) *
        DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠinterp(p.h_tot) * g³³(ᶠgⁱʲ))

    # --- Block (f.u₃, c.ρ): dual PGF ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 539-547
    # Term 1: (-κ/ᶠinterp(ρ)) * ᶠgradᵥ(κ*(T₀cp - K - Φ) * δρ)
    # Term 2: Exner/θ_v gravity correction (unique to ClimaAtmos)
    #   For dry air: κ = R_d/C_v, (R_d - κ·cv) = 0, θ_v = θ
    ᶜΦ = p.ᶜΦ
    @. ∂ᶠu₃ₜ∂ᶜρ =
        -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
        ᶠgradᵥ_matrix() *
        DiagonalMatrixRow(κ_d * (T_0 * C_p - p.K - ᶜΦ)) +
        DiagonalMatrixRow(C_p * ᶠinterp(p.θ) * ᶠgradᵥ(p.Π) / ᶠinterp(ᶜρ)) *
        ᶠinterp_matrix()

    # --- Block (f.u₃, c.ρe): pressure gradient ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl line 548
    # (-κ/ᶠinterp(ρ)) * ᶠgradᵥ(κ * δρe)
    @. ∂ᶠu₃ₜ∂ᶜρe =
        -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) * (ᶠgradᵥ_matrix() * κ_d)

    # --- Block (f.u₃, c.uₕ): NEW — KE coupling through horizontal velocity ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 590-591
    # ᶠp_grad_matrix ⋅ diag(-κ·ρ) ⋅ ∂ᶜK/∂ᶜuₕ
    # where ∂ᶜK/∂ᶜuₕ = diag(CT1(ᶜuₕ)')  [CT12 in 3D]
    # Combine the scalar diags: (-1/ᶠinterp(ρ)) * (-κ*ρ) = κ*ρ/ᶠinterp(ρ)
    # Then scale the center diagonal by CT1(uₕ)' for the adjoint
    @. ∂ᶠu₃ₜ∂ᶜuₕ =
        DiagonalMatrixRow(κ_d / ᶠinterp(ᶜρ)) *
        ᶠgradᵥ_matrix() *
        DiagonalMatrixRow(ᶜρ * adjoint(CT1(ᶜuₕ)))

    # --- Block (f.u₃, f.u₃): KE coupling with full ∂K/∂u₃ ---
    # Matches ClimaAtmos manual_sparse_jacobian.jl lines 592-594
    # ᶠp_grad_matrix ⋅ diag(-κ·ρ) ⋅ ∂ᶜK/∂ᶠu₃
    # Combine the scalar diags: (-1/ᶠinterp(ρ))*(-κ*ρ) = κ*ρ/ᶠinterp(ρ)
    @. ∂ᶠu₃ₜ∂ᶠu₃ =
        -(
            DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(-(ᶜρ * κ_d)) + ᶠgradᵥ_matrix()
        ) * ∂ᶜK∂ᶠu₃

    # --- Assemble residual Jacobian ---
    I = one(∂R∂Y)
    if transform
        @. ∂R∂Y = I / FT(δtγ) - ∂Yₜ∂Y
    else
        @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
    end
end

# ---------------------------------------------------------------------------
# Override CTS.allocate_cache for Krylov with FieldVector prototype
# ---------------------------------------------------------------------------
import Krylov: KrylovConstructor

function CTS.allocate_cache(alg::CTS.KrylovMethod, x_prototype::Fields.FieldVector)
    (; jacobian_free_jvp, forcing_term, kwargs, debugger) = alg
    type = CTS.solver_type(alg)
    kc = KrylovConstructor(similar(x_prototype))
    return (;
        jacobian_free_jvp_cache = isnothing(jacobian_free_jvp) ? nothing :
                                  CTS.allocate_cache(jacobian_free_jvp, x_prototype),
        forcing_term_cache = CTS.allocate_cache(forcing_term, x_prototype),
        solver = type(kc; kwargs...),
        debugger_cache = isnothing(debugger) ? nothing :
                         CTS.allocate_cache(debugger, x_prototype),
    )
end

# ---------------------------------------------------------------------------
# ODE problem setup with JFNK (ClimaAtmos-matching config)
# ---------------------------------------------------------------------------
jac = ImplicitEquationJacobian(Y, false)

if get(ENV, "SOLVER", "jfnk") == "direct"
    newtons_method = CTS.NewtonsMethod(; max_iters = 10)
else
    # Matching ClimaAtmos: newton_rtol=1e-4, krylov_memory=30
    newton_rtol = parse(Float64, get(ENV, "NEWTON_RTOL", "1e-4"))
    newtons_method = CTS.NewtonsMethod(;
        max_iters = parse(Int, get(ENV, "MAX_NEWTON_ITERS", "10")),
        krylov_method = CTS.KrylovMethod(;
            jacobian_free_jvp = CTS.ForwardDiffJVP(;
                step_adjustment = FT(1),
            ),
            forcing_term = CTS.EisenstatWalkerForcing(;
                initial_rtol = FT(0.1),
                γ = FT(0.9),
                α = FT(2),
            ),
            args = (),
            kwargs = (; memory = 30),
        ),
        convergence_checker = CTS.ConvergenceChecker(;
            norm_condition = CTS.MaximumRelativeError(FT(newton_rtol)),
        ),
    )
end

# ARS222 matching ClimaAtmos config (vs ARS233 in simple version)
ode_algo_name = get(ENV, "ODE_ALGO", "ARS222")
ode_algo_type = if ode_algo_name == "ARS222"
    CTS.ARS222()
elseif ode_algo_name == "ARS233"
    CTS.ARS233()
elseif ode_algo_name == "ARS343"
    CTS.ARS343()
else
    error("Unknown ODE algorithm: $ode_algo_name")
end

ode_algo = CTS.IMEXAlgorithm(ode_algo_type, newtons_method)

T_imp! = SciMLBase.ODEFunction(rhs!; jac_prototype = jac, Wfact = wfact!)

Δt = parse(Float64, get(ENV, "DT", "3.0"))
t_end = parse(Float64, get(ENV, "T_END", "900.0"))

problem = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(;
        T_imp! = T_imp!,
        dss! = (u, p, t) -> begin
            Spaces.weighted_dss!(u.c)
            Spaces.weighted_dss!(u.f)
        end,
    ),
    Y,
    (0.0, t_end),
    rhs_cache,
)

integrator = SciMLBase.init(
    problem,
    ode_algo;
    dt = Δt,
    saveat = t_end <= 100 ? collect(range(0.0, t_end, step = max(Δt, t_end / 20))) :
                            collect(0.0:10.0:t_end),
    adaptive = false,
    progress = true,
    progress_steps = 1,
)

if haskey(ENV, "CI_PERF_SKIP_RUN")
    throw(:exit_profile)
end

println("=== ClimaAtmos-matching density current (fully implicit) ===")
println("ODE algorithm: $ode_algo_name")
println("dt = $Δt s, t_end = $t_end s")
println("Newton rtol = $(get(ENV, "NEWTON_RTOL", "1e-4"))")
println("Solver: $(get(ENV, "SOLVER", "jfnk"))")

sol = @timev SciMLBase.solve!(integrator)

# Check for NaNs
for (i, u) in enumerate(sol.u)
    has_nan = any(isnan, parent(u.c.ρ)) || any(isnan, parent(u.c.ρe)) ||
              any(isnan, parent(u.f.u₃))
    if has_nan
        println("NaN detected at save index $i, t=$(sol.t[i])")
        println("  ρ NaN: ", any(isnan, parent(u.c.ρ)))
        println("  ρe NaN: ", any(isnan, parent(u.c.ρe)))
        println("  u₃ NaN: ", any(isnan, parent(u.f.u₃)))
        break
    end
end
first_u = sol.u[1]
last_u = sol.u[end]
println("t=0: min/max ρ = ", extrema(parent(first_u.c.ρ)),
    ", min/max ρe = ", extrema(parent(first_u.c.ρe)))
println("t=end: min/max ρ = ", extrema(parent(last_u.c.ρ)),
    ", min/max ρe = ", extrema(parent(last_u.c.ρe)))

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "density_current_2d_rhoe_vi_implicit_full"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρe ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(If2c.(u.f.u₃))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.uₕ)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

# Potential temperature
anim = Plots.@animate for u in sol.u
    ρ = u.c.ρ
    ρe = u.c.ρe
    w_c = If2c.(u.f.u₃)
    uₕ_loc = u.c.uₕ
    K = @. (norm(uₕ_loc)^2 + norm(w_c)^2) / 2
    p_loc = @. pressure_from_ρe(ρ, ρe, K, coords.z)
    T_loc = @. (ρe / ρ - Φ(coords.z) - K) / C_v + T_0
    π_exn = @. (p_loc / MSLP)^(R_d / C_p)
    θ_loc = @. T_loc / π_exn
    Plots.plot(θ_loc, clim = (285, 300))
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

Es = [sum(u.c.ρe) for u in sol.u]
Mass = [sum(u.c.ρ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy_cons.png"),
)
Plots.png(
    Plots.plot((Mass .- mass_0) ./ mass_0),
    joinpath(path, "mass_cons.png"),
)
