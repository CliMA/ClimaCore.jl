#=
Rising bubble 2D (flux form, total energy) with fully implicit timestepping.

Adapted from bubble_2d_invariant_rhoe.jl. Uses the same physics (dry rising
thermal bubble with total energy ρe) but restructured for fully implicit
timestepping via ClimaTimeSteppers JFNK solver.

State vector: Y.c = (ρ, ρe, ρuₕ), Y.f = (ρw,)
  - ρ:   density (center)
  - ρe:  total energy density (center)
  - ρuₕ: horizontal momentum (center, UVector)
  - ρw:  vertical momentum (face, Covariant3Vector)

Newton's method uses JFNK (Jacobian-Free Newton-Krylov):
  - GMRES (Krylov.jl) resolves horizontal acoustic coupling
  - Vertical-only analytical Jacobian as preconditioner (BlockArrowheadSolve)
  - ForwardDiffJVP for Jacobian-vector products
  - EisenstatWalkerForcing for adaptive Krylov tolerance

Run:
    julia --project=.buildkite examples/hybrid/plane/bubble_2d_invariant_rhoe_implicit.jl
=#

using Test
using LinearAlgebra, StaticArrays
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
const C3 = Geometry.Covariant3Vector
const CT3 = Geometry.Contravariant3Vector

# ---------------------------------------------------------------------------
# Function space setup
# ---------------------------------------------------------------------------
function hvspace_2D(
    xlim = (-500, 500),
    zlim = (0, 1000),
    helem = 10,
    velem = 40,
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

hv_center_space, hv_face_space = hvspace_2D((-500, 500), (0, 1000))

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature

Φ(z) = grav * z

# ---------------------------------------------------------------------------
# Thermodynamic functions for total energy formulation
# ---------------------------------------------------------------------------
# Internal energy from total energy: I = e - Φ(z) - K
# Temperature: T = I/Cv + T_0
# Pressure: p = ρ R_d T
function pressure_from_ρe(ρ, ρe, K, z)
    e = ρe / ρ
    I = e - Φ(z) - K
    T = I / C_v + T_0
    p = ρ * R_d * T
    return p
end

# ∂p/∂ρe at fixed ρ, K, z: p = ρ R_d (ρe/(ρ Cv) - Φ/(Cv) - K/Cv + T_0)
# ∂p/∂ρe = R_d / C_v
const ∂p∂ρe = R_d / C_v

# ∂p/∂ρ at fixed ρe, ρuₕ, ρw (conservative variables):
# p = R_d/Cv * (ρe - ρΦ - ½||ρu||²/ρ) + R_d*T_0*ρ
# ∂p/∂ρ = R_d/Cv * (-Φ + ½||ρu||²/ρ²) + R_d*T_0
#        = R_d/Cv * (K - Φ + Cv*T_0)     [Roe fluxes eq. 13a]
∂p∂ρ(z, K) = R_d / C_v * (K - Φ(z) + C_v * T_0)

# Total enthalpy: h_tot = (ρe + p) / ρ
function h_tot_from_ρe(ρ, ρe, K, z)
    p = pressure_from_ρe(ρ, ρe, K, z)
    return (ρe + p) / ρ
end

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
# Reference: dry rising thermal bubble (Straka et al. 1993 style)
function init_dry_rising_bubble_2d(x, z)
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5

    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0

    θ = θ_b + θ_p
    π_exn = 1.0 - grav * z / C_p / θ
    T = π_exn * θ
    p = MSLP * π_exn^(C_p / R_d)
    ρ = p / R_d / T
    e = C_v * (T - T_0) + grav * z  # total specific energy (no kinetic at t=0)
    ρe = ρ * e
    return (ρ = ρ, ρe = ρe)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    ic = init_dry_rising_bubble_2d(x, z)
    ρuₕ = ic.ρ * Geometry.UVector(FT(0))
    return (; ρ = ic.ρ, ρe = ic.ρe, ρuₕ)
end

function face_initial_condition(local_geometry)
    return (; ρw = C3(FT(0)))
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
# Vertical FD operators (for rhs! and Jacobian)
# ---------------------------------------------------------------------------
const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(FT(0))),
    top = Operators.SetGradient(C3(FT(0))),
)

# Operator matrices for the Jacobian
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)

# ---------------------------------------------------------------------------
# Pre-allocated cache for rhs! temporaries
# ---------------------------------------------------------------------------
function build_cache(Y)
    return (;
        uₕ = similar(Y.c.ρuₕ),   # UVector
        p = similar(Y.c.ρ),       # scalar (pressure)
        K = similar(Y.c.ρ),       # scalar (kinetic energy)
        h_tot = similar(Y.c.ρ),   # scalar (total enthalpy)
        w = similar(Y.f.ρw),      # C3
        Yfρ = similar(Y.c.ρ, axes(Y.f.ρw)),  # scalar on face space
        uₕf = similar(Y.c.ρuₕ, axes(Y.f.ρw)), # UVector on face space
    )
end

# Operators (created once, reused)
const hdiv = Operators.Divergence()
const hgrad = Operators.Gradient()
const hwdiv = Operators.WeakDivergence()
const hwgrad = Operators.WeakGradient()

const vdivf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)
const vvdivc2f = Operators.DivergenceC2F(
    bottom = Operators.SetDivergence(C3(FT(0))),
    top = Operators.SetDivergence(C3(FT(0))),
)
const uvdivf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(
        C3(FT(0)) ⊗ Geometry.UVector(0.0),
    ),
    top = Operators.SetValue(C3(FT(0)) ⊗ Geometry.UVector(0.0)),
)
const If = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const Ic = Operators.InterpolateF2C()
const ∂ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)
const ∂f = Operators.GradientC2F()
const ∂c = Operators.GradientF2C()
const B = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)
const Ih = Ref(
    Geometry.Axis2Tensor(
        (Geometry.UAxis(), Geometry.UAxis()),
        @SMatrix [1.0]
    ),
)

# ---------------------------------------------------------------------------
# RHS function (flux form with total energy)
# ---------------------------------------------------------------------------
function rhs!(dY, Y, cache, t)
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    ρuₕ = Y.c.ρuₕ
    ρw = Y.f.ρw

    dρ = dY.c.ρ
    dρe = dY.c.ρe
    dρuₕ = dY.c.ρuₕ
    dρw = dY.f.ρw

    # Pre-allocated temporaries
    (; uₕ, w, p, K, h_tot, Yfρ, uₕf) = cache

    z = coords.z
    @. uₕ = ρuₕ / ρ
    @. w = ρw / If(ρ)
    @. K = (norm(uₕ)^2 + norm(Ic(w))^2) / 2
    @. p = pressure_from_ρe(ρ, ρe, K, z)
    @. h_tot = (ρe + p) / ρ
    @. Yfρ = If(ρ)

    ### HYPERVISCOSITY (matches explicit bubble: energy + horizontal momentum only)
    @. dρe = hwdiv(hgrad(h_tot))
    @. dρuₕ = hwdiv(hgrad(uₕ))
    Spaces.weighted_dss!(dY.c)

    κ₄ = 5.0 # m^4/s
    @. dρe = -κ₄ * hwdiv(ρ * hgrad(dρe))
    @. dρuₕ = -κ₄ * hwdiv(ρ * hgrad(dρuₕ))

    # density: ∂ρ/∂t = -∇·(ρu)
    @. dρ = -∂(ρw)
    @. dρ -= hdiv(ρuₕ)

    # total energy: ∂ρe/∂t = -∇·((ρe + p)u)
    @. dρe += -(∂(ρw * If(h_tot)))
    @. dρe -= hdiv(uₕ * (ρe + p))

    # horizontal momentum: ∂ρuₕ/∂t = -∇·(ρuₕ⊗u) - ∇ₕp
    @. dρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    @. dρuₕ -= hdiv(ρuₕ ⊗ uₕ + p * Ih)

    # vertical momentum: ∂ρw/∂t = -∇·(ρw⊗u) - ∂p/∂z - ρg
    @. dρw = B(
        -(∂f(p)) - If(ρ) * ∂f(Φ(z)) -
        vvdivc2f(Ic(ρw ⊗ w)),
    )
    @. uₕf = If(ρuₕ / ρ)
    @. dρw -= hdiv(uₕf ⊗ ρw)

    ### DIFFUSION
    κ₂ = 1.0 # m^2/s
    # horizontal + vertical diffusion of momentum
    @. dρuₕ += hwdiv(κ₂ * (ρ * hgrad(ρuₕ / ρ)))
    @. dρuₕ += uvdivf2c(κ₂ * (Yfρ * ∂f(ρuₕ / ρ)))
    @. dρw += hwdiv(κ₂ * (Yfρ * hgrad(ρw / Yfρ)))
    @. dρw += vvdivc2f(κ₂ * (ρ * ∂c(ρw / Yfρ)))

    # horizontal + vertical diffusion of total energy
    @. dρe += hwdiv(κ₂ * (ρ * hgrad(h_tot)))
    @. dρe += ∂(κ₂ * (Yfρ * ∂f(h_tot)))

    Spaces.weighted_dss!(dY.c)
    Spaces.weighted_dss!(dY.f)
    return dY
end

# Build cache and verify rhs! works
const rhs_cache = build_cache(Y)
dYdt = similar(Y);
rhs!(dYdt, Y, rhs_cache, 0.0);

# ---------------------------------------------------------------------------
# Implicit Equation Jacobian (vertical-only preconditioner)
# ---------------------------------------------------------------------------
struct ImplicitEquationJacobian{TJ, RJ}
    ∂Yₜ∂Y::TJ
    ∂R∂Y::RJ
    transform::Bool
end

function ImplicitEquationJacobian(Y, transform)
    ᶜρ_name = @name(c.ρ)
    ᶜρe_name = @name(c.ρe)
    ᶠρw_name = @name(f.ρw)

    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}

    ᶜspace = axes(Y.c.ρ)
    ᶠspace = axes(Y.f.ρw)

    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        # ∂ᶜρₜ/∂ᶠρw: from -ᶜdivᵥ(ρw)
        (ᶜρ_name, ᶠρw_name) => Fields.zeros(BidiagonalRow_ACT3, ᶜspace),
        # ∂ᶜρeₜ/∂ᶠρw: from -ᶜdivᵥ(ρw * ᶠinterp(h_tot))
        (ᶜρe_name, ᶠρw_name) => Fields.zeros(BidiagonalRow_ACT3, ᶜspace),
        # ∂ᶠρwₜ/∂ᶜρ: from gravity + pressure gradient
        (ᶠρw_name, ᶜρ_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠρwₜ/∂ᶜρe: from pressure gradient -ᶠgradᵥ(p), ∂p/∂ρe = R_d/Cv
        (ᶠρw_name, ᶜρe_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠρwₜ/∂ᶠρw: set to zero (approximate)
        (ᶠρw_name, ᶠρw_name) => Fields.zeros(TridiagonalRow_C3xACT3, ᶠspace),
    )

    # identity_field_matrix ensures all diagonal blocks exist (including ρuₕ)
    I = MatrixFields.identity_field_matrix(Y)
    δtγ = FT(1)
    ∂R∂Y = transform ? I ./ δtγ .- ∂Yₜ∂Y : δtγ .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜρe_name)

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

# ldiv! for FieldVector (Newton's method from CTS)
ldiv!(
    δY::Fields.FieldVector,
    j::ImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

# ---------------------------------------------------------------------------
# wfact! — update the preconditioner Jacobian
# ---------------------------------------------------------------------------
function wfact!(j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, transform) = j
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜρuₕ = Y.c.ρuₕ
    ᶠρw = Y.f.ρw

    ᶜρ_name = @name(c.ρ)
    ᶜρe_name = @name(c.ρe)
    ᶠρw_name = @name(f.ρw)

    ∂ᶜρₜ∂ᶠρw = ∂Yₜ∂Y[ᶜρ_name, ᶠρw_name]
    ∂ᶜρeₜ∂ᶠρw = ∂Yₜ∂Y[ᶜρe_name, ᶠρw_name]
    ∂ᶠρwₜ∂ᶜρ = ∂Yₜ∂Y[ᶠρw_name, ᶜρ_name]
    ∂ᶠρwₜ∂ᶜρe = ∂Yₜ∂Y[ᶠρw_name, ᶜρe_name]
    ∂ᶠρwₜ∂ᶠρw = ∂Yₜ∂Y[ᶠρw_name, ᶠρw_name]

    # Metric factor: g³³ converts C3 → CT3
    ᶠgⁱʲ = Fields.local_geometry_field(ᶠρw).gⁱʲ
    g³³(gⁱʲ) = Geometry.AxisTensor(
        (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
        Geometry.components(gⁱʲ)[end],
    )

    z = coords.z

    # Compute kinetic energy (frozen for Jacobian linearization)
    ᶜuₕ = @. ᶜρuₕ / ᶜρ
    ᶜw = @. Ic(ᶠρw / If(ᶜρ))
    ᶜK = @. (norm(ᶜuₕ)^2 + norm(ᶜw)^2) / 2

    # Total enthalpy for energy flux Jacobian
    ᶜp = @. pressure_from_ρe(ᶜρ, ᶜρe, ᶜK, z)
    ᶜh_tot = @. (ᶜρe + ᶜp) / ᶜρ

    # --- Block (c.ρ, f.ρw): ∂ᶜρₜ/∂ᶠρw ---
    # ρₜ = -ᶜdivᵥ(ρw)
    @. ∂ᶜρₜ∂ᶠρw = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    # --- Block (c.ρe, f.ρw): ∂ᶜρeₜ/∂ᶠρw ---
    # ρeₜ = -ᶜdivᵥ(ρw * ᶠinterp(h_tot))
    # ∂(ρeₜ)/∂(ρw) = -ᶜdivᵥ_matrix * diag(ᶠinterp(h_tot)) * g³³
    @. ∂ᶜρeₜ∂ᶠρw =
        -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(ᶜh_tot) * g³³(ᶠgⁱʲ))

    # --- Block (f.ρw, c.ρ): ∂ᶠρwₜ/∂ᶜρ ---
    # ρwₜ contains -ᶠgradᵥ(p) - ᶠinterp(ρ)*ᶠgradᵥ(Φ)
    # Gravity term: -diag(ᶠgradᵥ(Φ)) * ᶠinterp_matrix
    # Pressure term: -ᶠgradᵥ_matrix * diag(∂p/∂ρ)
    # Split into two assignments to avoid broadcast type issues
    ᶜΦ = @. Φ(z)
    @. ∂ᶠρwₜ∂ᶜρ = -DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()
    @. ∂ᶠρwₜ∂ᶜρ -= (ᶠgradᵥ_matrix()) * DiagonalMatrixRow(∂p∂ρ(z, ᶜK))

    # --- Block (f.ρw, c.ρe): ∂ᶠρwₜ/∂ᶜρe ---
    # ρwₜ contains -ᶠgradᵥ(p) where ∂p/∂ρe = R_d/Cv (constant)
    # Broadcast the constant into a center-space field
    ᶜ∂p∂ρe = @. FT(∂p∂ρe) + ᶜρe * FT(0)
    @. ∂ᶠρwₜ∂ᶜρe = -(ᶠgradᵥ_matrix()) * DiagonalMatrixRow(ᶜ∂p∂ρe)

    # --- Block (f.ρw, f.ρw): set to zero ---
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂ᶠρwₜ∂ᶠρw .= Ref(zero(TridiagonalRow_C3xACT3))

    # --- Assemble ∂R∂Y = δtγ * ∂Yₜ∂Y - I (or I/δtγ - ∂Yₜ∂Y) ---
    I = one(∂R∂Y)
    if transform
        @. ∂R∂Y = I / FT(δtγ) - ∂Yₜ∂Y
    else
        @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
    end
end

# ---------------------------------------------------------------------------
# Override CTS.allocate_cache to use Krylov's prototype-based constructor
# (FieldVector can't be constructed via S(undef, n), but supports similar())
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
        debugger_cache = isnothing(debugger) ? nothing : CTS.allocate_cache(debugger, x_prototype),
    )
end

# ---------------------------------------------------------------------------
# ODE problem setup with JFNK
# ---------------------------------------------------------------------------
jac = ImplicitEquationJacobian(Y, false)

if get(ENV, "SOLVER", "jfnk") == "direct"
    newtons_method = CTS.NewtonsMethod(; max_iters = 10)
else
    newtons_method = CTS.NewtonsMethod(;
        max_iters = 10,
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
            norm_condition = CTS.MaximumRelativeError(FT(1e-6)),
        ),
    )
end

ode_algo = CTS.IMEXAlgorithm(CTS.ARS233(), newtons_method)

T_imp! = SciMLBase.ODEFunction(rhs!; jac_prototype = jac, Wfact = wfact!)

Δt = parse(Float64, get(ENV, "DT", "0.5"))
t_end = parse(Float64, get(ENV, "T_END", "1200.0"))

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

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev SciMLBase.solve!(integrator)

# Check for NaNs
for (i, u) in enumerate(sol.u)
    has_nan = any(isnan, parent(u.c.ρ)) || any(isnan, parent(u.c.ρe)) || any(isnan, parent(u.f.ρw))
    if has_nan
        println("NaN detected at save index $i, t=$(sol.t[i])")
        println("  ρ NaN: ", any(isnan, parent(u.c.ρ)))
        println("  ρe NaN: ", any(isnan, parent(u.c.ρe)))
        println("  ρw NaN: ", any(isnan, parent(u.f.ρw)))
        break
    end
end
first_u = sol.u[1]
last_u = sol.u[end]
println("t=0: min/max ρ = ", extrema(parent(first_u.c.ρ)), ", min/max ρe = ", extrema(parent(first_u.c.ρe)))
println("t=end: min/max ρ = ", extrema(parent(last_u.c.ρ)), ", min/max ρe = ", extrema(parent(last_u.c.ρe)))

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "bubble_2d_rhoe_implicit"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρe ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(If2c.(u.f.ρw) ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρuₕ ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

# Potential temperature: θ = T / π_exn
#   T = (e - Φ - K) / Cv + T_0,  π_exn = (p / MSLP)^(R_d / C_p)
anim = Plots.@animate for u in sol.u
    ρ = u.c.ρ
    ρe = u.c.ρe
    w_c = If2c.(u.f.ρw) ./ ρ
    uₕ = u.c.ρuₕ ./ ρ
    K = @. (norm(uₕ)^2 + norm(w_c)^2) / 2
    p = @. pressure_from_ρe(ρ, ρe, K, coords.z)
    T = @. (ρe / ρ - Φ(coords.z) - K) / C_v + T_0
    π_exn = @. (p / MSLP)^(R_d / C_p)
    θ = @. T / π_exn
    Plots.plot(θ)
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

Es = [sum(u.c.ρe) for u in sol.u]
Mass = [sum(u.c.ρ) for u in sol.u]

Plots.png(Plots.plot((Es .- energy_0) ./ energy_0), joinpath(path, "energy_cons.png"))
Plots.png(Plots.plot((Mass .- mass_0) ./ mass_0), joinpath(path, "mass_cons.png"))
