#=
Density current 2D (flux form) with fully implicit timestepping.

Uses ClimaTimeSteppers IMEXAlgorithm with ARS233 (implicit-only, T_exp! = nothing).
Note: ARS233 has 2 implicit stages (vs 3 for ARS343), reducing Newton solves per
step by ~33%. This trades temporal accuracy order for reduced per-step cost.
Newton's method uses JFNK (Jacobian-Free Newton-Krylov):
  - GMRES (Krylov.jl) resolves horizontal acoustic coupling
  - Vertical-only analytical Jacobian as preconditioner (BlockArrowheadSolve)
  - ForwardDiffJVP for Jacobian-vector products
  - EisenstatWalkerForcing for adaptive Krylov tolerance

The vertical preconditioner captures acoustic-gravity wave coupling:
  ∂ρₜ/∂ρw, ∂ρθₜ/∂ρw (mass/energy flux), ∂ρwₜ/∂ρ (gravity), ∂ρwₜ/∂ρθ (pressure).
Horizontal coupling (pressure ↔ uₕ) is resolved by GMRES iterations.

Run:
    julia --project=.buildkite examples/hybrid/plane/density_current_2d_flux_form_implicit.jl
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

# Geometry type aliases (matching staggered_nonhydrostatic_model.jl convention)
const C3 = Geometry.Covariant3Vector
const CT3 = Geometry.Contravariant3Vector

# ---------------------------------------------------------------------------
# Function space setup
# ---------------------------------------------------------------------------
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 64,
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

hv_center_space, hv_face_space = hvspace_2D((-25600, 25600), (0, 6400))

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

function pressure(ρθ)
    if ρθ >= 0
        return MSLP * (R_d * ρθ / MSLP)^γ
    else
        return NaN
    end
end

Φ(z) = grav * z

# ∂p/∂ρθ for the Jacobian
∂p∂ρθ(ρθ) = ρθ > 0 ? γ * R_d * (R_d * ρθ / MSLP)^(γ - 1) : FT(0)

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_density_current_2d(x, z)
    x_c = 0.0
    z_c = 3000.0
    r_c = 1.0
    x_r = 4000.0
    z_r = 2000.0
    θ_b = 300.0
    θ_c = -15.0
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    r = sqrt((x - x_c)^2 / x_r^2 + (z - z_c)^2 / z_r^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0

    θ = θ_b + θ_p
    π_exn = 1.0 - Φ(z) / cp_d / θ
    T = π_exn * θ
    p = p_0 * π_exn^(cp_d / R_d)
    ρ = p / R_d / T
    ρθ = ρ * θ
    return (ρ = ρ, ρθ = ρθ)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    ic = init_density_current_2d(x, z)
    ρuₕ = ic.ρ * Geometry.UVector(FT(0))
    return (; ρ = ic.ρ, ρθ = ic.ρθ, ρuₕ)
end

function face_initial_condition(local_geometry)
    return (; ρw = C3(FT(0)))
end

ᶜlocal_geometry = Fields.local_geometry_field(hv_center_space)
ᶠlocal_geometry = Fields.local_geometry_field(hv_face_space)

# Restructured state vector: c = center fields (Field of NamedTuples), f = face fields
# This pattern is required by CTS — Y.c and Y.f must be Fields, not FieldVectors.
Y = Fields.FieldVector(
    c = center_initial_condition.(ᶜlocal_geometry),
    f = face_initial_condition.(ᶠlocal_geometry),
)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
θ_0 = sum(Y.c.ρθ)
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
        # center fields (types must match element types)
        uₕ = similar(Y.c.ρuₕ),   # UVector
        p = similar(Y.c.ρ),       # scalar
        θ = similar(Y.c.ρ),       # scalar
        # face fields
        w = similar(Y.f.ρw),      # C3
        Yfρ = similar(Y.c.ρ, axes(Y.f.ρw)),  # scalar on face space
        uₕf = similar(Y.c.ρuₕ, axes(Y.f.ρw)), # UVector on face space
        # Constant fields (computed once at init)
        ᶜΦ = Φ.(coords.z),
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
# RHS function (adapted for c/f state vector with C3 ρw)
# ---------------------------------------------------------------------------
function rhs!(dY, Y, cache, t)
    ρ = Y.c.ρ
    ρθ = Y.c.ρθ
    ρuₕ = Y.c.ρuₕ
    ρw = Y.f.ρw

    dρ = dY.c.ρ
    dρθ = dY.c.ρθ
    dρuₕ = dY.c.ρuₕ
    dρw = dY.f.ρw

    # Pre-allocated temporaries
    (; uₕ, w, p, θ, Yfρ, uₕf) = cache

    @. uₕ = ρuₕ / ρ
    @. w = ρw / If(ρ)
    @. p = pressure(ρθ)
    @. θ = ρθ / ρ
    @. Yfρ = If(ρ)

    ### HYPERVISCOSITY
    @. dρθ = hwdiv(hgrad(θ))
    @. dρuₕ = hwdiv(hgrad(uₕ))
    @. dρw = hwdiv(hgrad(w))
    Spaces.weighted_dss!(dY.c)
    Spaces.weighted_dss!(dY.f)

    κ₄ = 0.0 # m^4/s
    @. dρθ = -κ₄ * hwdiv(ρ * hgrad(dρθ))
    @. dρuₕ = -κ₄ * hwdiv(ρ * hgrad(dρuₕ))
    @. dρw = -κ₄ * hwdiv(Yfρ * hgrad(dρw))

    # density
    @. dρ = -∂(ρw)
    @. dρ -= hdiv(ρuₕ)

    # potential temperature
    @. dρθ += -(∂(ρw * If(ρθ / ρ)))
    @. dρθ -= hdiv(uₕ * ρθ)

    # horizontal momentum
    @. dρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    @. dρuₕ -= hdiv(ρuₕ ⊗ uₕ + p * Ih)

    # vertical momentum (C3 form)
    z = coords.z
    @. dρw += B(
        -(∂f(p)) - If(ρ) * ∂f(Φ(z)) -
        vvdivc2f(Ic(ρw ⊗ w)),
    )
    @. uₕf = If(ρuₕ / ρ)
    @. dρw -= hdiv(uₕf ⊗ ρw)

    ### DIFFUSION
    κ₂ = 75.0 # m^2/s
    #  1a) horizontal div of horizontal grad of horiz momentum
    @. dρuₕ += hwdiv(κ₂ * (ρ * hgrad(ρuₕ / ρ)))
    #  1b) vertical div of vertical grad of horiz momentum
    @. dρuₕ += uvdivf2c(κ₂ * (Yfρ * ∂f(ρuₕ / ρ)))

    #  1c) horizontal div of horizontal grad of vert momentum
    @. dρw += hwdiv(κ₂ * (Yfρ * hgrad(ρw / Yfρ)))
    #  1d) vertical div of vertical grad of vert momentum
    @. dρw += vvdivc2f(κ₂ * (ρ * ∂c(ρw / Yfρ)))

    #  2a) horizontal div of horizontal grad of potential temperature
    @. dρθ += hwdiv(κ₂ * (ρ * hgrad(ρθ / ρ)))
    #  2b) vertical div of vertical grad of potential temperature
    @. dρθ += ∂(κ₂ * (Yfρ * ∂f(ρθ / ρ)))

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
    ᶜρθ_name = @name(c.ρθ)
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
        # ∂ᶜρθₜ/∂ᶠρw: from -ᶜdivᵥ(ρw * ᶠinterp(θ))
        (ᶜρθ_name, ᶠρw_name) => Fields.zeros(BidiagonalRow_ACT3, ᶜspace),
        # ∂ᶠρwₜ/∂ᶜρ: from gravity term -ᶠinterp(ρ)*ᶠgradᵥ(Φ)
        (ᶠρw_name, ᶜρ_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠρwₜ/∂ᶜρθ: from pressure gradient -ᶠgradᵥ(p)
        (ᶠρw_name, ᶜρθ_name) => Fields.zeros(BidiagonalRow_C3, ᶠspace),
        # ∂ᶠρwₜ/∂ᶠρw: from vertical advection (kinetic energy feedback)
        (ᶠρw_name, ᶠρw_name) => Fields.zeros(TridiagonalRow_C3xACT3, ᶠspace),
    )

    # identity_field_matrix ensures all diagonal blocks exist (including ρuₕ)
    I = MatrixFields.identity_field_matrix(Y)
    δtγ = FT(1)
    ∂R∂Y = transform ? I ./ δtγ .- ∂Yₜ∂Y : δtγ .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜρθ_name)

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

# Note: dense-vector ldiv! removed — Krylov.jl now operates directly on FieldVectors

# ---------------------------------------------------------------------------
# wfact! — update the preconditioner Jacobian
# ---------------------------------------------------------------------------
function wfact!(j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, transform) = j
    ᶜρ = Y.c.ρ
    ᶜρθ = Y.c.ρθ
    ᶠρw = Y.f.ρw

    ᶜρ_name = @name(c.ρ)
    ᶜρθ_name = @name(c.ρθ)
    ᶠρw_name = @name(f.ρw)

    ∂ᶜρₜ∂ᶠρw = ∂Yₜ∂Y[ᶜρ_name, ᶠρw_name]
    ∂ᶜρθₜ∂ᶠρw = ∂Yₜ∂Y[ᶜρθ_name, ᶠρw_name]
    ∂ᶠρwₜ∂ᶜρ = ∂Yₜ∂Y[ᶠρw_name, ᶜρ_name]
    ∂ᶠρwₜ∂ᶜρθ = ∂Yₜ∂Y[ᶠρw_name, ᶜρθ_name]
    ∂ᶠρwₜ∂ᶠρw = ∂Yₜ∂Y[ᶠρw_name, ᶠρw_name]

    # Metric factor: g³³ converts C3 → CT3
    ᶠgⁱʲ = Fields.local_geometry_field(ᶠρw).gⁱʲ
    g³³(gⁱʲ) = Geometry.AxisTensor(
        (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
        Geometry.components(gⁱʲ)[end],
    )

    # --- Block (c.ρ, f.ρw): ∂ᶜρₜ/∂ᶠρw ---
    # ρₜ = -ᶜdivᵥ(ρw)   [linear in ρw]
    # ∂(ρₜ)/∂(ρw) = -ᶜdivᵥ_matrix() * g³³
    @. ∂ᶜρₜ∂ᶠρw = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    # --- Block (c.ρθ, f.ρw): ∂ᶜρθₜ/∂ᶠρw ---
    # ρθₜ = -ᶜdivᵥ(ρw * ᶠinterp(θ))  where θ = ρθ/ρ
    # ∂(ρθₜ)/∂(ρw) = -ᶜdivᵥ_matrix() * diag(ᶠinterp(θ)) * g³³
    @. ∂ᶜρθₜ∂ᶠρw =
        -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(ᶜρθ / ᶜρ) * g³³(ᶠgⁱʲ))

    # --- Block (f.ρw, c.ρ): ∂ᶠρwₜ/∂ᶜρ ---
    # ρwₜ = -ᶠgradᵥ(p) - ᶠinterp(ρ)*ᶠgradᵥ(Φ) - ...
    # Only the gravity term depends on ρ: -ᶠinterp(ρ)*ᶠgradᵥ(Φ)
    # ᶠgradᵥ(Φ) is constant, so:
    # ∂(ρwₜ)/∂(ρ) = -diag(ᶠgradᵥ(Φ)) * ᶠinterp_matrix()
    ᶜΦ = p.ᶜΦ
    @. ∂ᶠρwₜ∂ᶜρ = -DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()

    # --- Block (f.ρw, c.ρθ): ∂ᶠρwₜ/∂ᶜρθ ---
    # ρwₜ contains -ᶠgradᵥ(p) where p = p(ρθ)
    # ∂(ρwₜ)/∂(ρθ) = -ᶠgradᵥ_matrix() * diag(∂p/∂ρθ)
    @. ∂ᶠρwₜ∂ᶜρθ = -(ᶠgradᵥ_matrix()) * DiagonalMatrixRow(∂p∂ρθ(ᶜρθ))

    # --- Block (f.ρw, f.ρw): ∂ᶠρwₜ/∂ᶠρw ---
    # Set to zero. The vertical advection self-coupling through kinetic energy
    # is omitted — the preconditioner doesn't need to be exact.
    #
    # Alternative (may help at large DT by reducing GMRES iterations, but the
    # triple matrix product is expensive per wfact! call):
    #   @. p.w = ᶠρw / If(ᶜρ)
    #   @. ∂ᶠρwₜ∂ᶠρw =
    #       -(ᶠgradᵥ_matrix()) *
    #       DiagonalMatrixRow(adjoint(CT3(ᶜinterp(p.w)))) *
    #       ᶜinterp_matrix()
    # (Ref: staggered_nonhydrostatic_model.jl, lines 580-581)
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

# Newton solver: JFNK (with Krylov GMRES) or direct (vertical preconditioner only).
# Set SOLVER=direct for direct Newton, SOLVER=jfnk for JFNK (default).
if get(ENV, "SOLVER", "jfnk") == "direct"
    # This is not practical
    newtons_method = CTS.NewtonsMethod(; max_iters = 10)
else
    # JFNK: GMRES resolves horizontal coupling; vertical preconditioner
    # accelerates convergence of fast vertical acoustic-gravity modes.
    newtons_method = CTS.NewtonsMethod(;
        max_iters = 10,
        krylov_method = CTS.KrylovMethod(;
            jacobian_free_jvp = CTS.ForwardDiffJVP(;
                step_adjustment = FT(1),
            ),
            forcing_term = CTS.EisenstatWalkerForcing(;
                initial_rtol = FT(0.5),
                γ = FT(1),
                α = FT(2),
            ),
            args = (),
            kwargs = (; memory = 30),
        ),
    )
end

# ARS233: 2 implicit stages (vs 3 for ARS343) since it requires fewer Newton solves per step.
# For higher-order temporal accuracy, use CTS.ARS343().
ode_algo = CTS.IMEXAlgorithm(CTS.ARS233(), newtons_method)

T_imp! = SciMLBase.ODEFunction(rhs!; jac_prototype = jac, Wfact = wfact!)

Δt = parse(Float64, get(ENV, "DT", "25.0"))
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
                            collect(0.0:50.0:t_end),
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
    has_nan = any(isnan, parent(u.c.ρ)) || any(isnan, parent(u.c.ρθ)) || any(isnan, parent(u.f.ρw))
    if has_nan
        println("NaN detected at save index $i, t=$(sol.t[i])")
        println("  ρ NaN: ", any(isnan, parent(u.c.ρ)))
        println("  ρθ NaN: ", any(isnan, parent(u.c.ρθ)))
        println("  ρw NaN: ", any(isnan, parent(u.f.ρw)))
        break
    end
end
first_u = sol.u[1]
last_u = sol.u[end]
println("t=0: min/max ρ = ", extrema(parent(first_u.c.ρ)), ", min/max ρθ = ", extrema(parent(first_u.c.ρθ)))
println("t=end: min/max ρ = ", extrema(parent(last_u.c.ρ)), ", min/max ρθ = ", extrema(parent(last_u.c.ρθ)))

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dc_fluxform_implicit"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρθ ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(If2c.(u.f.ρw) ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρuₕ ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

θs = [sum(u.c.ρθ) for u in sol.u]
Mass = [sum(u.c.ρ) for u in sol.u]

Plots.png(Plots.plot((θs .- θ_0) ./ θ_0), joinpath(path, "energy.png"))
Plots.png(Plots.plot((Mass .- mass_0) ./ mass_0), joinpath(path, "mass.png"))
