#=
Schar mountain wave test (total energy ρe) with fully implicit timestepping.

Adapted from explicit/topo_schar_nh.jl. Uses the same physics (non-hydrostatic
flow over Schar-type terrain with total energy formulation) but restructured for
fully implicit timestepping via ClimaTimeSteppers JFNK solver.

Reference: Schär et al. (2002), MWR, doi:10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2
Section 3(b)

State vector: Y.c = (ρ, ρe, ρq, ρuₕ), Y.f = (ρw,)
  - ρ:   density (center)
  - ρe:  total energy density (center)
  - ρq:  tracer density (center)
  - ρuₕ: horizontal momentum (center, UVector)
  - ρw:  vertical momentum (face, Covariant3Vector)

Newton's method uses JFNK (Jacobian-Free Newton-Krylov):
  - GMRES (Krylov.jl) resolves horizontal acoustic coupling
  - Vertical-only analytical Jacobian as preconditioner (BlockArrowheadSolve)
  - ForwardDiffJVP for Jacobian-vector products
  - EisenstatWalkerForcing for adaptive Krylov tolerance

Run:
    julia --project=.buildkite examples/hybrid/plane/implicit/topo_schar_nh_rhoe_implicit.jl
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
    Operators,
    Hypsography

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
# Physical constants
# ---------------------------------------------------------------------------
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature

# ---------------------------------------------------------------------------
# Schar mountain topography
# ---------------------------------------------------------------------------
function warp_schar(coord)
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    a = 5000
    λ = 4000
    h₀ = 250.0
    if abs(x) <= a
        h = h₀ * exp(-(x / a)^2) * (cos(π * x / λ))^2
    else
        h = FT(0)
    end
end

# ---------------------------------------------------------------------------
# Domain parameters
# ---------------------------------------------------------------------------
const nx = 32
const nz = 40
const np = 4
const Lx = 120000
const Lz = 25000

# ---------------------------------------------------------------------------
# Function space setup (with terrain-following coordinates)
# ---------------------------------------------------------------------------
function hvspace_2D(
    xlim = (-Lx / 2, Lx / 2),
    zlim = (0, Lz),
    xelem = nx,
    zelem = nz,
    npoly = np,
    warp_fn = warp_schar,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(device, horzmesh)
    quad = Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    z_surface = Geometry.ZPoint.(warp_fn.(Fields.coordinate_field(horzspace)))
    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    hv_center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    return (hv_center_space, hv_face_space)
end

hv_center_space, hv_face_space = hvspace_2D()

Φ(z) = grav * z

# ---------------------------------------------------------------------------
# Thermodynamic functions for total energy formulation
# ---------------------------------------------------------------------------
function pressure_from_ρe(ρ, ρe, K, z)
    e = ρe / ρ
    I = e - Φ(z) - K
    T = I / C_v + T_0
    p = ρ * R_d * T
    return p
end

# Pressure derivatives w.r.t. conservative variables (ρ, ρuₕ, ρw, ρe).
# From Roe fluxes (eq. 5): p = αm(ρe - ½||ρu||²/ρ - ρΦ + ρ cvm T₀)
# where αm = Rm/cvm = R_d/C_v for dry air.
#
# ∂p/∂(ρe) = αm = R_d/C_v                                    [eq. 13e]
const ∂p∂ρe = R_d / C_v

# ∂p/∂ρ at fixed ρe, ρuₕ, ρw:
#   K = ½||ρu||²/ρ² depends on ρ, so ∂K/∂ρ = -2K/ρ
#   ∂p/∂ρ = αm(½||ρu||²/ρ² - Φ + cvm T₀) = αm(K - Φ + Cv T₀) [eq. 13a]
∂p∂ρ(z, K) = R_d / C_v * (K - Φ(z) + C_v * T_0)

# ---------------------------------------------------------------------------
# Initial conditions (stably stratified atmosphere, Schär et al. 2002)
# ---------------------------------------------------------------------------
function init_advection_over_mountain(x, z)
    θ₀ = 280.0
    cp_d = C_p
    cv_d = C_v
    p₀ = MSLP
    g = grav

    𝒩 = 0.01
    θ = θ₀ * exp(𝒩^2 * z / g)
    π_exner = 1 + g^2 / 𝒩^2 / cp_d / θ₀ * (exp(-𝒩^2 * z / g) - 1)
    T = π_exner * θ
    ρ = p₀ / (R_d * θ) * (π_exner)^(cp_d / R_d)
    e = cv_d * (T - T_0) + Φ(z) + 50.0  # +50 = initial kinetic energy (u=10 m/s)
    ρe = ρ * e
    ρq = ρ * 0.0
    return (ρ = ρ, ρe = ρe, ρq = ρq)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

function center_initial_condition(local_geometry)
    (; x, z) = local_geometry.coordinates
    ic = init_advection_over_mountain(x, z)
    ρuₕ = ic.ρ * Geometry.UVector(FT(10))  # u₀ = 10 m/s
    return (; ρ = ic.ρ, ρe = ic.ρe, ρq = ic.ρq, ρuₕ)
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

# Store initial horizontal velocity for Rayleigh sponge relaxation
const u_init = Y.c.ρuₕ ./ Y.c.ρ

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
energy_0 = sum(Y.c.ρe)
mass_0 = sum(Y.c.ρ)

# ---------------------------------------------------------------------------
# Rayleigh sponge layers
# ---------------------------------------------------------------------------
function rayleigh_sponge(
    z;
    z_sponge = 12500.0,
    z_max = 25000.0,
    α = 0.5,
    τ = 0.5,
    γ_s = 2.0,
)
    if z >= z_sponge
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α * sinpi(τ * r)^γ_s
        return β_sponge
    else
        return eltype(z)(0)
    end
end

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
    # Pre-compute bottom-boundary terrain metric for set_bottom_w!
    bottom_lg = Fields.level(
        Fields.local_geometry_field(hv_face_space),
        ClimaCore.Utilities.half,
    )
    bottom_g33 = bottom_lg.gⁱʲ.components.data.:4

    return (;
        uₕ = similar(Y.c.ρuₕ),   # UVector
        p = similar(Y.c.ρ),       # scalar (pressure)
        K = similar(Y.c.ρ),       # scalar (kinetic energy)
        h_tot = similar(Y.c.ρ),   # scalar (total enthalpy)
        w = similar(Y.f.ρw),      # C3
        Yfρ = similar(Y.c.ρ, axes(Y.f.ρw)),  # scalar on face space
        uₕf = similar(Y.c.ρuₕ, axes(Y.f.ρw)), # UVector on face space
        # Constant fields (computed once at init)
        ᶜΦ = Φ.(coords.z),
        ᶜ∂p∂ρe_field = fill!(similar(Y.c.ρ), R_d / C_v),
        # Sponge coefficients (constant, depend only on geometry)
        β_sponge = rayleigh_sponge.(coords.z),
        βf_sponge = rayleigh_sponge.(face_coords.z),
        # Bottom boundary terrain metrics (constant)
        bottom_lg = bottom_lg,
        bottom_g33 = bottom_g33,
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
# Bottom boundary w condition (terrain-following: w_3 = -g^31/g^33 * u_1)
# ---------------------------------------------------------------------------
function set_bottom_w!(fw, cuₕ, cache)
    fuₕ = If.(cuₕ)
    u₁_bc =
        Geometry.contravariant3.(
            Fields.level(fuₕ, ClimaCore.Utilities.half),
            cache.bottom_lg,
        )
    u₃_bc = Geometry.Covariant3Vector.(-1 .* u₁_bc ./ cache.bottom_g33)
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. fw = apply_boundary_w(fw)
end

# ---------------------------------------------------------------------------
# RHS function (flux form with total energy)
# ---------------------------------------------------------------------------
function rhs!(dY, Y, cache, t)
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    ρq = Y.c.ρq
    ρuₕ = Y.c.ρuₕ
    ρw = Y.f.ρw

    dρ = dY.c.ρ
    dρe = dY.c.ρe
    dρq = dY.c.ρq
    dρuₕ = dY.c.ρuₕ
    dρw = dY.f.ρw

    # Pre-allocated temporaries
    (; uₕ, w, p, K, h_tot, Yfρ, uₕf) = cache

    z = coords.z

    @. uₕ = ρuₕ / ρ

    # Set bottom boundary condition on w for terrain-following coords
    set_bottom_w!(ρw, uₕ, cache)
    @. w = ρw / If(ρ)
    @. K = (norm(uₕ)^2 + norm(Ic(w))^2) / 2
    @. p = pressure_from_ρe(ρ, ρe, K, z)
    @. h_tot = (ρe + p) / ρ
    @. Yfρ = If(ρ)

    ### HYPERVISCOSITY
    κ₄ = 2e7 # m^4/s (matching explicit Schar case)
    @. dρe = hwdiv(hgrad(h_tot))
    @. dρuₕ = hwdiv(hgrad(uₕ))
    Spaces.weighted_dss!(dY.c)

    @. dρe = -κ₄ * hwdiv(ρ * hgrad(dρe))
    @. dρuₕ = -κ₄ * hwdiv(ρ * hgrad(dρuₕ))

    # density: ∂ρ/∂t = -∇·(ρu)
    @. dρ = -∂(ρw)
    @. dρ -= hdiv(ρuₕ)
    @. dρ -= vdivf2c(If(ρ .* uₕ))

    # total energy: ∂ρe/∂t = -∇·((ρe + p)u)
    @. dρe += -(∂(ρw * If(h_tot)))
    @. dρe -= hdiv(uₕ * (ρe + p))
    @. dρe -= vdivf2c(If(uₕ .* (ρe + p)))

    # tracer: ∂ρq/∂t = -∇·(ρq u)
    @. dρq = -hdiv(uₕ * ρq)
    @. dρq -= ∂(ρw * If(ρq / ρ))
    @. dρq -= vdivf2c(If(uₕ .* ρq))

    # horizontal momentum: ∂ρuₕ/∂t = -∇·(ρuₕ⊗u) - ∇ₕp
    @. dρuₕ += -uvdivf2c(ρw ⊗ If(uₕ))
    @. dρuₕ -= hdiv(ρuₕ ⊗ uₕ + p * Ih)
    @. dρuₕ -= uvdivf2c(If(ρuₕ ⊗ uₕ))

    # vertical momentum: ∂ρw/∂t = -∇·(ρw⊗u) - ∂p/∂z - ρg
    @. dρw = B(
        -(∂f(p)) - If(ρ) * ∂f(Φ(z)) -
        vvdivc2f(Ic(ρw ⊗ w)),
    )
    @. uₕf = If(ρuₕ / ρ)
    @. dρw -= hdiv(uₕf ⊗ ρw)

    # Rayleigh sponge (relax momentum toward initial state near top)
    # Use pre-computed sponge coefficients from cache (constant fields)
    @. dρuₕ -= ρ * cache.β_sponge * (uₕ - u_init)
    @. dρw -= If(ρ) * cache.βf_sponge * w

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

    # Evaluate kinetic energy at current state for linearized derivatives.
    # Use pre-allocated cache fields (shared with rhs!) to avoid allocations.
    @. p.uₕ = ᶜρuₕ / ᶜρ
    @. p.w = ᶠρw / If(ᶜρ)
    @. p.K = (norm(p.uₕ)^2 + norm(Ic(p.w))^2) / 2

    # Total enthalpy for energy flux Jacobian
    @. p.p = pressure_from_ρe(ᶜρ, ᶜρe, p.K, z)
    @. p.h_tot = (ᶜρe + p.p) / ᶜρ

    # --- Block (c.ρ, f.ρw): ∂ᶜρₜ/∂ᶠρw ---
    @. ∂ᶜρₜ∂ᶠρw = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    # --- Block (c.ρe, f.ρw): ∂ᶜρeₜ/∂ᶠρw ---
    @. ∂ᶜρeₜ∂ᶠρw =
        -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(p.h_tot) * g³³(ᶠgⁱʲ))

    # --- Block (f.ρw, c.ρ): ∂ᶠρwₜ/∂ᶜρ ---
    # Gravity term: -diag(ᶠgradᵥ(Φ)) * ᶠinterp_matrix
    # Pressure term: -ᶠgradᵥ_matrix * diag(∂p/∂ρ)
    ᶜΦ = p.ᶜΦ
    @. ∂ᶠρwₜ∂ᶜρ = -DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()
    @. ∂ᶠρwₜ∂ᶜρ -= (ᶠgradᵥ_matrix()) * DiagonalMatrixRow(∂p∂ρ(z, p.K))

    # --- Block (f.ρw, c.ρe): ∂ᶠρwₜ/∂ᶜρe ---
    # ∂p/∂ρe = R_d/Cv (constant, pre-computed in cache)
    @. ∂ᶠρwₜ∂ᶜρe = -(ᶠgradᵥ_matrix()) * DiagonalMatrixRow(p.ᶜ∂p∂ρe_field)

    # --- Block (f.ρw, f.ρw): ∂ᶠρwₜ/∂ᶠρw ---
    # Set to zero. The vertical advection self-coupling and pressure-K coupling
    # are omitted — the preconditioner doesn't need to be exact.
    #
    # Alternative (may help at large DT by reducing GMRES iterations, but the
    # triple matrix product is expensive per wfact! call):
    #   @. ∂ᶠρwₜ∂ᶠρw =
    #       -(ᶠgradᵥ_matrix()) *
    #       DiagonalMatrixRow(-(ᶜρ * R_d / C_v) * adjoint(CT3(ᶜinterp(p.w)))) *
    #       ᶜinterp_matrix()
    # For rhoe, ∂p/∂K = -ρ R_d/Cv couples pressure to kinetic energy.
    # (Ref: staggered_nonhydrostatic_model.jl, lines 582-588)
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂ᶠρwₜ∂ᶠρw .= Ref(zero(TridiagonalRow_C3xACT3))

    # --- Assemble ∂R∂Y ---
    I = one(∂R∂Y)
    if transform
        @. ∂R∂Y = I / FT(δtγ) - ∂Yₜ∂Y
    else
        @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
    end
end

# ---------------------------------------------------------------------------
# Override CTS.allocate_cache to use Krylov's prototype-based constructor
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
                initial_rtol = FT(0.5),
                γ = FT(1),
                α = FT(2),
            ),
            args = (),
            kwargs = (; memory = 30),
        ),
    )
end

ode_algo = CTS.IMEXAlgorithm(CTS.ARS233(), newtons_method)

T_imp! = SciMLBase.ODEFunction(rhs!; jac_prototype = jac, Wfact = wfact!)

# Default: Δt=50s (explicit CFL ~ 1.5s), t_end=15 hours
Δt = parse(Float64, get(ENV, "DT", "10.0"))
t_end = parse(Float64, get(ENV, "T_END", "54000.0"))

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
    saveat = collect(0.0:500.0:t_end),
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

dir = "schar_rhoe_implicit"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρe ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρq ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    w_c = @. Geometry.Covariant13Vector(If2c(u.f.ρw / If(u.c.ρ)))
    w = @. Geometry.project(Geometry.WAxis(), w_c)
    Plots.plot(w, ylim = (0, 12000), xlim = (-10000, 10000))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.c.ρuₕ ./ u.c.ρ)
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

# Potential temperature: θ = T / π_exn
anim = Plots.@animate for u in sol.u
    ρ = u.c.ρ
    ρe = u.c.ρe
    w_c = If2c.(u.f.ρw) ./ ρ
    uₕ = u.c.ρuₕ ./ ρ
    K = @. (norm(uₕ)^2 + norm(w_c)^2) / 2
    p_field = @. pressure_from_ρe(ρ, ρe, K, coords.z)
    T = @. (ρe / ρ - Φ(coords.z) - K) / C_v + T_0
    π_exn = @. (p_field / MSLP)^(R_d / C_p)
    θ = @. T / π_exn
    Plots.plot(θ)
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

# Conservation diagnostics
Es = [sum(u.c.ρe) for u in sol.u]
Mass = [sum(u.c.ρ) for u in sol.u]

Plots.png(Plots.plot((Es .- energy_0) ./ energy_0), joinpath(path, "energy_cons.png"))
Plots.png(Plots.plot((Mass .- mass_0) ./ mass_0), joinpath(path, "mass_cons.png"))
