# The staggered nonhydrostatic model shared by the `hybrid/` cases: density and
# total energy at cell centers, horizontal momentum at centers and vertical
# momentum at faces. Defines the explicit and implicit (vertical acoustic)
# tendencies and the Jacobian; the case file defines the constants below first.
#
# The state selects the form: velocities `uₕ`, `w` (vector-invariant; CG, or DG
# via `dg_vector_invariant_tendency.jl`) or momenta `ρuₕ`, `ρw` (flux form; DG
# via `dg_tendency.jl`).
using LinearAlgebra: ×, norm, norm_sqr, dot
using ClimaCore: Operators, Fields

include("implicit_equation_jacobian.jl")
include("hyperdiffusion.jl")

# Constants required before `include("staggered_nonhydrostatic_model.jl")`
# const FT = ?    # floating-point type
# const p_0 = ?   # reference pressure
# const R_d = ?   # dry specific gas constant
# const κ = ?     # kappa
# const T_tri = ? # triple point temperature
# const grav = ?  # gravitational acceleration
# const Ω = ?     # planet's rotation rate (only required if space is spherical)
# const f = ?     # Coriolis frequency (only required if space is flat)

# To add additional terms to the explicit part of the tendency, define new
# methods for `additional_cache` and `additional_tendency!`.

const cp_d = R_d / κ     # heat capacity at constant pressure
const cv_d = cp_d - R_d  # heat capacity at constant volume
const γ = cp_d / cv_d    # heat capacity ratio

const C3 = Geometry.Covariant3Vector
const C12 = Geometry.Covariant12Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT3 = Geometry.Contravariant3Vector
const CT12 = Geometry.Contravariant12Vector

const divₕ = Operators.Divergence()
const split_divₕ = Operators.SplitDivergence()
const wdivₕ = Operators.Divergence{Operators.WeakForm}()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.Gradient{Operators.WeakForm}()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.Curl{Operators.WeakForm}()

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C(
    top = Operators.SetValue(CT3(FT(0))),
    bottom = Operators.SetValue(CT3(FT(0))),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(FT(0))),
    top = Operators.SetGradient(C3(FT(0))),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(FT(0), FT(0))),
    top = Operators.SetCurl(CT12(FT(0), FT(0))),
)
const ᶠupwind_product1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind_product3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.Extrapolate(1),
    top = Operators.Extrapolate(1),
)

const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠupwind_product1_matrix = MatrixFields.operator_matrix(ᶠupwind_product1)
const ᶠupwind_product3_matrix = MatrixFields.operator_matrix(ᶠupwind_product3)

const ᶠno_flux = Operators.SetBoundaryOperator(
    top = Operators.SetValue(CT3(FT(0))),
    bottom = Operators.SetValue(CT3(FT(0))),
)
const ᶠno_flux_row1 = Operators.SetBoundaryOperator(
    top = Operators.SetValue(zero(BidiagonalMatrixRow{CT3{FT}})),
    bottom = Operators.SetValue(zero(BidiagonalMatrixRow{CT3{FT}})),
)
const ᶠno_flux_row3 = Operators.SetBoundaryOperator(
    top = Operators.SetValue(zero(QuaddiagonalMatrixRow{CT3{FT}})),
    bottom = Operators.SetValue(zero(QuaddiagonalMatrixRow{CT3{FT}})),
)

pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)

##
## Reference state of the Exner pressure gradient (DG forms)
##
# Dry reference state, a function of pressure: temperature
# `T_r = T_min + (T_sfc - T_min) Π^7`, potential temperature `θ_r = T_r / Π`,
# and the geopotential `Φ_r` that holds it in hydrostatic balance,
# `c_pd θ_r ∇Π + ∇Φ_r = 0`. The pressure gradient plus gravity is then
# `c_pd θ′ ∇Π + ∇(Φ - Φ_r)` with `θ′ = θ - θ_r`: only the departure from the
# reference is discretized.
const T_sfc_ref = FT(288)
const T_min_ref = FT(215)
const T_ref_exponent = 7
const p_msl = FT(101325)
exner(p) = (p / p_0)^κ
temperature_ref(p) = T_min_ref + (T_sfc_ref - T_min_ref) * exner(p)^T_ref_exponent
theta_ref(p) = temperature_ref(p) / exner(p)
phi_ref(p) =
    -cp_d * (
        T_min_ref * log(exner(p)) +
        (T_sfc_ref - T_min_ref) / T_ref_exponent * (exner(p)^T_ref_exponent - 1)
    )
# Dry potential temperature departure from the reference, `p = ρ R_d T`.
theta_prime(ρ, p) = p / (ρ * R_d) / exner(p) - theta_ref(p)
# Reference pressure at geopotential `Φ` (fixed-point inversion of `Φ_r`).
function pressure_ref(Φ)
    p = p_msl * exp(-Φ / (R_d * T_sfc_ref))
    for _ in 1:8
        p *= exp((phi_ref(p) - Φ) / (R_d * temperature_ref(p)))
    end
    return p
end

##
## The horizontal momentum variable
##
# `uₕ`, `w` (vector-invariant) or `ρuₕ`, `ρw` (flux form), read off the state.

struct VectorInvariantForm end
struct FluxForm end
momentum_form(Yc) =
    hasfield(eltype(Yc), :ρuₕ) ? FluxForm() : VectorInvariantForm()
horizontal_momentum(Yc) = hasfield(eltype(Yc), :ρuₕ) ? Yc.ρuₕ : Yc.uₕ
vertical_momentum(Yf) = hasfield(eltype(Yf), :ρw) ? Yf.ρw : Yf.w

# Flux-form velocities, into cache scratch.
function flux_form_velocities!(Y, p)
    @. p.ᶜuₕ = Y.c.ρuₕ / Y.c.ρ
    @. p.ᶠw = Y.f.ρw / ᶠinterp(Y.c.ρ)
    return p.ᶜuₕ, p.ᶠw
end

include("dg_tendency.jl")
include("dg_vector_invariant_tendency.jl")

get_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, dt, upwinding_mode) = merge(
    default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode),
    additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt),
)

function default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode)
    ᶜcoord = ᶜlocal_geometry.coordinates
    ᶜf_coriolis = if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        @. 2 * Ω * sind(ᶜcoord.lat)
    else
        map(_ -> f, ᶜlocal_geometry)
    end
    ᶜf = @. CT3(Geometry.WVector(ᶜf_coriolis))
    ᶠupwind_product, ᶠupwind_product_matrix, ᶠno_flux_row =
        if upwinding_mode == :first_order
            ᶠupwind_product1, ᶠupwind_product1_matrix, ᶠno_flux_row1
        elseif upwinding_mode == :third_order
            ᶠupwind_product3, ᶠupwind_product3_matrix, ᶠno_flux_row3
        else
            nothing, nothing, nothing
        end
    return (;
        dg_cache(ᶜlocal_geometry, ᶠlocal_geometry, ᶜf_coriolis, Y)...,
        ᶜuvw = similar(ᶜlocal_geometry, C123{FT}),
        ᶜK = similar(ᶜlocal_geometry, FT),
        ᶜΦ = grav .* ᶜcoord.z,
        ᶜp = similar(ᶜlocal_geometry, FT),
        ᶜω³ = similar(ᶜlocal_geometry, CT3{FT}),
        ᶠω¹² = similar(ᶠlocal_geometry, CT12{FT}),
        ᶠu¹² = similar(ᶠlocal_geometry, CT12{FT}),
        ᶠu³ = similar(ᶠlocal_geometry, CT3{FT}),
        ᶜf,
        ∂ᶜK∂ᶠw = similar(
            ᶜlocal_geometry,
            BidiagonalMatrixRow{typeof(CT3(FT(0))')},
        ),
        ᶠupwind_product,
        ᶠupwind_product_matrix,
        ᶠno_flux_row,
        ghost_buffer = (
            c = Spaces.create_dss_buffer(Y.c),
            f = Spaces.create_dss_buffer(Y.f),
            χ = Spaces.create_dss_buffer(Y.c.ρ), # for hyperdiffusion
            χw = Spaces.create_dss_buffer(
                vertical_momentum(Y.f).components.data.:1,
            ), # for hyperdiffusion
            χuₕ = Spaces.create_dss_buffer(horizontal_momentum(Y.c)), # for hyperdiffusion
        ),
    )
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)

implicit_tendency!(Yₜ, Y, p, t) =
    implicit_tendency!(momentum_form(Y.c), Yₜ, Y, p, t)

function implicit_tendency!(::VectorInvariantForm, Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ᶠupwind_product) = p

    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

    @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw))

    ᶜρe = Y.c.ρe
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
    if isnothing(ᶠupwind_product)
        @. Yₜ.c.ρe = -(ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw))
    else
        @. Yₜ.c.ρe = -(ᶜdivᵥ(
            ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / Y.c.ρ),
        ))
    end

    Yₜ.c.uₕ .= (zero(eltype(Yₜ.c.uₕ)),)

    if Spaces.discretization(axes(Y.c)) isa Grids.DG
        # Exner form
        @. Yₜ.f.w = -(
            ᶠgradᵥ(ᶜΦ) - ᶠgradᵥ(phi_ref(ᶜp)) +
            cp_d * ᶠinterp(theta_prime(ᶜρ, ᶜp)) * ᶠgradᵥ(exner(ᶜp)) +
            ᶠgradᵥ(ᶜK)
        )
    else
        @. Yₜ.f.w = -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(ᶜK + ᶜΦ))
    end

    return Yₜ
end

# Flux-form vertical acoustics: mass and energy fluxes linear in `ρw`, and the
# Exner pressure gradient plus gravity.
function implicit_tendency!(::FluxForm, Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶠρw = Y.f.ρw
    (; ᶜK, ᶜΦ, ᶜp) = p
    ᶜuₕ, ᶠw = flux_form_velocities!(Y, p)
    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
    @. ᶜp = pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)

    @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠρw))
    @. Yₜ.c.ρe = -(ᶜdivᵥ(ᶠρw * ᶠinterp((Y.c.ρe + ᶜp) / ᶜρ)))
    Yₜ.c.ρuₕ .= (zero(eltype(Yₜ.c.ρuₕ)),)
    # Exner form, times the face density
    @. Yₜ.f.ρw = -(
        ᶠinterp(ᶜρ) * (
            ᶠgradᵥ(ᶜΦ) - ᶠgradᵥ(phi_ref(ᶜp)) +
            cp_d * ᶠinterp(theta_prime(ᶜρ, ᶜp)) * ᶠgradᵥ(exner(ᶜp))
        )
    )
    return Yₜ
end

function remaining_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    default_remaining_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    Spaces.weighted_dss_start!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_start!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_internal!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_internal!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_ghost!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_ghost!(Yₜ.f, p.ghost_buffer.f)
    return Yₜ
end

# Explicit tendency by discretization and form.
default_remaining_tendency!(Yₜ, Y, p, t) = default_remaining_tendency!(
    Spaces.discretization(axes(Y.c)),
    momentum_form(Y.c),
    Yₜ,
    Y,
    p,
    t,
)

function default_remaining_tendency!(
    ::Grids.CG,
    ::VectorInvariantForm,
    Yₜ,
    Y,
    p,
    t,
)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶠu³, ᶜf) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜK = norm_sqr(ᶜuvw) / 2

    # Mass conservation
    @. Yₜ.c.ρ -= split_divₕ(ᶜρ * ᶜuvw, 1)
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))

    # Energy conservation

    ᶜρe = Y.c.ρe
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
    @. Yₜ.c.ρe -= split_divₕ(ᶜρ * ᶜuvw, (ᶜρe + ᶜp) / ᶜρ)
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # Momentum conservation

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= (zero(eltype(ᶜω³)),)
        @. ᶠω¹² = CT12(curlₕ(ᶠw))
    end
    @. ᶠω¹² += ᶠcurlᵥ(ᶜuₕ)

    # TODO: Modify to account for topography
    @. ᶠu¹² = CT12(ᶠinterp(ᶜuₕ))
    @. ᶠu³ = CT3(ᶠw)

    @. Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × CT12(ᶜuₕ)
    if point_type <: Geometry.Abstract3DPoint
        @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²
end

additional_tendency!(Yₜ, Y, p, t) = nothing

implicit_equation_jacobian!(j, Y, p, δtγ, t) =
    implicit_equation_jacobian!(momentum_form(Y.c), j, Y, p, δtγ, t)

# Flux-form Jacobian: `ᶜK` and `h_tot` frozen, dry `∂p/∂ρe` and `∂p/∂ρ`; the
# Exner gradient is linearized as the equivalent `∂ᵥp + ᶠinterp(ρ) ∂ᵥΦ`.
function implicit_equation_jacobian!(::FluxForm, j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y) = j
    ᶜρ = Y.c.ρ
    (; ᶜK, ᶜΦ, ᶜp) = p
    ᶜuₕ, ᶠw = flux_form_velocities!(Y, p)
    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
    @. ᶜp = pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)

    ᶜρ_name = @name(c.ρ)
    ᶜ𝔼_name = @name(c.ρe)
    ᶠ𝕄_name = @name(f.ρw)
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )

    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜ𝔼_name]
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶠ𝕄_name]

    # ᶜρₜ = -ᶜdivᵥ(ᶠρw), ᶜρeₜ = -ᶜdivᵥ(ᶠρw * ᶠinterp(ᶜh_tot))
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))
    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
        -(ᶜdivᵥ_matrix()) *
        DiagonalMatrixRow(ᶠinterp((Y.c.ρe + ᶜp) / ᶜρ) * g³³(ᶠgⁱʲ))
    # ᶠρwₜ = -ᶠgradᵥ(ᶜp) - ᶠinterp(ᶜρ) * ᶠgradᵥ(ᶜΦ), with
    # ∂p/∂ρe = R_d / cv_d and ∂p/∂ρ = R_d * (T_tri - (ᶜK + ᶜΦ) / cv_d)
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = -(ᶠgradᵥ_matrix() * R_d / cv_d)
    @. ∂ᶠ𝕄ₜ∂ᶜρ =
        -(ᶠgradᵥ_matrix()) *
        DiagonalMatrixRow(R_d * (T_tri - (ᶜK + ᶜΦ) / cv_d)) -
        DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()
    # `ρw` enters its own tendency only through the explicit advection.
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 .= (zero(eltype(∂ᶠ𝕄ₜ∂ᶠ𝕄)),)

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end

function implicit_equation_jacobian!(::VectorInvariantForm, j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, flags) = j
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ∂ᶜK∂ᶠw) = p
    (; ᶠupwind_product, ᶠupwind_product_matrix, ᶠno_flux_row) = p

    ᶜρ_name = @name(c.ρ)
    ᶜ𝔼_name = @name(c.ρe)
    ᶠ𝕄_name = @name(f.w)
    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜ𝔼_name]
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶠ𝕄_name]

    ᶠgⁱʲ = Fields.local_geometry_field(ᶠw).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )
    # If ∂(ᶜχ)/∂(ᶠw) = 0, then
    # ∂(ᶠupwind_product(ᶠw, ᶜχ))/∂(ᶠw) =
    #     ∂(ᶠupwind_product(ᶠw, ᶜχ))/∂(CT3(ᶠw)) * ∂(CT3(ᶠw))/∂(ᶠw) =
    #     vec_data(ᶠupwind_product(ᶠw + εw, ᶜχ)) / vec_data(CT3(ᶠw + εw)) * ᶠg³³
    # The vec_data function extracts the scalar component of a CT3 vector,
    # allowing us to compute the ratio between parallel or antiparallel vectors.
    # Adding a small increment εw to w allows us to avoid NaNs when w = 0. Since
    # ᶠupwind_product is undefined at the boundaries, we also need to wrap it in
    # a call to ᶠno_flux whenever we compute its derivative.
    vec_data(vector) = vector[1]
    εw = (C3(eps(FT)),)

    # ᶜK =
    #     norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2 =
    #     ACT12(ᶜuₕ) * ᶜuₕ / 2 + ACT3(ᶜinterp(ᶠw)) * ᶜinterp(ᶠw) / 2
    # This discrete derivative maps to how the cell-centered kinetic energy
    # changes with respect to the face-centered vertical velocity, which requires
    # interpolating the velocity to the cell centers before taking the dot product.
    # ∂(ᶜK)/∂(ᶠw) = ACT3(ᶜinterp(ᶠw)) * ᶜinterp_matrix()
    @. ∂ᶜK∂ᶠw = DiagonalMatrixRow(adjoint(CT3(ᶜinterp(ᶠw)))) * ᶜinterp_matrix()

    # ᶜρₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw)
    # ∂(ᶜρₜ)/∂(ᶠw) = -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) * ᶠg³³
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(ᶜρ) * g³³(ᶠgⁱʲ))

    ᶜρe = Y.c.ρe
    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)

    if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
        if isnothing(ᶠupwind_product)
            # ᶜρeₜ = -ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw)
            # ∂(ᶜρeₜ)/∂(ᶠw) =
            #     -ᶜdivᵥ_matrix() * (
            #         ᶠinterp(ᶜρe + ᶜp) * ᶠg³³ +
            #         CT3(ᶠw) * ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw)
            #     )
            # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw) =
            #     ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) * ∂(ᶜp)/∂(ᶠw)
            # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) = ᶠinterp_matrix()
            # ∂(ᶜp)/∂(ᶠw) = ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw)
            # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) * (
                    DiagonalMatrixRow(ᶠinterp(ᶜρe + ᶜp) * g³³(ᶠgⁱʲ)) +
                    DiagonalMatrixRow(CT3(ᶠw)) *
                    ᶠinterp_matrix() *
                    DiagonalMatrixRow(-(ᶜρ * R_d / cv_d)) *
                    ∂ᶜK∂ᶠw
                )
        else
            # ᶜρeₜ =
            #     -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / ᶜρ))
            # ∂(ᶜρeₜ)/∂(ᶠw) =
            #     -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) * (
            #         ∂(ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / ᶜρ))/∂(ᶠw) +
            #         ᶠupwind_product_matrix(ᶠw) * ∂((ᶜρe + ᶜp) / ᶜρ)/∂(ᶠw)
            # ∂((ᶜρe + ᶜp) / ᶜρ)/∂(ᶠw) = 1 / ᶜρ * ∂(ᶜp)/∂(ᶠw)
            # ∂(ᶜp)/∂(ᶠw) = ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw)
            # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) *
                DiagonalMatrixRow(ᶠinterp(ᶜρ)) *
                (
                    DiagonalMatrixRow(
                        vec_data(
                            ᶠno_flux(
                                ᶠupwind_product(ᶠw + εw, (ᶜρe + ᶜp) / ᶜρ),
                            ),
                        ) / vec_data(CT3(ᶠw + εw)) * g³³(ᶠgⁱʲ),
                    ) +
                    ᶠno_flux_row(ᶠupwind_product_matrix(ᶠw)) *
                    (-R_d / cv_d * ∂ᶜK∂ᶠw)
                )
        end
    elseif flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK
        # same as above, but we approximate ∂(ᶜp)/∂(ᶜK) = 0, so that
        # ∂ᶜ𝔼ₜ∂ᶠ𝕄 has 3 diagonals instead of 5
        if isnothing(ᶠupwind_product)
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) *
                DiagonalMatrixRow(ᶠinterp(ᶜρe + ᶜp) * g³³(ᶠgⁱʲ))
        else
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(
                    ᶠinterp(ᶜρ) * vec_data(
                        ᶠno_flux(ᶠupwind_product(ᶠw + εw, (ᶜρe + ᶜp) / ᶜρ)),
                    ) / vec_data(CT3(ᶠw + εw)) * g³³(ᶠgⁱʲ),
                )
        end
    else
        error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact or :no_∂ᶜp∂ᶜK when using ρe")
    end

    # TODO: As an optimization, we can rewrite ∂ᶠ𝕄ₜ∂ᶜ𝔼 as 1 / ᶠinterp(ᶜρ) * M,
    # where M is a constant matrix field. When ∂ᶠ𝕄ₜ∂ᶜρ_mode is set to
    # :hydrostatic_balance, we can also do the same for ∂ᶠ𝕄ₜ∂ᶜρ.
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :exact &&
       flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :hydrostatic_balance
        error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :hydrostatic_balance")
    end
    # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
    # ∂(ᶠwₜ)/∂(ᶜρe) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe)
    # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
    # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe) = ᶠgradᵥ_matrix() * R_d / cv_d
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
        -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) * (ᶠgradᵥ_matrix() * R_d / cv_d)

    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
        # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
        # ∂(ᶠwₜ)/∂(ᶜρ) =
        #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
        #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
        # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) =
        #     ᶠgradᵥ_matrix() * R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri)
        # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
        # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_matrix()
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri)) +
            DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2) * ᶠinterp_matrix()
    elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :hydrostatic_balance
        # same as above, but we assume that ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) =
        # -ᶠgradᵥ(ᶜΦ) and that ᶜK is negligible compared ot ᶜΦ
        @. ∂ᶠ𝕄ₜ∂ᶜρ =
            -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(R_d * (-(ᶜΦ) / cv_d + T_tri)) -
            DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ)) * ᶠinterp_matrix()
    end

    # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
    # ∂(ᶠwₜ)/∂(ᶠw) =
    #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶠw) +
    #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) * ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶠw) =
    #     (
    #         ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) +
    #         ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) * ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶜK)
    #     ) * ∂(ᶜK)/∂(ᶠw)
    # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
    # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) =
    #     ᶜ𝔼_name == :ρe ? ᶠgradᵥ_matrix() * (-ᶜρ * R_d / cv_d) : 0
    # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) = -1
    # ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶜK) = ᶠgradᵥ_matrix()
    @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 =
        -(
            DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(-(ᶜρ * R_d / cv_d)) + ᶠgradᵥ_matrix()
        ) * ∂ᶜK∂ᶠw

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end
