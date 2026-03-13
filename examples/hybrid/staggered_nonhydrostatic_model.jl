using LinearAlgebra: ×, norm, norm_sqr, dot, Adjoint
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
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

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
const ᶜFC = Operators.FluxCorrectionC2C(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠupwind_product1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind_product3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
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

pressure_ρθ(ρθ) = p_0 * (ρθ * R_d / p_0)^γ
pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)
pressure_ρe_int(ρe_int, ρ) = R_d * (ρe_int / cv_d + ρ * T_tri)

get_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, dt, upwinding_mode) = merge(
    default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode),
    additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt),
)

function default_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, upwinding_mode)
    ᶜcoord = ᶜlocal_geometry.coordinates
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
    else
        ᶜf = map(_ -> f, ᶜlocal_geometry)
    end
    ᶜf = @. CT3(Geometry.WVector(ᶜf))
    ᶠupwind_product, ᶠupwind_product_matrix, ᶠno_flux_row =
        if upwinding_mode == :first_order
            ᶠupwind_product1, ᶠupwind_product1_matrix, ᶠno_flux_row1
        elseif upwinding_mode == :third_order
            ᶠupwind_product3, ᶠupwind_product3_matrix, ᶠno_flux_row3
        else
            nothing, nothing, nothing
        end
    return (;
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
            BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}},
        ),
        ᶠupwind_product,
        ᶠupwind_product_matrix,
        ᶠno_flux_row,
        ghost_buffer = (
            c = Spaces.create_dss_buffer(Y.c),
            f = Spaces.create_dss_buffer(Y.f),
            χ = Spaces.create_dss_buffer(Y.c.ρ), # for hyperdiffusion
            χw = Spaces.create_dss_buffer(Y.f.w.components.data.:1), # for hyperdiffusion
            χuₕ = Spaces.create_dss_buffer(Y.c.uₕ), # for hyperdiffusion
        ),
    )
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)

function implicit_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ᶠupwind_product) = p

    @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

    @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw))

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρθ = -(ᶜdivᵥ(ᶠinterp(ᶜρθ) * ᶠw))
        else
            @. Yₜ.c.ρθ =
                -(ᶜdivᵥ(ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, ᶜρθ / Y.c.ρ)))
        end
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe = -(ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw))
        else
            @. Yₜ.c.ρe = -(ᶜdivᵥ(
                ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / Y.c.ρ),
            ))
        end
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw) -
                ᶜinterp(dot(ᶠgradᵥ(ᶜp), CT3(ᶠw)))
            )
            # or, equivalently,
            # Yₜ.c.ρe_int = -(ᶜdivᵥ(ᶠinterp(ᶜρe_int) * ᶠw) + ᶜp * ᶜdivᵥ(ᶠw))
        else
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(
                    ᶠinterp(Y.c.ρ) *
                    ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / Y.c.ρ),
                ) - ᶜinterp(dot(ᶠgradᵥ(ᶜp), CT3(ᶠw)))
            )
        end
    end

    Yₜ.c.uₕ .= (zero(eltype(Yₜ.c.uₕ)),)

    @. Yₜ.f.w = -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(ᶜK + ᶜΦ))

    # TODO: Add flux correction to the Jacobian
    # @. Yₜ.c.ρ += ᶜFC(ᶠw, ᶜρ)
    # if :ρθ in propertynames(Y.c)
    #     @. Yₜ.c.ρθ += ᶜFC(ᶠw, ᶜρθ)
    # elseif :ρe in propertynames(Y.c)
    #     @. Yₜ.c.ρe += ᶜFC(ᶠw, ᶜρe)
    # elseif :ρe_int in propertynames(Y.c)
    #     @. Yₜ.c.ρe_int += ᶜFC(ᶠw, ᶜρe_int)
    # end

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

function default_remaining_tendency!(Yₜ, Y, p, t)
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

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)
        @. Yₜ.c.ρθ -= split_divₕ(ᶜρ * ᶜuvw, ᶜρθ / ᶜρ)
        @. Yₜ.c.ρθ -= ᶜdivᵥ(ᶠinterp(ᶜρθ * ᶜuₕ))
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
        @. Yₜ.c.ρe -= split_divₕ(ᶜρ * ᶜuvw, (ᶜρe + ᶜp) / ᶜρ)
        @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)
        if point_type <: Geometry.Abstract3DPoint
            @. Yₜ.c.ρe_int -=
                split_divₕ(ᶜρ * ᶜuvw, (ᶜρe_int + ᶜp) / ᶜρ) - dot(gradₕ(ᶜp), CT12(ᶜuₕ))
        else
            @. Yₜ.c.ρe_int -=
                split_divₕ(ᶜρ * ᶜuvw, (ᶜρe_int + ᶜp) / ᶜρ) - dot(gradₕ(ᶜp), CT1(ᶜuₕ))
        end
        @. Yₜ.c.ρe_int -= ᶜdivᵥ(ᶠinterp((ᶜρe_int + ᶜp) * ᶜuₕ))
        # or, equivalently,
        # @. Yₜ.c.ρe_int -= divₕ(ᶜρe_int * ᶜuvw) + ᶜp * divₕ(ᶜuvw)
        # @. Yₜ.c.ρe_int -=
        #     ᶜdivᵥ(ᶠinterp(ᶜρe_int * ᶜuₕ)) + ᶜp * ᶜdivᵥ(ᶠinterp(ᶜuₕ))
    end

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

# ==========================================================================
# Fully implicit tendency: all terms in one function for T_imp! with JFNK.
#
# Usage:
#   T_imp! = SciMLBase.ODEFunction(
#       fully_implicit_tendency!;
#       jac_prototype = jac,
#       Wfact = implicit_equation_jacobian!,
#   )
#   prob = SciMLBase.ODEProblem(
#       CTS.ClimaODEFunction(; T_imp!, dss! = ...),
#       Y, tspan, cache,
#   )
#
# The existing implicit_equation_jacobian! serves as the vertical-only
# preconditioner for JFNK (GMRES resolves horizontal coupling).
# ==========================================================================
function fully_implicit_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶠu³, ᶜf) = p
    (; ᶠupwind_product) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜK = norm_sqr(ᶜuvw) / 2

    # === Mass conservation ===
    # Vertical flux
    @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw))
    # Horizontal flux
    @. Yₜ.c.ρ -= split_divₕ(ᶜρ * ᶜuvw, 1)
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))

    # === Energy conservation ===
    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)

        # Vertical flux (with optional upwinding)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρθ = -(ᶜdivᵥ(ᶠinterp(ᶜρθ) * ᶠw))
        else
            @. Yₜ.c.ρθ =
                -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, ᶜρθ / ᶜρ)))
        end
        # Horizontal flux
        @. Yₜ.c.ρθ -= split_divₕ(ᶜρ * ᶜuvw, ᶜρθ / ᶜρ)
        @. Yₜ.c.ρθ -= ᶜdivᵥ(ᶠinterp(ᶜρθ * ᶜuₕ))

    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)

        # Vertical flux (with optional upwinding)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe = -(ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw))
        else
            @. Yₜ.c.ρe = -(ᶜdivᵥ(
                ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / ᶜρ),
            ))
        end
        # Horizontal flux
        @. Yₜ.c.ρe -= split_divₕ(ᶜρ * ᶜuvw, (ᶜρe + ᶜp) / ᶜρ)
        @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)

        # Vertical flux (with pdV work correction, optional upwinding)
        if isnothing(ᶠupwind_product)
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw) -
                ᶜinterp(dot(ᶠgradᵥ(ᶜp), CT3(ᶠw)))
            )
        else
            @. Yₜ.c.ρe_int = -(
                ᶜdivᵥ(
                    ᶠinterp(ᶜρ) *
                    ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / ᶜρ),
                ) - ᶜinterp(dot(ᶠgradᵥ(ᶜp), CT3(ᶠw)))
            )
        end
        # Horizontal flux (with pressure work correction)
        if point_type <: Geometry.Abstract3DPoint
            @. Yₜ.c.ρe_int -=
                split_divₕ(ᶜρ * ᶜuvw, (ᶜρe_int + ᶜp) / ᶜρ) -
                dot(gradₕ(ᶜp), CT12(ᶜuₕ))
        else
            @. Yₜ.c.ρe_int -=
                split_divₕ(ᶜρ * ᶜuvw, (ᶜρe_int + ᶜp) / ᶜρ) -
                dot(gradₕ(ᶜp), CT1(ᶜuₕ))
        end
        @. Yₜ.c.ρe_int -= ᶜdivᵥ(ᶠinterp((ᶜρe_int + ᶜp) * ᶜuₕ))
    end

    # === Momentum conservation ===
    # Vorticity terms
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= (zero(eltype(ᶜω³)),)
        @. ᶠω¹² = CT12(curlₕ(ᶠw))
    end
    @. ᶠω¹² += ᶠcurlᵥ(ᶜuₕ)

    @. ᶠu¹² = CT12(ᶠinterp(ᶜuₕ))
    @. ᶠu³ = CT3(ᶠw)

    # Horizontal momentum: vorticity + Coriolis + pressure gradient + KE gradient
    @. Yₜ.c.uₕ = -(ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × CT12(ᶜuₕ))
    if point_type <: Geometry.Abstract3DPoint
        @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    # Vertical momentum: pressure gradient + gravity + kinetic energy gradient
    @. Yₜ.f.w = -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(ᶜK + ᶜΦ))
    # Horizontal vorticity × velocity (cross-component coupling)
    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²

    # === Hyperdiffusion ===
    if hasproperty(p, :κ₄) && p.κ₄ > 0
        _fully_implicit_hyperdiffusion!(Yₜ, Y, p, t)
    end

    # === Additional tendency (user-defined) ===
    additional_tendency!(Yₜ, Y, p, t)

    # === DSS (required for spectral element consistency) ===
    Spaces.weighted_dss_start!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_start!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_internal!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_internal!(Yₜ.f, p.ghost_buffer.f)
    Spaces.weighted_dss_ghost!(Yₜ.c, p.ghost_buffer.c)
    Spaces.weighted_dss_ghost!(Yₜ.f, p.ghost_buffer.f)

    return Yₜ
end

# Hyperdiffusion for fully_implicit_tendency! (inlined to avoid separate file)
function _fully_implicit_hyperdiffusion!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    if use_tempest_mode
        @. ᶜχ = wdivₕ(gradₕ(ᶜρ))
        Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ))
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))
        else
            error("use_tempest_mode must be false when not using ρθ")
        end

        (; ᶠχw_data) = p
        @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
        Spaces.weighted_dss!(ᶠχw_data, ghost_buffer.χ)
        @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))
    else
        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ / ᶜρ))
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe + ᶜp) / ᶜρ))
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρe -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe_int in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ))
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρe_int -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        end
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
end

function implicit_equation_jacobian!(j, Y, p, δtγ, t)
    (; ∂Yₜ∂Y, ∂R∂Y, transform, flags) = j
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ∂ᶜK∂ᶠw) = p
    (; ᶠupwind_product, ᶠupwind_product_matrix, ᶠno_flux_row) = p

    ᶜρ_name = @name(c.ρ)
    ᶜ𝔼_name = if :ρθ in propertynames(Y.c)
        @name(c.ρθ)
    elseif :ρe in propertynames(Y.c)
        @name(c.ρe)
    elseif :ρe_int in propertynames(Y.c)
        @name(c.ρe_int)
    end
    ᶠ𝕄_name = @name(f.w)
    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶜ𝔼_name]
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶠ𝕄_name, ᶠ𝕄_name]

    ᶠgⁱʲ = Fields.local_geometry_field(ᶠw).gⁱʲ
    g³³(gⁱʲ) = Geometry.AxisTensor(
        (Geometry.Contravariant3Axis(), Geometry.Contravariant3Axis()),
        Geometry.components(gⁱʲ)[end],
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
    # ∂(ᶜK)/∂(ᶠw) = ACT3(ᶜinterp(ᶠw)) * ᶜinterp_matrix()
    @. ∂ᶜK∂ᶠw = DiagonalMatrixRow(adjoint(CT3(ᶜinterp(ᶠw)))) * ᶜinterp_matrix()

    # ᶜρₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw)
    # ∂(ᶜρₜ)/∂(ᶠw) = -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) * ᶠg³³
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(ᶜρ) * g³³(ᶠgⁱʲ))

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)

        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρθ")
        end

        if isnothing(ᶠupwind_product)
            # ᶜρθₜ = -ᶜdivᵥ(ᶠinterp(ᶜρθ) * ᶠw)
            # ∂(ᶜρθₜ)/∂(ᶠw) = -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρθ) * ᶠg³³
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(ᶠinterp(ᶜρθ) * g³³(ᶠgⁱʲ))
        else
            # ᶜρθₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, ᶜρθ / ᶜρ))
            # ∂(ᶜρθₜ)/∂(ᶠw) =
            #     -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) *
            #     ∂(ᶠupwind_product(ᶠw, ᶜρθ / ᶜρ))/∂(ᶠw)
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(
                    ᶠinterp(ᶜρ) *
                    vec_data(ᶠno_flux(ᶠupwind_product(ᶠw + εw, ᶜρθ / ᶜρ))) /
                    vec_data(CT3(ᶠw + εw)) * g³³(ᶠgⁱʲ),
                )
        end
    elseif :ρe in propertynames(Y.c)
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
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)

        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρe_int")
        end

        if isnothing(ᶠupwind_product)
            # ᶜρe_intₜ =
            #     -ᶜdivᵥ(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw) +
            #     ᶜinterp(adjoint(ᶠgradᵥ(ᶜp)) * CT3(ᶠw))
            # ∂(ᶜρe_intₜ)/∂(ᶠw) =
            #     -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρe_int + ᶜp) * ᶠg³³ +
            #     ᶜinterp_matrix() * adjoint(ᶠgradᵥ(ᶜp)) * ᶠg³³
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) *
                DiagonalMatrixRow(ᶠinterp(ᶜρe_int + ᶜp) * g³³(ᶠgⁱʲ)) +
                ᶜinterp_matrix() *
                DiagonalMatrixRow(adjoint(ᶠgradᵥ(ᶜp)) * g³³(ᶠgⁱʲ))
        else
            # ᶜρe_intₜ =
            #     -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / ᶜρ)) +
            #     ᶜinterp(adjoint(ᶠgradᵥ(ᶜp)) * CT3(ᶠw))
            # ∂(ᶜρe_intₜ)/∂(ᶠw) =
            #     -ᶜdivᵥ_matrix() * ᶠinterp(ᶜρ) *
            #     ∂(ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / ᶜρ))/∂(ᶠw) +
            #     ᶜinterp_matrix() * adjoint(ᶠgradᵥ(ᶜp)) * ᶠg³³
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(
                    ᶠinterp(ᶜρ) * vec_data(
                        ᶠno_flux(ᶠupwind_product(ᶠw + εw, (ᶜρe_int + ᶜp) / ᶜρ)),
                    ) / vec_data(CT3(ᶠw + εw)) * g³³(ᶠgⁱʲ),
                ) +
                ᶜinterp_matrix() *
                DiagonalMatrixRow(adjoint(ᶠgradᵥ(ᶜp)) * g³³(ᶠgⁱʲ))
        end
    end

    # TODO: As an optimization, we can rewrite ∂ᶠ𝕄ₜ∂ᶜ𝔼 as 1 / ᶠinterp(ᶜρ) * M,
    # where M is a constant matrix field. When ∂ᶠ𝕄ₜ∂ᶜρ_mode is set to
    # :hydrostatic_balance, we can also do the same for ∂ᶠ𝕄ₜ∂ᶜρ.
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :exact &&
       flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :hydrostatic_balance
        error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :hydrostatic_balance")
    end
    if :ρθ in propertynames(Y.c)
        # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
        # ∂(ᶠwₜ)/∂(ᶜρθ) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ)
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
        # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ) =
        #     ᶠgradᵥ_matrix() * γ * R_d * (ᶜρθ * R_d / p_0)^(γ - 1)
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
            -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(γ * R_d * (ᶜρθ * R_d / p_0)^(γ - 1))

        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρ) = ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
            # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_matrix()
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2) * ᶠinterp_matrix()
        elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :hydrostatic_balance
            # same as above, but we assume that ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) =
            # -ᶠgradᵥ(ᶜΦ)
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                -DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ)) * ᶠinterp_matrix()
        end
    elseif :ρe in propertynames(Y.c)
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
    elseif :ρe_int in propertynames(Y.c)
        # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
        # ∂(ᶠwₜ)/∂(ᶜρe_int) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int)
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
        # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int) = ᶠgradᵥ_matrix() * R_d / cv_d
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
            DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) * (ᶠgradᵥ_matrix() * R_d / cv_d)

        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρ) =
            #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
            #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) = ᶠgradᵥ_matrix() * R_d * T_tri
            # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
            # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_matrix()
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                -DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
                (ᶠgradᵥ_matrix() * R_d * T_tri) +
                DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2) * ᶠinterp_matrix()
        elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :hydrostatic_balance
            # same as above, but we assume that ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) =
            # -ᶠgradᵥ(ᶜΦ)
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) *
                (ᶠgradᵥ_matrix() * R_d * T_tri) -
                DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ)) * ᶠinterp_matrix()
        end
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
    if :ρθ in propertynames(Y.c) || :ρe_int in propertynames(Y.c)
        @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 = -(ᶠgradᵥ_matrix()) * ∂ᶜK∂ᶠw
    elseif :ρe in propertynames(Y.c)
        @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 =
            -(
                DiagonalMatrixRow(1 / ᶠinterp(ᶜρ)) *
                ᶠgradᵥ_matrix() *
                DiagonalMatrixRow(-(ᶜρ * R_d / cv_d)) + ᶠgradᵥ_matrix()
            ) * ∂ᶜK∂ᶠw
    end

    I = one(∂R∂Y)
    if transform
        @. ∂R∂Y = I / FT(δtγ) - ∂Yₜ∂Y
    else
        @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
    end
end
