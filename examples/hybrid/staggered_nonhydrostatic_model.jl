using LinearAlgebra: ×, norm, norm_sqr, dot

using ClimaCore: Operators, Fields

include("../implicit_solver_debugging_tools.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")
include("schur_complement_W.jl")
include("hyperdiffusion.jl")

const ∇◦ₕ = Operators.Divergence()
const W∇◦ₕ = Operators.WeakDivergence()
const ∇ₕ = Operators.Gradient()
const W∇ₕ = Operators.WeakGradient()
const ∇⨉ₕ = Operators.Curl()
const W∇⨉ₕ = Operators.WeakCurl()

const ᶜI = Operators.InterpolateF2C()
const ᶠI = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜ∇◦ᵥ = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
)
const ᶠ∇ᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
const ᶠ∇⨉ᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
)
const ᶜFC = Operators.FluxCorrectionC2C(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

const ᶜI_stencil = Operators.Operator2Stencil(ᶜI)
const ᶠI_stencil = Operators.Operator2Stencil(ᶠI)
const ᶜ∇◦ᵥ_stencil = Operators.Operator2Stencil(ᶜ∇◦ᵥ)
const ᶠ∇ᵥ_stencil = Operators.Operator2Stencil(ᶠ∇ᵥ)

const C123 = Geometry.Covariant123Vector

pressure_ρθ(ρθ) = p_0 * (ρθ * R_d / p_0)^γ
pressure_ρe(ρe, K, Φ, ρ) = ρ * R_d * ((ρe / ρ - K - Φ) / cv_d + T_tri)
pressure_ρe_int(ρe_int, ρ) = R_d * (ρe_int / cv_d + ρ * T_tri)

get_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = merge(
    default_cache(ᶜlocal_geometry, ᶠlocal_geometry),
    additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt),
)

function default_cache(ᶜlocal_geometry, ᶠlocal_geometry)
    ᶜcoord = ᶜlocal_geometry.coordinates
    ᶜΦ = @. grav * ᶜcoord.z
    ᶠ∇ᵥᶜΦ = @. ᶠ∇ᵥ(ᶜΦ)
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
    else
        ᶜf = map(_ -> f, ᶜlocal_geometry)
    end
    ᶜf = @. Geometry.Contravariant3Vector(Geometry.WVector(ᶜf))
    return (;
        ᶜuvw = similar(ᶜlocal_geometry, Geometry.Covariant123Vector{FT}),
        ᶜK = similar(ᶜlocal_geometry, FT),
        ᶜΦ,
        ᶠ∇ᵥᶜΦ,
        ᶜp = similar(ᶜlocal_geometry, FT),
        ᶜω³ = similar(ᶜlocal_geometry, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(ᶠlocal_geometry, Geometry.Contravariant12Vector{FT}),
        ᶠu¹² = similar(ᶠlocal_geometry, Geometry.Contravariant12Vector{FT}),
        ᶠu³ = similar(ᶠlocal_geometry, Geometry.Contravariant3Vector{FT}),
        ᶜf,
    )
end

additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)

function implicit_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜΦ, ᶠ∇ᵥᶜΦ, ᶜp) = p

    # Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
    # allocation because the cache is stored separately from Y, which means that
    # similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.
    if eltype(Y) <: Dual
        ᶜp = similar(ᶜρ)
    end

    @. Yₜ.c.ρ = -(ᶜ∇◦ᵥ(ᶠI(ᶜρ) * ᶠw))

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)
        @. Yₜ.c.ρθ = -(ᶜ∇◦ᵥ(ᶠI(ᶜρθ) * ᶠw))
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, norm_sqr(C123(ᶜuₕ) + C123(ᶜI(ᶠw))) / 2, ᶜΦ, ᶜρ)
        @. Yₜ.c.ρe = -(ᶜ∇◦ᵥ(ᶠI(ᶜρe + ᶜp) * ᶠw))
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)
        @. Yₜ.c.ρe_int = -(
            ᶜ∇◦ᵥ(ᶠI(ᶜρe_int + ᶜp) * ᶠw) -
            ᶜI(dot(ᶠ∇ᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)))
        )
        # or, equivalently,
        # @. Yₜ.c.ρe_int = -(ᶜ∇◦ᵥ(ᶠI(ᶜρe_int) * ᶠw) + ᶜp * ᶜ∇◦ᵥ(ᶠw))
    end

    Yₜ.c.uₕ .= Ref(zero(eltype(Yₜ.c.uₕ)))

    @. Yₜ.f.w = -(ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) + ᶠ∇ᵥᶜΦ)

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
    additional_remaining_tendency!(Yₜ, Y, p, t)
    Spaces.weighted_dss!(Yₜ.c)
    Spaces.weighted_dss!(Yₜ.f)
    return Yₜ
end

function default_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶠu³, ᶜf) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜI(ᶠw))
    @. ᶜK = norm_sqr(ᶜuvw) / 2

    # Mass conservation

    @. Yₜ.c.ρ -= ∇◦ₕ(ᶜρ * ᶜuvw)
    @. Yₜ.c.ρ -= ᶜ∇◦ᵥ(ᶠI(ᶜρ * ᶜuₕ))

    # Energy conservation

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)
        @. Yₜ.c.ρθ -= ∇◦ₕ(ᶜρθ * ᶜuvw)
        @. Yₜ.c.ρθ -= ᶜ∇◦ᵥ(ᶠI(ᶜρθ * ᶜuₕ))
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
        @. Yₜ.c.ρe -= ∇◦ₕ((ᶜρe + ᶜp) * ᶜuvw)
        @. Yₜ.c.ρe -= ᶜ∇◦ᵥ(ᶠI((ᶜρe + ᶜp) * ᶜuₕ))
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)
        if point_type <: Geometry.Abstract3DPoint
            @. Yₜ.c.ρe_int -=
                ∇◦ₕ((ᶜρe_int + ᶜp) * ᶜuvw) -
                dot(∇ₕ(ᶜp), Geometry.Contravariant12Vector(ᶜuₕ))
        else
            @. Yₜ.c.ρe_int -=
                ∇◦ₕ((ᶜρe_int + ᶜp) * ᶜuvw) -
                dot(∇ₕ(ᶜp), Geometry.Contravariant1Vector(ᶜuₕ))
        end
        @. Yₜ.c.ρe_int -= ᶜ∇◦ᵥ(ᶠI((ᶜρe_int + ᶜp) * ᶜuₕ))
        # or, equivalently,
        # @. Yₜ.c.ρe_int -= ∇◦ₕ(ᶜρe_int * ᶜuvw) + ᶜp * ∇◦ₕ(ᶜuvw)
        # @. Yₜ.c.ρe_int -= ᶜ∇◦ᵥ(ᶠI(ᶜρe_int * ᶜuₕ)) + ᶜp * ᶜ∇◦ᵥ(ᶠI(ᶜuₕ))
    end

    # Momentum conservation

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = ∇⨉ₕ(ᶜuₕ)
        @. ᶠω¹² = ∇⨉ₕ(ᶠw)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= Ref(zero(eltype(ᶜω³)))
        @. ᶠω¹² = Geometry.Contravariant12Vector(∇⨉ₕ(ᶠw))
    end
    @. ᶠω¹² += ᶠ∇⨉ᵥ(ᶜuₕ)

    # TODO: Modify to account for topography
    @. ᶠu¹² = Geometry.Contravariant12Vector(ᶠI(ᶜuₕ))
    @. ᶠu³ = Geometry.Contravariant3Vector(ᶠw)

    @. Yₜ.c.uₕ -=
        ᶜI(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × Geometry.Contravariant12Vector(ᶜuₕ)
    if point_type <: Geometry.Abstract3DPoint
        @. Yₜ.c.uₕ -= ∇ₕ(ᶜp) / ᶜρ + ∇ₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. Yₜ.c.uₕ -= Geometry.Covariant12Vector(∇ₕ(ᶜp) / ᶜρ + ∇ₕ(ᶜK + ᶜΦ))
    end

    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹² + ᶠ∇ᵥ(ᶜK) # TODO: put 2nd term in implicit_tendency
end

additional_remaining_tendency!(Yₜ, Y, p, t) = nothing

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

function Wfact!(W, Y, p, dtγ, t)
    (; flags, dtγ_ref, ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶠ∇ᵥᶜΦ, ᶜp) = p

    dtγ_ref[] = dtγ

    # If we let ᶠw_data = ᶠw.components.data.:1 and ᶠw_unit = one(ᶠw), then
    # ᶠw = ᶠw_data * ᶠw_unit. The Jacobian blocks involve ᶠw_data, not ᶠw.

    # ᶜρₜ = -ᶜ∇◦ᵥ(ᶠI(ᶜρ) * ᶠw)
    # ∂(ᶜρₜ)/∂(ᶠw_data) = -ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρ) * ᶠw_unit)
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρ) * one(ᶠw)))

    if :ρθ in propertynames(Y.c)
        ᶜρθ = Y.c.ρθ
        @. ᶜp = pressure_ρθ(ᶜρθ)

        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρθ")
        end

        # ᶜρθₜ = -ᶜ∇◦ᵥ(ᶠI(ᶜρθ) * ᶠw)
        # ∂(ᶜρθₜ)/∂(ᶠw_data) = -ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρθ) * ᶠw_unit)
        @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρθ) * one(ᶠw)))
    elseif :ρe in propertynames(Y.c)
        ᶜρe = Y.c.ρe
        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜI(ᶠw))) / 2
        @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)

        # ᶜI(ᶠw) = ᶜI(ᶠw)_data * ᶜI(ᶠw)_unit = ᶜI(ᶠw_data) * ᶜI(ᶠw)_unit
        # norm_sqr(ᶜI(ᶠw)) =
        #     norm_sqr(ᶜI(ᶠw_data) * ᶜI(ᶠw)_unit) =
        #     ᶜI(ᶠw_data)^2 * norm(ᶜI(ᶠw)_unit)^2
        # ᶜK =
        #     norm_sqr(C123(ᶜuₕ) + C123(ᶜI(ᶠw))) / 2 =
        #     norm_sqr(ᶜuₕ) / 2 + norm_sqr(ᶜI(ᶠw)) / 2 =
        #     norm_sqr(ᶜuₕ) / 2 + ᶜI(ᶠw_data)^2 * norm(ᶜI(ᶠw)_unit)^2 / 2

        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
            # ᶜρeₜ = -ᶜ∇◦ᵥ(ᶠI(ᶜρe + ᶜp) * ᶠw)
            # ∂(ᶜρeₜ)/∂(ᶠw_data) =
            #     -ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρe + ᶜp) * ᶠw_unit) -
            #     ᶜ∇◦ᵥ_stencil(ᶠw) * ∂(ᶠI(ᶜρe + ᶜp))/∂(ᶠw_data)
            # ∂(ᶠI(ᶜρe + ᶜp))/∂(ᶠw_data) =
            #     ∂(ᶠI(ᶜρe + ᶜp))/∂(ᶜp) * ∂(ᶜp)/∂(ᶠw_data)
            # ∂(ᶠI(ᶜρe + ᶜp))/∂(ᶜp) = ᶠI_stencil(1)
            # ∂(ᶜp)/∂(ᶠw_data) = ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw_data)
            # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
            # ∂(ᶜK)/∂(ᶠw_data) =
            #     ∂(ᶜK)/∂(ᶜI(ᶠw_data)) * ∂(ᶜI(ᶠw_data))/∂(ᶠw_data)
            # ∂(ᶜK)/∂(ᶜI(ᶠw_data)) = ᶜI(ᶠw_data) * norm(ᶜI(ᶠw)_unit)^2
            # ∂(ᶜI(ᶠw_data))/∂(ᶠw_data) = ᶜI_stencil(1)
            ᶠw_data = ᶠw.components.data.:1
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                -(ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρe + ᶜp) * one(ᶠw))) - compose(
                    ᶜ∇◦ᵥ_stencil(ᶠw),
                    compose(
                        ᶠI_stencil(one(ᶜp)),
                        -(ᶜρ * R_d / cv_d) *
                        ᶜI(ᶠw_data) *
                        norm(one(ᶜI(ᶠw)))^2 *
                        ᶜI_stencil(one(ᶠw_data)),
                    ),
                )
        elseif flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂p∂K
            # same as above, but we approximate ∂(ᶜp)/∂(ᶜK) = 0, so that ∂ᶜ𝔼ₜ∂ᶠ𝕄
            # has 3 diagonals instead of 5
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρe + ᶜp) * one(ᶠw)))
        else
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact or :no_∂p∂K when using ρe")
        end
    elseif :ρe_int in propertynames(Y.c)
        ᶜρe_int = Y.c.ρe_int
        @. ᶜp = pressure_ρe_int(ᶜρe_int, ᶜρ)

        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρe_int")
        end

        # ᶜρe_intₜ =
        #     -ᶜ∇◦ᵥ(ᶠI(ᶜρe_int + ᶜp) * ᶠw) +
        #     ᶜI(dot(ᶠ∇ᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)))
        # ∂(ᶜρe_intₜ)/∂(ᶠw_data) =
        #     -ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρe_int + ᶜp) * ᶠw_unit) +
        #     ᶜI_stencil(dot(ᶠ∇ᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw_unit)))
        @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
            -(ᶜ∇◦ᵥ_stencil(ᶠI(ᶜρe_int + ᶜp) * one(ᶠw))) +
            ᶜI_stencil(dot(ᶠ∇ᵥ(ᶜp), Geometry.Contravariant3Vector(one(ᶠw))))
    end

    # To convert ∂(ᶠwₜ)/∂(ᶜ𝔼) to ∂(ᶠw_data)ₜ/∂(ᶜ𝔼), we must extract the third
    # component of each vector-valued stencil coefficient.
    to_scalar_coefs(vector_coefs) =
        map(vector_coef -> vector_coef.u₃, vector_coefs)

    # TODO: If we use :∇Φ_shenanigans, optimize it to `cached_stencil / ᶠI(ᶜρ)`.
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :exact && flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :∇Φ_shenanigans
        error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :∇Φ_shenanigans")
    end

    if :ρθ in propertynames(Y.c)
        # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
        # ∂(ᶠwₜ)/∂(ᶜρθ) = ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) * ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρθ)
        # ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) = -1 / ᶠI(ᶜρ)
        # ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρθ) = ᶠ∇ᵥ_stencil(γ * R_d * (ᶜρθ * R_d / p_0)^(γ - 1))
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
            -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(γ * R_d * (ᶜρθ * R_d / p_0)^(γ - 1)),
        )

        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρ) = ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) * ∂(ᶠI(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) = ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2
            # ∂(ᶠI(ᶜρ))/∂(ᶜρ) = ᶠI_stencil(1)
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                to_scalar_coefs(ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2 * ᶠI_stencil(one(ᶜρ)))
        elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :∇Φ_shenanigans
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ′) - ᶠ∇ᵥᶜΦ / ᶠI(ᶜρ′) * ᶠI(ᶜρ), where
            # ᶜρ′ = ᶜρ but we approximate ∂(ᶜρ′)/∂(ᶜρ) = 0
            @. ∂ᶠ𝕄ₜ∂ᶜρ =
                to_scalar_coefs(-(ᶠ∇ᵥᶜΦ) / ᶠI(ᶜρ) * ᶠI_stencil(one(ᶜρ)))
        end
    elseif :ρe in propertynames(Y.c)
        # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
        # ∂(ᶠwₜ)/∂(ᶜρe) = ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) * ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρe)
        # ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) = -1 / ᶠI(ᶜρ)
        # ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρe) = ᶠ∇ᵥ_stencil(R_d / cv_d)
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
            to_scalar_coefs(-1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d / cv_d * one(ᶜρe)))

        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρ) =
            #     ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) * ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρ) +
            #     ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) * ∂(ᶠI(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) = -1 / ᶠI(ᶜρ)
            # ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρ) = ᶠ∇ᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri))
            # ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) = ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2
            # ∂(ᶠI(ᶜρ))/∂(ᶜρ) = ᶠI_stencil(1)
            @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri)) +
                ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2 * ᶠI_stencil(one(ᶜρ)),
            )
        elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :∇Φ_shenanigans
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp′) / ᶠI(ᶜρ′) - ᶠ∇ᵥᶜΦ / ᶠI(ᶜρ′) * ᶠI(ᶜρ), where
            # ᶜρ′ = ᶜρ but we approximate ∂ᶜρ′/∂ᶜρ = 0, and where ᶜp′ = ᶜp but
            # with ᶜK = 0
            @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d * (-(ᶜΦ) / cv_d + T_tri)) -
                ᶠ∇ᵥᶜΦ / ᶠI(ᶜρ) * ᶠI_stencil(one(ᶜρ)),
            )
        end
    elseif :ρe_int in propertynames(Y.c)
        # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
        # ∂(ᶠwₜ)/∂(ᶜρe_int) = ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) * ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρe_int)
        # ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) = -1 / ᶠI(ᶜρ)
        # ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρe_int) = ᶠ∇ᵥ_stencil(R_d / cv_d)
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
            -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d / cv_d * one(ᶜρe_int)),
        )

        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ) - ᶠ∇ᵥᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρ) =
            #     ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) * ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρ) +
            #     ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) * ∂(ᶠI(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠ∇ᵥ(ᶜp)) = -1 / ᶠI(ᶜρ)
            # ∂(ᶠ∇ᵥ(ᶜp))/∂(ᶜρ) = ᶠ∇ᵥ_stencil(R_d * T_tri)
            # ∂(ᶠwₜ)/∂(ᶠI(ᶜρ)) = ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2
            # ∂(ᶠI(ᶜρ))/∂(ᶜρ) = ᶠI_stencil(1)
            @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d * T_tri * one(ᶜρe_int)) +
                ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ)^2 * ᶠI_stencil(one(ᶜρ)),
            )
        elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :∇Φ_shenanigans
            # ᶠwₜ = -ᶠ∇ᵥ(ᶜp) / ᶠI(ᶜρ′) - ᶠ∇ᵥᶜΦ / ᶠI(ᶜρ′) * ᶠI(ᶜρ), where
            # ᶜp′ = ᶜp but we approximate ∂ᶜρ′/∂ᶜρ = 0
            @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                -1 / ᶠI(ᶜρ) * ᶠ∇ᵥ_stencil(R_d * T_tri * one(ᶜρe_int)) -
                ᶠ∇ᵥᶜΦ / ᶠI(ᶜρ) * ᶠI_stencil(one(ᶜρ)),
            )
        end
    end

    if W.test
        # Checking every column takes too long, so just check one.
        i, j, h = 1, 1, 1
        if :ρθ in propertynames(Y.c)
            ᶜ𝔼_name = :ρθ
        elseif :ρe in propertynames(Y.c)
            ᶜ𝔼_name = :ρe
        elseif :ρe_int in propertynames(Y.c)
            ᶜ𝔼_name = :ρe_int
        end
        args = (implicit_tendency!, Y, p, t, i, j, h)
        @assert column_matrix(∂ᶜρₜ∂ᶠ𝕄, i, j, h) ==
                exact_column_jacobian_block(args..., (:c, :ρ), (:f, :w))
        @assert column_matrix(∂ᶠ𝕄ₜ∂ᶜ𝔼, i, j, h) ≈
                exact_column_jacobian_block(args..., (:f, :w), (:c, ᶜ𝔼_name))
        ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx = column_matrix(∂ᶜ𝔼ₜ∂ᶠ𝕄, i, j, h)
        ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact =
            exact_column_jacobian_block(args..., (:c, ᶜ𝔼_name), (:f, :w))
        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            @assert ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx ≈ ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact
        else
            err = norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_approx .- ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact) / norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_exact)
            @assert err < 1e-6
            # Note: the highest value seen so far is ~3e-7 (only applies to ρe)
        end
        ∂ᶠ𝕄ₜ∂ᶜρ_approx = column_matrix(∂ᶠ𝕄ₜ∂ᶜρ, i, j, h)
        ∂ᶠ𝕄ₜ∂ᶜρ_exact = exact_column_jacobian_block(args..., (:f, :w), (:c, :ρ))
        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            @assert ∂ᶠ𝕄ₜ∂ᶜρ_approx ≈ ∂ᶠ𝕄ₜ∂ᶜρ_exact
        else
            err = norm(∂ᶠ𝕄ₜ∂ᶜρ_approx .- ∂ᶠ𝕄ₜ∂ᶜρ_exact) / norm(∂ᶠ𝕄ₜ∂ᶜρ_exact)
            @assert err < 0.03
            # Note: the highest value seen so far for ρe is ~0.01, and the
            # highest value seen so far for ρθ is ~0.02
        end
    end
end
