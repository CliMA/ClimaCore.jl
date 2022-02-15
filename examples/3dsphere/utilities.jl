using LinearAlgebra
using LinearAlgebra: norm_sqr

using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Operators, Spaces, Fields

include("../ordinary_diff_eq_bug_fixes.jl")
include("../schur_complement_W.jl")

const R = 6.371229e6    # Earth's radius
const grav = 9.80616    # Earth's gravitational acceleration
const Ω = 7.29212e-5    # Earth's rotation rate (radians / sec)
const R_d = 287.0       # dry specific gas constant (R / molar mass of dry air)
const κ = 2 / 7         # kappa
const γ = 1.4           # heat capacity ratio
const cp_d = R_d / κ    # heat capacity at constant pressure
const cv_d = cp_d - R_d # heat capacity at constant volume
const T_tri = 273.16    # triple point temperature
const p_0 = 1.0e5       # reference pressure

pressure(ρθ) = p_0 * (ρθ * R_d / p_0)^γ
pressure(ρ, e, K, Φ) = ρ * R_d * ((e - K - Φ) / cv_d + T_tri)

function local_geometry_fields(FT, zmax, velem, helem, npoly)
    vdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(FT)),
        Geometry.ZPoint{FT}(zmax);
        boundary_tags = (:bottom, :top),
    )
    vmesh = Meshes.IntervalMesh(vdomain, nelems = velem)
    vspace = Spaces.CenterFiniteDifferenceSpace(vmesh)

    hdomain = Domains.SphereDomain(R)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(hmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return (
        Fields.local_geometry_field(center_space),
        Fields.local_geometry_field(face_space),
    )
end

const Iᶜ = Operators.InterpolateF2C()
const Iᶠ = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ∇◦ᵥᶜ = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
)
const ∇ᵥᶠ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
)

const Iᶜ_stencil = Operators.Operator2Stencil(Iᶜ)
const Iᶠ_stencil = Operators.Operator2Stencil(Iᶠ)
const ∇◦ᵥᶜ_stencil = Operators.Operator2Stencil(∇◦ᵥᶜ)
const ∇ᵥᶠ_stencil = Operators.Operator2Stencil(∇ᵥᶠ)

function implicit_cache_values(Y, dt)
    Φ = grav .* Fields.local_geometry_field(axes(Y.Yc.ρ)).coordinates.z
    return (; P = similar(Y.Yc.ρ), Φ, ∇ᵥΦ = ∇ᵥᶠ.(Φ))
end

function implicit_tendency!(Yₜ, Y, p, t)
    @unpack ρ = Y.Yc
    @unpack uₕ, w = Y
    @unpack P, Φ, ∇ᵥΦ = p

    @. Yₜ.Yc.ρ = -∇◦ᵥᶜ(Iᶠ(ρ) * w)

    if eltype(Y) <: Dual
        P = similar(Y.Yc.ρ)
    end

    if :ρθ in propertynames(Y.Yc)
        ρθ = Y.Yc.ρθ
        @. P = pressure(ρθ)
        @. Yₜ.Yc.ρθ = -∇◦ᵥᶜ(w * Iᶠ(ρθ))
    elseif :ρe in propertynames(Y.Yc)
        ρe = Y.Yc.ρe
        V = Geometry.Covariant123Vector
        @. P = pressure(ρ, ρe / ρ, norm_sqr(V(uₕ) + V(Iᶜ(w))) / 2, Φ)
        @. Yₜ.Yc.ρe = -∇◦ᵥᶜ(w * Iᶠ(ρe + P))
    end

    @. Yₜ.w = -∇ᵥᶠ(P) / Iᶠ(ρ) - ∇ᵥΦ

    Yₜ.uₕ .= Ref(zero(eltype(uₕ)))

    # Spaces.weighted_dss!(Yₜ.Yc)
    # Spaces.weighted_dss!(Yₜ.uₕ)
    # Spaces.weighted_dss!(Yₜ.w)

    return Yₜ
end

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

function Wfact!(W, Y, p, dtγ, t)
    @unpack flags, dtγ_ref, ∂ρₜ∂𝕄, ∂𝔼ₜ∂𝕄, ∂𝕄ₜ∂𝔼, ∂𝕄ₜ∂ρ = W
    @unpack ρ = Y.Yc
    @unpack uₕ, w = Y
    @unpack P, Φ, ∇ᵥΦ = p

    dtγ_ref[] = dtγ

    # ρₜ = -∇◦ᵥᶜ(Iᶠ(ρ) * w)
    # ∂ρₜ/∂w = -∇◦ᵥᶜ_stencil(Iᶠ(ρ) * one(w))
    @. ∂ρₜ∂𝕄 = -∇◦ᵥᶜ_stencil(Iᶠ(ρ) * one(w))

    if :ρθ in propertynames(Y.Yc)
        ρθ = Y.Yc.ρθ
        @. P = pressure(ρθ)

        if flags.∂𝔼ₜ∂𝕄_mode != :exact
            error("∂𝔼ₜ∂𝕄_mode must be :exact when using ρθ")
        end

        # ρθₜ = -∇◦ᵥᶜ(Iᶠ(ρθ) * w)
        # ∂ρθₜ/∂w = -∇◦ᵥᶜ_stencil(Iᶠ(ρθ) * one(w))
        @. ∂𝔼ₜ∂𝕄 = -∇◦ᵥᶜ_stencil(Iᶠ(ρθ) * one(w))
    elseif :ρe in propertynames(Y.Yc)
        ρe = Y.Yc.ρe
        V = Geometry.Covariant123Vector
        @. P = pressure(ρ, ρe / ρ, norm_sqr(V(uₕ) + V(Iᶜ(w))) / 2, Φ)

        if flags.∂𝔼ₜ∂𝕄_mode == :exact
            # ρeₜ = -∇◦ᵥᶜ(Iᶠ(ρe + P) * w)
            # ∂ρeₜ/∂w = -∇◦ᵥᶜ_stencil(Iᶠ(ρe + P) * one(w))
            # ∂ρeₜ/∂(Iᶠ(ρe + P)) = -∇◦ᵥᶜ_stencil(w)
            # ∂(Iᶠ(ρe + P))/∂P = Iᶠ_stencil(one(P))
            # ∂P/∂(norm_sqr(V(uₕ) + V(Iᶜ(w)))) = -ρ * R_d / cv_d
            # ∂(norm_sqr(V(uₕ) + V(Iᶜ(w)))/∂(Iᶜ(w)) =
            #     ∂_norm_w_∂_w_data^2 * Iᶜ(w_data)
            # ∂(Iᶜ(w))/∂w = Iᶜ_stencil(one(w_data))
            w_data = w.components.data.:1
            if eltype(w) <: Geometry.Covariant3Vector
                ∂_norm_w_∂_w_data =
                    Fields.local_geometry_field(axes(P)).∂ξ∂x.components.data.:9
            elseif eltype(w) <: Geometry.WVector
                ∂_norm_w_∂_w_data = 1
            end
            @. ∂𝔼ₜ∂𝕄 =
                -∇◦ᵥᶜ_stencil(Iᶠ(ρe + P) * one(w)) + compose(
                    compose(-∇◦ᵥᶜ_stencil(w), Iᶠ_stencil(one(P))),
                    -ρ * R_d / cv_d *
                    ∂_norm_w_∂_w_data^2 *
                    Iᶜ(w_data) *
                    Iᶜ_stencil(one(w_data)),
                )
        elseif flags.∂𝔼ₜ∂𝕄_mode == :constant_P
            # ρeₜ = -∇◦ᵥᶜ(Iᶠ(ρe + P′) * w); ∂P′/∂ρ = 0
            @. ∂𝔼ₜ∂𝕄 = -∇◦ᵥᶜ_stencil(Iᶠ(ρe + P) * one(w))
        else
            error("∂𝔼ₜ∂𝕄_mode must be :exact or :constant_P when using ρe")
        end
    end

    # To get scalar stencils, we must extract each Covariant3Vector's value.
    to_scalar_coefs(vector_coefs) =
        map(vector_coef -> vector_coef.u₃, vector_coefs)

    if :ρθ in propertynames(Y.Yc)
        # wₜ = -∇ᵥᶠ(P) / Iᶠ(ρ) - ∇ᵥΦ
        # ∂wₜ/∂(∇ᵥᶠ(P)) = -1 / Iᶠ(ρ)
        # ∂(∇ᵥᶠ(P))/∂(ρθ) = ∇ᵥᶠ_stencil(γ * R_d * (ρθ * R_d / p_0)^(γ - 1))
        @. ∂𝕄ₜ∂𝔼 = to_scalar_coefs(
            -1 / Iᶠ(ρ) * ∇ᵥᶠ_stencil(γ * R_d * (ρθ * R_d / p_0)^(γ - 1)),
        )
        if flags.∂𝕄ₜ∂ρ_mode == :exact
            # wₜ = -∇ᵥᶠ(P) / Iᶠ(ρ) - ∇ᵥΦ
            # ∂wₜ/∂(Iᶠ(ρ)) = ∇ᵥᶠ(P) / Iᶠ(ρ)^2
            # ∂(Iᶠ(ρ))/∂ρ = Iᶠ_stencil(one(ρ))
            @. ∂𝕄ₜ∂ρ = to_scalar_coefs(∇ᵥᶠ(P) / Iᶠ(ρ)^2 * Iᶠ_stencil(one(ρ)))
        elseif flags.∂𝕄ₜ∂ρ_mode == :∇Φ_shenanigans
            # wₜ = -∇ᵥᶠ(P) / Iᶠ(ρ′) - ∇ᵥΦ / Iᶠ(ρ′) * Iᶠ(ρ); ∂ρ′/∂ρ = 0
            @. ∂𝕄ₜ∂ρ = to_scalar_coefs(∇ᵥΦ / Iᶠ(ρ) * Iᶠ_stencil(one(ρ)))
        else
            error("∂𝕄ₜ∂ρ_mode must be :exact or :∇Φ_shenanigans when using ρθ")
        end
    elseif :ρe in propertynames(Y.Yc)
        # wₜ = -∇ᵥᶠ(P) / Iᶠ(ρ) - ∇ᵥΦ
        # ∂wₜ/∂(∇ᵥᶠ(P)) = -1 / Iᶠ(ρ)
        # ∂(∇ᵥᶠ(P))/∂(ρe) = ∇ᵥᶠ_stencil(R_d / cv_d * one(ρe))
        @. ∂𝕄ₜ∂𝔼 =
            to_scalar_coefs(-1 / Iᶠ(ρ) * ∇ᵥᶠ_stencil(R_d / cv_d * one(ρe)))
        if flags.∂𝕄ₜ∂ρ_mode == :exact
            # wₜ = -∇ᵥᶠ(P) / Iᶠ(ρ) - ∇ᵥΦ
            # ∂(∇ᵥᶠ(P))/∂ρ = ∇ᵥᶠ_stencil(
            #     R_d * (-(Φ + norm_sqr(V(uₕ) + V(Iᶜ(w))) / 2) / cv_d + T_tri)
            # )
            # ∂wₜ/∂(Iᶠ(ρ)) = ∇ᵥᶠ(P) / Iᶠ(ρ)^2
            # ∂(Iᶠ(ρ))/∂ρ = Iᶠ_stencil(one(ρ))
            @. ∂𝕄ₜ∂ρ = to_scalar_coefs(
                -1 / Iᶠ(ρ) * ∇ᵥᶠ_stencil(
                    R_d *
                    (-(Φ + norm_sqr(V(uₕ) + V(Iᶜ(w))) / 2) / cv_d + T_tri),
                ) + ∇ᵥᶠ(P) / Iᶠ(ρ)^2 * Iᶠ_stencil(one(ρ)),
            )
        elseif flags.∂𝕄ₜ∂ρ_mode == :constant_P
            # wₜ = -∇ᵥᶠ(P′) / Iᶠ(ρ) - ∇ᵥΦ; ∂P′/∂ρ = 0
            @. ∂𝕄ₜ∂ρ = to_scalar_coefs(∇ᵥᶠ(P) / Iᶠ(ρ)^2 * Iᶠ_stencil(one(ρ)))
        elseif flags.∂𝕄ₜ∂ρ_mode == :∇Φ_shenanigans
            # wₜ = -∇ᵥᶠ(P′) / Iᶠ(ρ′) - ∇ᵥΦ / Iᶠ(ρ′) * Iᶠ(ρ); ∂P′/∂ρ = ∂ρ′/∂ρ = 0
            @. ∂𝕄ₜ∂ρ = to_scalar_coefs(∇ᵥΦ / Iᶠ(ρ) * Iᶠ_stencil(one(ρ)))
        end
    end

    # Spaces.weighted_dss!(∂ρₜ∂𝕄)
    # Spaces.weighted_dss!(∂𝔼ₜ∂𝕄)
    # Spaces.weighted_dss!(∂𝕄ₜ∂𝔼)
    # Spaces.weighted_dss!(∂𝕄ₜ∂ρ)

    if W.test
        # Checking every column takes too long, so just check one.
        i, j, h = 1, 1, 1
        if :ρθ in propertynames(Y.Yc)
            𝔼_name = :ρθ
        elseif :ρe in propertynames(Y.Yc)
            𝔼_name = :ρe
        end
        args = (implicit_tendency!, Y, p, t, i, j, h)
        @assert column_matrix(∂ρₜ∂𝕄, i, j, h) ≈
                exact_column_jacobian(args..., (:w,), (:Yc, :ρ))
        @assert column_matrix(∂𝕄ₜ∂𝔼, i, j, h) ≈
                exact_column_jacobian(args..., (:Yc, 𝔼_name), (:w,))
        ∂𝔼ₜ∂𝕄_approx = column_matrix(∂𝔼ₜ∂𝕄, i, j, h)
        ∂𝔼ₜ∂𝕄_exact = exact_column_jacobian(args..., (:w,), (:Yc, 𝔼_name))
        if flags.∂𝔼ₜ∂𝕄_mode == :exact
            @assert ∂𝔼ₜ∂𝕄_approx ≈ ∂𝔼ₜ∂𝕄_exact
        else
            @assert norm((∂𝔼ₜ∂𝕄_approx .- ∂𝔼ₜ∂𝕄_exact)) / norm(∂𝔼ₜ∂𝕄_exact) <
                    1e-5 # highest seen so far
        end
        ∂𝕄ₜ∂ρ_approx = column_matrix(∂𝕄ₜ∂ρ, i, j, h)
        ∂𝕄ₜ∂ρ_exact = exact_column_jacobian(args..., (:Yc, :ρ), (:w,))
        if flags.∂𝕄ₜ∂ρ_mode == :exact
            @assert ∂𝕄ₜ∂ρ_approx ≈ ∂𝕄ₜ∂ρ_exact
        else
            @assert norm((∂𝕄ₜ∂ρ_approx .- ∂𝕄ₜ∂ρ_exact)) / norm(∂𝕄ₜ∂ρ_exact) < 3 # highest seen so far
        end
    end
end
