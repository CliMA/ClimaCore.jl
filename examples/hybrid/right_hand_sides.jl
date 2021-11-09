const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio

# P = ρ * R_d * T = ρ * R_d * θ * (P / MSLP)^(R_d / C_p) ==>
# (P / MSLP)^(1 - R_d / C_p) = R_d / MSLP * ρθ ==>
# P = MSLP * (R_d / MSLP)^γ * ρθ^γ
const P_ρθ_factor = MSLP * (R_d / MSLP)^γ
# P = ρ * R_d * T = ρ * R_d * (ρe_int / ρ / C_v) = (γ - 1) * ρe_int
const P_ρe_factor = γ - 1

norm_sqr(uₕ, w) =
    LinearAlgebra.norm_sqr(
        Geometry.transform(Geometry.UWAxis(), uₕ) +
        Geometry.transform(Geometry.UWAxis(), w)
    )

# axes
const û = Geometry.UAxis
const ŵ = Geometry.WAxis

# horizontal operators
const ∇◦ₕ = Operators.Divergence()
const ∇ₕ = Operators.Gradient()

# vertical operators
const If = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const If_uₕ = Operators.InterpolateC2F(
    bottom = Operators.SetValue(Geometry.UVector(0.0)),
    top = Operators.SetValue(Geometry.UVector(0.0)),
)
const Ic = Operators.InterpolateF2C()
const ∇◦ᵥf = Operators.DivergenceC2F()
const ∇◦ᵥc = Operators.DivergenceF2C()
const ∇ᵥf = Operators.GradientC2F()
const B_w = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(Geometry.WVector(0.0)),
    top = Operators.SetValue(Geometry.WVector(0.0)),
)

ClimaCore.RecursiveApply.rmul(x::AbstractArray, y::AbstractArray) = x * y

function rhs!(dY, Y, p, t)
    @unpack uₕ, uₕ_f, P, Φ, ∇Φ = p
    if :ρw in propertynames(Y)
        ρw = Y.ρw
    elseif :w in propertynames(Y)
        ρw = p.ρw
        @. ρw = Y.w * If(Y.Yc.ρ)
    end
    @. uₕ = Y.Yc.ρuₕ / Y.Yc.ρ
    @. uₕ_f = If_uₕ(uₕ)

    # ∂ρ/∂t = -∇◦ρu
    @. dY.Yc.ρ = -∇◦ᵥc(ρw)
    @. dY.Yc.ρ -= ∇◦ₕ(Y.Yc.ρuₕ)

    # ∂ρθ/∂t = -∇◦ρθu
    # ∂ρe/∂t = -∇◦(ρe + P)u
    if :ρθ in propertynames(Y.Yc)
        @. P = P_ρθ_factor * Y.Yc.ρθ^γ
        if :ρw in propertynames(Y)
            @. dY.Yc.ρθ = -∇◦ᵥc(ρw * If(Y.Yc.ρθ / Y.Yc.ρ))
        elseif :w in propertynames(Y)
            @. dY.Yc.ρθ = -∇◦ᵥc(Y.w * If(Y.Yc.ρθ))
        end
        @. dY.Yc.ρθ -= ∇◦ₕ(uₕ * Y.Yc.ρθ)
    elseif :ρe_tot in propertynames(Y.Yc)
        if :ρw in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot - Y.Yc.ρ * Φ -
                norm_sqr(Y.Yc.ρuₕ, Ic(ρw)) / (2. * Y.Yc.ρ)
            )
            @. dY.Yc.ρe_tot = -∇◦ᵥc(ρw * If((Y.Yc.ρe_tot + P) / Y.Yc.ρ))
        elseif :w in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot - Y.Yc.ρ * (Φ + norm_sqr(uₕ, Ic(Y.w)) / 2.)
            )
            @. dY.Yc.ρe_tot = -∇◦ᵥc(Y.w * If(Y.Yc.ρe_tot + P))
        end
        @. dY.Yc.ρe_tot -= ∇◦ₕ(uₕ * (Y.Yc.ρe_tot + P))
    end

    # ∂ρu/∂t = -∇P - ρ∇Φ - ∇◦(ρu ⊗ u)
    # ∂u/∂t = -(∇P)/ρ - ∇Φ - u◦∇u
    if :ρw in propertynames(Y)
        @. dY.ρw = B_w(
            -Geometry.transform(ŵ(), ∇ᵥf(P)) - If(Y.Yc.ρ) * ∇Φ -
            ∇◦ᵥf(Ic(ρw ⊗ ρw) / Y.Yc.ρ)
        )
        @. dY.ρw -= ∇◦ₕ(uₕ_f ⊗ ρw)
    elseif :w in propertynames(Y)
        @. dY.w = B_w(
            -Geometry.transform(ŵ(), ∇ᵥf(P)) / If(Y.Yc.ρ) - ∇Φ - 
            adjoint(∇ᵥf(Ic(Y.w))) *
                Geometry.transform(Geometry.Contravariant3Axis(), Y.w)
        )
        @. dY.w -= adjoint(∇ₕ(Y.w)) *
            Geometry.transform(Geometry.Contravariant1Axis(), uₕ_f)
    end
    eᵤᵤ = Ref(Geometry.Axis2Tensor((û(), û()), @SMatrix [1.]))
    @. dY.Yc.ρuₕ = -∇◦ᵥc(ρw ⊗ uₕ_f)
    @. dY.Yc.ρuₕ -= ∇◦ₕ(P * eᵤᵤ + Y.Yc.ρuₕ ⊗ uₕ)

    Spaces.weighted_dss!(dY.Yc)
    if :ρw in propertynames(Y)
        Spaces.weighted_dss!(dY.ρw)
    elseif :w in propertynames(Y)
        Spaces.weighted_dss!(dY.w)
    end
    return dY
end

# function rhs_implicit!(dY, Y, p, t)
#     @unpack P, Φ, ∇Φ = p

#     # ∂ρ/∂t ≈ -∇◦ᵥρu
#     if :ρw in propertynames(Y)
#         @. dY.Yc.ρ = -∇◦ᵥc(Y.ρw)
#     elseif :w in propertynames(Y)
#         @. dY.Yc.ρ = -∇◦ᵥc(Y.w * If(Y.Yc.ρ))
#     end

#     # ∂ρθ/∂t ≈ -∇◦ᵥρθu
#     # ∂ρe/∂t ≈ -∇◦ᵥ(ρe + P)u
#     if :ρθ in propertynames(Y.Yc)
#         @. P = P_ρθ_factor * Y.Yc.ρθ^γ
#         if :ρw in propertynames(Y)
#             @. dY.Yc.ρθ = -∇◦ᵥc(Y.ρw * If(Y.Yc.ρθ / Y.Yc.ρ))
#         elseif :w in propertynames(Y)
#             @. dY.Yc.ρθ = -∇◦ᵥc(Y.w * If(Y.Yc.ρθ))
#         end
#     elseif :ρe_tot in propertynames(Y.Yc)
#         if :ρw in propertynames(Y)
#             @. P = P_ρe_factor * (
#                 Y.Yc.ρe_tot - Y.Yc.ρ * Φ -
#                 norm_sqr(Y.Yc.ρuₕ, Ic(Y.ρw)) / (2. * Y.Yc.ρ)
#             )
#             @. dY.Yc.ρe_tot = -∇◦ᵥc(Y.ρw * If((Y.Yc.ρe_tot + P) / Y.Yc.ρ))
#         elseif :w in propertynames(Y)
#             @. P = P_ρe_factor * (
#                 Y.Yc.ρe_tot -
#                 Y.Yc.ρ * (Φ + norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ, Ic(Y.w)) / 2.)
#             )
#             @. dY.Yc.ρe_tot = -∇◦ᵥc(Y.w * If(Y.Yc.ρe_tot + P))
#         end
#     end

#     # ∂ρu/∂t ≈ -∇ᵥP - ρ∇ᵥΦ
#     # ∂u/∂t ≈ -(∇ᵥP)/ρ - ∇ᵥΦ
#     if :ρw in propertynames(Y)
#         @. dY.ρw = B_w(-Geometry.transform(ŵ(), ∇ᵥf(P)) - If(Y.Yc.ρ) * ∇Φ)
#     elseif :w in propertynames(Y)
#         @. dY.w = B_w(-Geometry.transform(ŵ(), ∇ᵥf(P)) / If(Y.Yc.ρ) - ∇Φ)
#     end
#     # `dY.Yc.ρuₕ .= Ref(Geometry.UVector(0.))` gives an error
#     Fields.field_values(dY.Yc.ρuₕ) .= Ref(Geometry.UVector(0.))

#     return dY
# end

# Replace fields with parent arrays.
function rhs_implicit!(dY, Y, p, t)
    @unpack P, Φ, ∇Φ = p

    N = size(parent(Y.Yc.ρ), 1)
    M = length(parent(Y.Yc.ρ)) ÷ N
    arr_c(field) = reshape(parent(field), N, M)
    arr_f(field) = reshape(parent(field), N + 1, M)

    z = arr_c(p.coords.z)
    z_f = arr_f(p.face_coords.z)
    Δz = arr_c(p.uₕ)
    Δz_f = arr_f(p.uₕ_f)
    @views @. Δz = z_f[2:N + 1, :] .- z_f[1:N, :]
    @views @. Δz_f[2:N, :] = z[2:N, :] .- z[1:N - 1, :]

    function interp_f!(dest_f, src_c)
        @views @. dest_f[2:N, :] = (src_c[1:N - 1, :] + src_c[2:N, :]) / 2.
        @views @. dest_f[1, :] = dest_f[2, :]
        @views @. dest_f[N + 1, :] = dest_f[N, :]
    end
    function interp_c!(dest_c, src_f)
        @views @. dest_c = (src_f[1:N, :] + src_f[2:N + 1, :]) / 2.
    end
    function neg_deriv_f!(dest_f, src_c)
        @views @. dest_f[2:N, :] =
            (src_c[1:N - 1, :] - src_c[2:N, :]) / Δz_f[2:N, :]
        @views @. dest_f[1, :] = 0.
        @views @. dest_f[N + 1, :] = 0.
    end
    function neg_deriv_c!(dest_c, src_f)
        @views @. dest_c = (src_f[1:N, :] - src_f[2:N + 1, :]) / Δz
    end

    P = arr_c(P)
    Φ = arr_c(Φ)
    ∇Φ = arr_f(∇Φ)
    ρ = arr_c(Y.Yc.ρ)
    dρ = arr_c(dY.Yc.ρ)
    ρuₕ = arr_c(Y.Yc.ρuₕ)
    dρuₕ = arr_c(dY.Yc.ρuₕ)

    temp_c = dρuₕ
    if :ρw in propertynames(Y)
        ρw = arr_f(Y.ρw)
        dρw = arr_f(dY.ρw)
        temp_f = dρw
    elseif :w in propertynames(Y)
        w = arr_f(Y.w)
        dw = arr_f(dY.w)
        temp_f = dw
    end

    # ∂ρ/∂t ≈ -∇◦ᵥρu
    if :w in propertynames(Y)
        ρ_f = ρw = temp_f
        interp_f!(ρ_f, ρ)
        @. ρw = ρ_f * w
    end
    neg_deriv_c!(dρ, ρw)

    # ∂ρθ/∂t ≈ -∇◦ᵥρθu
    # ∂ρe/∂t ≈ -∇◦ᵥ(ρe + P)u
    if :ρθ in propertynames(Y.Yc)
        ρθ = arr_c(Y.Yc.ρθ)
        dρθ = arr_c(dY.Yc.ρθ)
        @. P = P_ρθ_factor * ρθ^γ
        if :ρw in propertynames(Y)
            θ = temp_c
            θ_f = ρwθ = temp_f
            @. θ = ρθ / ρ
            interp_f!(θ_f, θ)
            @. ρwθ = ρw * θ_f
        elseif :w in propertynames(Y)
            ρθ_f = ρwθ = temp_f
            interp_f!(ρθ_f, ρθ)
            @. ρwθ = w * ρθ_f
        end
        neg_deriv_c!(dρθ, ρwθ)
    elseif :ρe_tot in propertynames(Y.Yc)
        ρe_tot = arr_c(Y.Yc.ρe_tot)
        dρe_tot = arr_c(dY.Yc.ρe_tot)
        if :ρw in propertynames(Y)
            ρw_c = h = temp_c
            h_f = ρwh = temp_f
            interp_c!(ρw_c, ρw)
            @. P = P_ρe_factor * (ρe_tot - ρ * Φ - (ρuₕ^2 + ρw_c^2) / (2. * ρ))
            @. h = (ρe_tot + P) / ρ
            interp_f!(h_f, h)
            @. ρwh = ρw * h_f
        elseif :w in propertynames(Y)
            w_c = ρh = temp_c
            ρh_f = ρwh = temp_f
            interp_c!(w_c, w)
            @. P = P_ρe_factor * (ρe_tot - ρ * (Φ + ((ρuₕ / ρ)^2 + w_c^2) / 2.))
            @. ρh = ρe_tot + P
            interp_f!(ρh_f, ρh)
            @. ρwh = w * ρh_f
        end
        neg_deriv_c!(dρe_tot, ρwh)
    end

    # ∂ρu/∂t ≈ -∇ᵥP - ρ∇ᵥΦ
    # ∂u/∂t ≈ -(∇ᵥP)/ρ - ∇ᵥΦ
    if :ρw in propertynames(Y)
        neg_deriv_f!(dρw, P)
        @views @. dρw[2:N, :] =
            dρw[2:N, :] - ((ρ[1:N - 1, :] + ρ[2:N, :]) / 2.) * ∇Φ[2:N, :]
    elseif :w in propertynames(Y)
        neg_deriv_f!(dw, P)
        @views @. dw[2:N, :] =
            dw[2:N, :] / ((ρ[1:N - 1, :] + ρ[2:N, :]) / 2.) - ∇Φ[2:N, :]
    end
    dρuₕ .= 0.

    return dY
end

# Sets dY to the value of rhs! - rhs_implicit!.
function rhs_remainder!(dY, Y, p, t)
    @unpack uₕ, uₕ_f, P, Φ, ∇Φ = p
    if :ρw in propertynames(Y)
        ρw = Y.ρw
    elseif :w in propertynames(Y)
        ρw = p.ρw
        @. ρw = Y.w * If(Y.Yc.ρ)
    end
    @. uₕ = Y.Yc.ρuₕ / Y.Yc.ρ
    @. uₕ_f = If_uₕ(uₕ)

    # ∂ρ/∂t Remainder = -∇◦ₕρu
    @. dY.Yc.ρ = -∇◦ₕ(Y.Yc.ρuₕ)

    # ∂ρθ/∂t Remainder = -∇◦ₕρθu
    # ∂ρe/∂t Remainder = -∇◦ₕ(ρe + P)u
    if :ρθ in propertynames(Y.Yc)
        @. P = P_ρθ_factor * Y.Yc.ρθ^γ
        @. dY.Yc.ρθ = -∇◦ₕ(uₕ * Y.Yc.ρθ)
    elseif :ρe_tot in propertynames(Y.Yc)
        if :ρw in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot - Y.Yc.ρ * Φ -
                norm_sqr(Y.Yc.ρuₕ, Ic(ρw)) / (2. * Y.Yc.ρ)
            )
        elseif :w in propertynames(Y)
            @. P = P_ρe_factor * (
                Y.Yc.ρe_tot - Y.Yc.ρ * (Φ + norm_sqr(uₕ, Ic(Y.w)) / 2.)
            )
        end
        @. dY.Yc.ρe_tot = -∇◦ₕ(uₕ * (Y.Yc.ρe_tot + P))
    end

    # ∂ρu/∂t Remainder = -∇ₕP - ρ∇ₕΦ - ∇◦(ρu ⊗ u)
    # ∂u/∂t Remainder = -(∇ₕP)/ρ - ∇ₕΦ - u◦∇u
    if :ρw in propertynames(Y)
        @. dY.ρw = B_w(-∇◦ᵥf(Ic(ρw ⊗ ρw) / Y.Yc.ρ))
        @. dY.ρw -= ∇◦ₕ(uₕ_f ⊗ ρw)
    elseif :w in propertynames(Y)
        @. dY.w = B_w(
            -adjoint(∇ᵥf(Ic(Y.w))) *
                Geometry.transform(Geometry.Contravariant3Axis(), Y.w)
        )
        @. dY.w -= adjoint(∇ₕ(Y.w)) *
            Geometry.transform(Geometry.Contravariant1Axis(), uₕ_f)
    end
    eᵤᵤ = Ref(Geometry.Axis2Tensor((û(), û()), @SMatrix [1.]))
    @. dY.Yc.ρuₕ = -∇◦ᵥc(ρw ⊗ uₕ_f)
    @. dY.Yc.ρuₕ -= ∇◦ₕ(P * eᵤᵤ + Y.Yc.ρuₕ ⊗ uₕ)

    Spaces.weighted_dss!(dY.Yc)
    if :ρw in propertynames(Y)
        Spaces.weighted_dss!(dY.ρw)
    elseif :w in propertynames(Y)
        Spaces.weighted_dss!(dY.w)
    end
    return dY
end


struct CustomWRepresentation{T,AT1,AT2,AT3,VT}
    # grid information
    velem::Int
    helem::Int
    npoly::Int

    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flag for computing the Jacobian
    J_𝕄ρ_overwrite::Symbol

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::T

    # cache for the grid values used to compute the Jacobian
    Δz::AT1
    Δz_f::AT1

    # nonzero blocks of the Jacobian (∂ρₜ/∂𝕄, ∂𝔼ₜ/∂𝕄, ∂𝕄ₜ/∂𝔼, and ∂𝕄ₜ/∂ρ)
    J_ρ𝕄::AT2
    J_𝔼𝕄::AT2
    J_𝕄𝔼::AT2
    J_𝕄ρ::AT2

    # cache for the Schur complement
    S::AT3

    # cache for variable values used to compute the Jacobian
    vals::VT
end

function CustomWRepresentation(
    velem,
    helem,
    npoly,
    coords,
    face_coords,
    transform,
    J_𝕄ρ_overwrite;
    FT = Float64,
)
    N = velem
    M = helem * (npoly + 1)

    dtγ_ref = Ref(zero(FT))

    z = reshape(parent(coords.z), N , M)
    z_f = reshape(parent(face_coords.z), N + 1, M)

    @views Δz = z_f[2:N + 1, :] .- z_f[1:N, :]
    @views Δz_f = z[2:N, :] .- z[1:N - 1, :]

    J_ρ𝕄 = (; d = Array{FT}(undef, N, M), d2 = Array{FT}(undef, N, M))
    J_𝔼𝕄 = (; d = Array{FT}(undef, N, M), d2 = Array{FT}(undef, N, M))
    J_𝕄𝔼 = (; d = Array{FT}(undef, N, M), d2 = Array{FT}(undef, N, M))
    J_𝕄ρ = (; d = Array{FT}(undef, N, M), d2 = Array{FT}(undef, N, M))

    S = Tridiagonal(
        Array{FT}(undef, N),
        Array{FT}(undef, N + 1),
        Array{FT}(undef, N),
    )

    vals = (;
        ρ_f = similar(face_coords.z),
        𝔼_value_f = similar(face_coords.z),
        P_value = similar(coords.z),
    )

    CustomWRepresentation{
        typeof(dtγ_ref),
        typeof(Δz),
        typeof(J_ρ𝕄),
        typeof(S),
        typeof(vals),
    }(
        velem,
        helem,
        npoly,
        transform,
        J_𝕄ρ_overwrite,
        dtγ_ref,
        Δz,
        Δz_f,
        J_ρ𝕄,
        J_𝔼𝕄,
        J_𝕄𝔼,
        J_𝕄ρ,
        S,
        vals,
    )
end

import Base: similar
# We only use Wfact, but the implicit/imex solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(cf::CustomWRepresentation{T,AT}) where {T, AT} = cf

function Wfact!(W, Y, p, dtγ, t)
    @unpack velem, helem, npoly, dtγ_ref, Δz, Δz_f, J_ρ𝕄, J_𝔼𝕄, J_𝕄𝔼, J_𝕄ρ,
        J_𝕄ρ_overwrite, vals = W
    @unpack ρ_f, 𝔼_value_f, P_value = vals
    @unpack P, Φ, ∇Φ = p
    N = velem
    M = (npoly + 1) * helem
    ∇Φ = reshape(parent(∇Φ), N + 1, M)
    dtγ_ref[] = dtγ

    # Rewriting in terms of parent arrays.
    N = size(parent(Y.Yc.ρ), 1)
    M = length(parent(Y.Yc.ρ)) ÷ N
    arr_c(field) = reshape(parent(field), N, M)
    arr_f(field) = reshape(parent(field), N + 1, M)
    function interp_f!(dest_f, src_c)
        @views @. dest_f[2:N, :] = (src_c[1:N - 1, :] + src_c[2:N, :]) / 2.
        @views @. dest_f[1, :] = dest_f[2, :]
        @views @. dest_f[N + 1, :] = dest_f[N, :]
    end
    function interp_c!(dest_c, src_f)
        @views @. dest_c = (src_f[1:N, :] + src_f[2:N + 1, :]) / 2.
    end
    ρ_f = arr_f(ρ_f)
    𝔼_value_f = arr_f(𝔼_value_f)
    P_value = arr_c(P_value)
    P = arr_c(P)
    Φ = arr_c(Φ)
    ρ = arr_c(Y.Yc.ρ)
    ρuₕ = arr_c(Y.Yc.ρuₕ)
    if :ρθ in propertynames(Y.Yc)
        ρθ = arr_c(Y.Yc.ρθ)
    elseif :ρe_tot in propertynames(Y.Yc)
        ρe_tot = arr_c(Y.Yc.ρe_tot)
    end
    if :ρw in propertynames(Y)
        ρw = arr_f(Y.ρw)
    elseif :w in propertynames(Y)
        w = arr_f(Y.w)
    end

    if :ρw in propertynames(Y)
        # dY.Yc.ρ = -∇◦ᵥc(Y.ρw) ==>
        # ∂ρ[n]/∂t = (ρw[n] - ρw[n + 1]) / Δz[n] ==>
        #     ∂(∂ρ[n]/∂t)/∂ρw[n] = 1 / Δz[n]
        #     ∂(∂ρ[n]/∂t)/∂ρw[n + 1] = -1 / Δz[n]
        @. J_ρ𝕄.d = 1. / Δz
        @. J_ρ𝕄.d2 = -1. / Δz
    elseif :w in propertynames(Y)
        # @. ρ_f = If(Y.Yc.ρ)
        # ρ_f = reshape(parent(ρ_f), N + 1, M)
        interp_f!(ρ_f, ρ)
        # dY.Yc.ρ = -∇◦ᵥc(Y.w * If(Y.Yc.ρ)) ==>
        # ∂ρ[n]/∂t = (w[n] ρ_f[n] - w[n + 1] ρ_f[n + 1]) / Δz[n] ==>
        #     ∂(∂ρ[n]/∂t)/∂w[n] = ρ_f[n] / Δz[n]
        #     ∂(∂ρ[n]/∂t)/∂w[n + 1] = -ρ_f[n + 1] / Δz[n]
        @views @. J_ρ𝕄.d = ρ_f[1:N, :] / Δz
        @views @. J_ρ𝕄.d2 = -ρ_f[2:N + 1, :] / Δz
    end

    # dY.Yc.𝔼 = -∇◦ᵥc(Y.𝕄 * 𝔼_value_f) ==>
    # ∂𝔼[n]/∂t = (𝕄[n] 𝔼_value_f[n] - 𝕄[n + 1] 𝔼_value_f[n + 1]) / Δz[n] ==>
    #     ∂(∂𝔼[n]/∂t)/∂𝕄[n] = 𝔼_value_f[n] / Δz[n]
    #     ∂(∂𝔼[n]/∂t)/∂𝕄[n + 1] = -𝔼_value_f[n + 1] / Δz[n]
    if :ρθ in propertynames(Y.Yc)
        if :ρw in propertynames(Y)
            # dY.Yc.ρθ = -∇◦ᵥc(Y.ρw * If(Y.Yc.ρθ / Y.Yc.ρ))
            # @. 𝔼_value_f = If(Y.Yc.ρθ / Y.Yc.ρ)
            θ = P_value
            @. θ = ρθ / ρ # temporary
            interp_f!(𝔼_value_f, θ)
        elseif :w in propertynames(Y)
            # dY.Yc.ρθ = -∇◦ᵥc(Y.w * If(Y.Yc.ρθ))
            # @. 𝔼_value_f = If(Y.Yc.ρθ)
            interp_f!(𝔼_value_f, ρθ)
        end
    elseif :ρe_tot in propertynames(Y.Yc)
        if :ρw in propertynames(Y)
            # @. P = P_ρe_factor * (
            #     Y.Yc.ρe_tot - Y.Yc.ρ * Φ -
            #     norm_sqr(Y.Yc.ρuₕ, Ic(Y.ρw)) / (2. * Y.Yc.ρ)
            # )
            ρw_c = P_value
            interp_c!(ρw_c, ρw)
            @. P = P_ρe_factor * (ρe_tot - ρ * Φ - (ρuₕ^2 + ρw_c^2) / (2. * ρ))
            # dY.Yc.ρe_tot = -∇◦ᵥc(Y.ρw * If((Y.Yc.ρe_tot + P) / Y.Yc.ρ))
            # @. 𝔼_value_f = If((Y.Yc.ρe_tot + P) / Y.Yc.ρ)
            h = P_value
            @. h = (ρe_tot + P) / ρ
            interp_f!(𝔼_value_f, h)
        elseif :w in propertynames(Y)
            # @. P = P_ρe_factor * (
            #     Y.Yc.ρe_tot -
            #     Y.Yc.ρ * (Φ + norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ, Ic(Y.w)) / 2.)
            # )
            w_c = P_value
            interp_c!(w_c, w)
            @. P = P_ρe_factor * (ρe_tot - ρ * (Φ + ((ρuₕ / ρ)^2 + w_c^2) / 2.))
            # dY.Yc.ρe_tot = -∇◦ᵥc(Y.w * If(Y.Yc.ρe_tot + P))
            # @. 𝔼_value_f = If(Y.Yc.ρe_tot + P)
            ρh = P_value
            @. ρh = ρe_tot + P
            interp_f!(𝔼_value_f, ρh)
        end
    end
    # 𝔼_value_f = reshape(parent(𝔼_value_f), N + 1, M)
    @views @. J_𝔼𝕄.d = 𝔼_value_f[1:N, :] / Δz
    @views @. J_𝔼𝕄.d2 = -𝔼_value_f[2:N + 1, :] / Δz

    # dY.𝕄 = B_w(...) ==>
    # ∂𝕄[1]/∂t = ∂𝕄[N + 1]/∂t = 0 ==>
    #     ∂(∂𝕄[1]/∂t)/∂ρ[1] = ∂(∂𝕄[1]/∂t)/∂𝔼[1] =
    #     ∂(∂𝕄[N + 1]/∂t)/∂ρ[N] = ∂(∂𝕄[N + 1]/∂t)/∂𝔼[N] = 0
    @. J_𝕄ρ.d[1, :] = J_𝕄𝔼.d[1, :] = J_𝕄ρ.d2[N, :] = J_𝕄𝔼.d2[N, :] = 0.
    # if :ρw in propertynames(Y)
        # dY.ρw = B_w(Geometry.transform(ŵ(), -∇ᵥf(P) - If(Y.Yc.ρ) * ∇Φ)) ==>
        # For all 1 < n < N + 1, ∂ρw[n]/∂t =
        # (P[n - 1] - P[n]) / Δz_f[n] - (ρ[n - 1] + ρ[n]) ∇Φ[n] / 2 ==>
        #     ∂(∂ρw[n]/∂t)/∂𝔼[n] = -∂P[n]/∂𝔼[n] / Δz_f[n]
        #     ∂(∂ρw[n]/∂t)/∂𝔼[n - 1] = ∂P[n - 1]/∂𝔼[n - 1] / Δz_f[n]
        #     ∂(∂ρw[n]/∂t)/∂ρ[n] = -∂P[n]/∂ρ[n] / Δz_f[n] - ∇Φ[n] / 2
        #     ∂(∂ρw[n]/∂t)/∂ρ[n - 1] = ∂P[n - 1]/∂ρ[n - 1] / Δz_f[n] - ∇Φ[n] / 2
    # elseif :w in propertynames(Y)
        # dY.w = B_w(Geometry.transform(ŵ(), -∇ᵥf(P) / If(Y.Yc.ρ) - ∇Φ)) ==>
        # For all 1 < n < N + 1, ∂w[n]/∂t =
        # (P[n - 1] - P[n]) / ((ρ[n - 1] + ρ[n]) / 2 * Δz_f[n]) - ∇Φ[n] ==>
        #     ∂(∂w[n]/∂t)/∂𝔼[n] = -∂P[n]/∂𝔼[n] / (ρ_f[n] Δz_f[n])
        #     ∂(∂w[n]/∂t)/∂𝔼[n - 1] = ∂P[n - 1]/∂𝔼[n - 1] / (ρ_f[n] Δz_f[n])
        #     ∂(∂w[n]/∂t)/∂ρ[n] =
        #         -∂P[n]/∂ρ[n] / (ρ_f[n] Δz_f[n]) +
        #         (P[n] - P[n - 1]) / (2 ρ_f[n]^2 Δz_f[n])
        #     ∂(∂w[n]/∂t)/∂ρ[n - 1] =
        #         ∂P[n - 1]/∂ρ[n - 1] / (ρ_f[n] Δz_f[n]) +
        #         (P[n] - P[n - 1]) / (2 ρ_f[n]^2 Δz_f[n])
    # end
    if :ρθ in propertynames(Y.Yc)
        # ∂P/∂𝔼 = γ * P_ρθ_factor * Y.Yc.ρθ^(γ - 1)
        # ∂P/∂ρ = 0
        # @. P_value = (γ * P_ρθ_factor) * Y.Yc.ρθ^(γ - 1)
        # ∂P∂𝔼 = reshape(parent(P_value), N, M)
        ∂P∂𝔼 = P_value
        @. ∂P∂𝔼 = (γ * P_ρθ_factor) * ρθ^(γ - 1)
        if :ρw in propertynames(Y)
            @views @. J_𝕄𝔼.d[2:N, :] = -∂P∂𝔼[2:N, :] / Δz_f
            @views @. J_𝕄𝔼.d2[1:N - 1, :] = ∂P∂𝔼[1:N - 1, :] / Δz_f

            if J_𝕄ρ_overwrite == :none
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    -∇Φ[2:N, :] / 2.
            end
        elseif :w in propertynames(Y)
            @views @. J_𝕄𝔼.d[2:N, :] = -∂P∂𝔼[2:N, :] / (ρ_f[2:N, :] * Δz_f)
            @views @. J_𝕄𝔼.d2[1:N - 1, :] =
                ∂P∂𝔼[1:N - 1, :] / (ρ_f[2:N, :] * Δz_f)

            if J_𝕄ρ_overwrite == :grav
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    -∇Φ[2:N, :] / (2. * ρ_f[2:N, :])
            elseif J_𝕄ρ_overwrite == :none
                # @. P = P_ρθ_factor * Y.Yc.ρθ^γ
                # P = reshape(parent(P), N, M)
                @. P = P_ρθ_factor * ρθ^γ
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    (P[2:N, :] - P[1:N - 1, :]) / (2. * ρ_f[2:N, :]^2 * Δz_f)
            end
        end
    elseif :ρe_tot in propertynames(Y.Yc)
        # ∂P/∂𝔼 = P_ρe_factor
        if :ρw in propertynames(Y)
            @. J_𝕄𝔼.d[2:N, :] = -P_ρe_factor / Δz_f
            @. J_𝕄𝔼.d2[1:N - 1, :] = P_ρe_factor / Δz_f

            # ∂P/∂ρ = P_ρe_factor *
            #     (-Φ + norm_sqr(Y.Yc.ρuₕ, Ic(Y.ρw)) / (2. * Y.Yc.ρ^2))
            @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] = -∇Φ[2:N, :] / 2.
            if J_𝕄ρ_overwrite == :none
                # @. P_value = P_ρe_factor *
                #     (-Φ + norm_sqr(Y.Yc.ρuₕ, Ic(Y.ρw)) / (2. * Y.Yc.ρ^2))
                # ∂P∂ρ = reshape(parent(P_value), N, M)
                ∂P∂ρ = ρw_c = P_value
                interp_c!(ρw_c, ρw)
                @. ∂P∂ρ = P_ρe_factor * (-Φ + (ρuₕ^2 + ρw_c^2) / (2. * ρ^2))
                @views @. J_𝕄ρ.d[2:N, :] += -∂P∂ρ[2:N, :] / Δz_f
                @views @. J_𝕄ρ.d2[1:N - 1, :] += ∂P∂ρ[1:N - 1, :] / Δz_f
            end
        elseif :w in propertynames(Y)
            @views @. J_𝕄𝔼.d[2:N, :] = -P_ρe_factor / (ρ_f[2:N, :] * Δz_f)
            @views @. J_𝕄𝔼.d2[1:N - 1, :] = P_ρe_factor / (ρ_f[2:N, :] * Δz_f)

            # ∂P/∂ρ =
            #     P_ρe_factor * (
            #         -Φ - norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ, Ic(Y.w)) / 2. +
            #         LinearAlgebra.norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ)
            #     )
            if J_𝕄ρ_overwrite == :grav
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    -∇Φ[2:N, :] / (2. * ρ_f[2:N, :])
            elseif J_𝕄ρ_overwrite == :none || J_𝕄ρ_overwrite == :pres
                # P = reshape(parent(P), N, M)
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    (P[2:N, :] - P[1:N - 1, :]) / (2. * ρ_f[2:N, :]^2 * Δz_f)
                if J_𝕄ρ_overwrite == :none
                    # @. P_value =
                    #     P_ρe_factor * (
                    #         -Φ - norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ, Ic(Y.w)) / 2. +
                    #         LinearAlgebra.norm_sqr(Y.Yc.ρuₕ / Y.Yc.ρ)
                    #     )
                    # ∂P∂ρ = reshape(parent(P_value), N, M)
                    ∂P∂ρ = w_c = P_value
                    interp_c!(w_c, w)
                    @. ∂P∂ρ = P_ρe_factor * (-Φ + ((ρuₕ / ρ)^2 - w_c^2) / 2.)
                    @views @. J_𝕄ρ.d[2:N, :] +=
                        -∂P∂ρ[2:N, :] / (ρ_f[2:N, :] * Δz_f)
                    @views @. J_𝕄ρ.d2[1:N - 1, :] +=
                        ∂P∂ρ[1:N - 1, :] / (ρ_f[2:N, :] * Δz_f)
                end
            end
        end
    end
end

function hacky_view(J, m, is_upper, nrows, ncols)
    d = view(J.d, :, m)
    d2 = view(J.d2, :, m)
    GeneralBidiagonal{eltype(d), typeof(d)}(d, d2, is_upper, nrows, ncols)
end

function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        @unpack velem, helem, npoly, transform, dtγ_ref, J_ρ𝕄, J_𝔼𝕄, J_𝕄𝔼,
            J_𝕄ρ, S = A
        dtγ = dtγ_ref[]

        xρ = x.Yc.ρ
        bρ = b.Yc.ρ
        if :ρθ in propertynames(x.Yc)
            x𝔼 = x.Yc.ρθ
            b𝔼 = b.Yc.ρθ
        elseif :ρe_tot in propertynames(x.Yc)
            x𝔼 = x.Yc.ρe_tot
            b𝔼 = b.Yc.ρe_tot
        end
        if :ρw in propertynames(x)
            x𝕄 = x.ρw
            b𝕄 = b.ρw
        elseif :w in propertynames(x)
            x𝕄 = x.w
            b𝕄 = b.w
        end
        
        N = velem
        # TODO: Remove duplicate column computations.
        for i in 1:npoly + 1, h in 1:helem
            m = (h - 1) * (npoly + 1) + i
            schur_solve!(
                reshape(parent(Spaces.column(xρ, i, 1, h)), N),
                reshape(parent(Spaces.column(x𝔼, i, 1, h)), N),
                reshape(parent(Spaces.column(x𝕄, i, 1, h)), N + 1),
                hacky_view(J_ρ𝕄, m, true, N, N + 1),
                hacky_view(J_𝔼𝕄, m, true, N, N + 1),
                hacky_view(J_𝕄ρ, m, false, N + 1, N),
                hacky_view(J_𝕄𝔼, m, false, N + 1, N),
                reshape(parent(Spaces.column(bρ, i, 1, h)), N),
                reshape(parent(Spaces.column(b𝔼, i, 1, h)), N),
                reshape(parent(Spaces.column(b𝕄, i, 1, h)), N + 1),
                dtγ,
                S,
            )
        end

        @. x.Yc.ρuₕ = -b.Yc.ρuₕ

        if transform
            x .*= dtγ
        end
    end
end