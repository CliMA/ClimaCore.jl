import ClimaCore:
        DataLayouts,
        Geometry

# P = ρ * R_d * T = ρ * R_d * θ * (P / MSLP)^(R_d / C_p) ==>
# (P / MSLP)^(1 - R_d / C_p) = R_d / MSLP * ρθ ==>
# P = MSLP * (R_d / MSLP)^γ * ρθ^γ
const P_ρθ_factor = p_0 * (R_d / p_0)^γ
# P = ρ * R_d * T = ρ * R_d * (ρe_int / ρ / C_v) = (γ - 1) * ρe_int
const P_ρe_factor = γ - 1




# # vertical operators
# const If = Operators.InterpolateC2F(
#     bottom = Operators.Extrapolate(),
#     top = Operators.Extrapolate(),
# )
# const If_uₕ = Operators.InterpolateC2F(
#     bottom = Operators.SetValue(Geometry.UVector(0.0)),
#     top = Operators.SetValue(Geometry.UVector(0.0)),
# )
# const Ic = Operators.InterpolateF2C()
# const ∇◦ᵥf = Operators.DivergenceC2F()
# const ∇◦ᵥc = Operators.DivergenceF2C()
# const ∇ᵥf = Operators.GradientC2F()
# const B_w = Operators.SetBoundaryOperator(
#     bottom = Operators.SetValue(Geometry.WVector(0.0)),
#     top = Operators.SetValue(Geometry.WVector(0.0)),
# )



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
    Δξ₃::AT1
    J::AT1
    g³³::AT1
    Δξ₃_f::AT1
    J_f::AT1
    g³³_f::AT1

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
    center_local_geometry,
    face_local_geometry,
    transform,
    J_𝕄ρ_overwrite;
    FT = Float64,
)
    N = velem
    # cubed sphere
    M = 6 * helem^2 * (npoly + 1)^2

    dtγ_ref = Ref(zero(FT))

    J = reshape(parent(center_local_geometry.J), N , M)
    Δξ₃ = similar(J); fill!(Δξ₃, 1)
    g³³ = reshape(parent(center_local_geometry.∂x∂ξ)[:,:,:,end,:], N , M)
    J_f = reshape(parent(face_local_geometry.J), N + 1, M)
    Δξ₃_f = similar(J_f); fill!(Δξ₃_f, 1)
    g³³_f = reshape(parent(face_local_geometry.∂x∂ξ)[:,:,:,end,:], N + 1, M)

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
        ρ_f = similar(J_f),
        𝔼_value_f = similar(J_f),
        P_value = similar(J),
    )

    CustomWRepresentation{
        typeof(dtγ_ref),
        typeof(Δξ₃),
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
        Δξ₃,
        J,
        g³³,
        Δξ₃_f,
        J_f,
        g³³_f,
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
    @unpack velem, helem, npoly, dtγ_ref, Δξ₃, J, g³³, Δξ₃_f, J_f, g³³_f, J_ρ𝕄, J_𝔼𝕄, J_𝕄𝔼, J_𝕄ρ,
        J_𝕄ρ_overwrite, vals = W
    @unpack ρ_f, 𝔼_value_f, P_value = vals
    @unpack P, Φ, ∇Φ = p
    N = velem
    M = 6*helem^2 * (npoly + 1)^2
    # ∇Φ = ∂Φ/∂ξ³
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
        # TODO check 
        @views @. J_ρ𝕄.d = ρ_f[1:N, :] * J_f[1:N, :] * g³³_f[1:N, :] / (J * Δξ³)
        @views @. J_ρ𝕄.d2 = -ρ_f[2:N + 1, :] * J_f[2:N + 1, :] * g³³_f[2:N + 1, :] / (J * Δξ³)
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
            # TODO check
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
    # TODO check
    @views @. J_𝔼𝕄.d = 𝔼_value_f[1:N, :] * J_f[1:N, :] * g³³_f[1:N, :] / (J * Δξ³)
    @views @. J_𝔼𝕄.d2 = -𝔼_value_f[2:N + 1, :] * J_f[2:N + 1, :] * g³³_f[2:N + 1, :] / (J * Δξ³)

    # dY.𝕄 = B_w(...) ==>
    # ∂𝕄[1]/∂t = ∂𝕄[N + 1]/∂t = 0 ==>
    #     ∂(∂𝕄[1]/∂t)/∂ρ[1] = ∂(∂𝕄[1]/∂t)/∂𝔼[1] =
    #     ∂(∂𝕄[N + 1]/∂t)/∂ρ[N] = ∂(∂𝕄[N + 1]/∂t)/∂𝔼[N] = 0
    @. J_𝕄ρ.d[1, :] = J_𝕄𝔼.d[1, :] = J_𝕄ρ.d2[N, :] = J_𝕄𝔼.d2[N, :] = 0.

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
            # TODO check
            @views @. J_𝕄𝔼.d[2:N, :] = -∂P∂𝔼[2:N, :] / (ρ_f[2:N, :] * Δξ³³_f[2:N, :])
            @views @. J_𝕄𝔼.d2[1:N - 1, :] =
                ∂P∂𝔼[1:N - 1, :] / (ρ_f[2:N, :] * Δξ³³_f[2:N, :])

            if J_𝕄ρ_overwrite == :grav
                # TODO check
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

            if J_𝕄ρ_overwrite == :grav
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    -∇Φ[2:N, :] / (2. * ρ_f[2:N, :])
            elseif J_𝕄ρ_overwrite == :none || J_𝕄ρ_overwrite == :pres
                # P = reshape(parent(P), N, M)
                @views @. J_𝕄ρ.d[2:N, :] = J_𝕄ρ.d2[1:N - 1, :] =
                    (P[2:N, :] - P[1:N - 1, :]) / (2. * ρ_f[2:N, :]^2 * Δz_f)
                if J_𝕄ρ_overwrite == :none

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
        # TODO: numbering
        for i in 1:npoly + 1, j in 1:npoly + 1, h in 1:6*helem^2
            m = (h - 1) * (npoly + 1)^2 + (j-1)*(npoly + 1) + i
            schur_solve!(
                reshape(parent(Spaces.column(xρ, i, j, 1, h)), N),
                reshape(parent(Spaces.column(x𝔼, i, j, 1, h)), N),
                reshape(parent(Spaces.column(x𝕄, i, j, 1, h)), N + 1),
                hacky_view(J_ρ𝕄, m, true, N, N + 1),
                hacky_view(J_𝔼𝕄, m, true, N, N + 1),
                hacky_view(J_𝕄ρ, m, false, N + 1, N),
                hacky_view(J_𝕄𝔼, m, false, N + 1, N),
                reshape(parent(Spaces.column(bρ, i, j, 1, h)), N),
                reshape(parent(Spaces.column(b𝔼, i, j, 1, h)), N),
                reshape(parent(Spaces.column(b𝕄, i, j, 1, h)), N + 1),
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