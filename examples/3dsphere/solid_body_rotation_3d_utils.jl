push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators,
    DataLayouts

using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33



global_logger(TerminalLogger())

include("implicit_solver_utils.jl")
include("ordinary_diff_eq_bug_fixes.jl")

const R = 6.4e6 # radius
const Ω = 7.2921e-5 # Earth rotation (radians / sec)
const z_top = 3.0e4 # height position of the model top
const grav = 9.8 # gravitational constant
const p_0 = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)
const cp_d = R_d * γ / (γ - 1)
const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height
# P = ρ * R_d * T = ρ * R_d * θ * (P / p_0)^(R_d / C_p) ==>
# (P / p_0)^(1 - R_d / C_p) = R_d / p_0 * ρθ ==>
# P = p_0 * (R_d / p_0)^γ * ρθ^γ
const P_ρθ_factor = p_0 * (R_d / p_0)^γ
# P = ρ * R_d * T = ρ * R_d * (ρe_int / ρ / C_v) = (γ - 1) * ρe_int
const P_ρe_factor = γ - 1




# geopotential
gravitational_potential(z) = grav * z
# Π(ρθ) = cp_d * (R_d * ρθ / p_0)^(R_d / cv_d)
pressure(ρθ) = (ρθ*R_d/p_0)^γ * p_0



const hdiv = Operators.Divergence()
const hwdiv = Operators.Divergence()
const hgrad = Operators.Gradient()
const hwgrad = Operators.Gradient()
const hcurl = Operators.Curl()
const hwcurl = Operators.Curl() # Operator.WeakCurl()
const If2c = Operators.InterpolateF2C()
const Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const vdivf2c = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
)
const vcurlc2f = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
)
const vgradc2f = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
)

# initial conditions for density and ρθ
function init_sbr_thermo(z)

    p = p_0 * exp(-z / H)

    ρ = p / (R_d * T_0)

    θ = T_0*(p_0/p)^(R_d/cp_d)

    return (ρ = ρ, ρθ = ρ*θ)
end

function rhs!(dY, Y, p, t)
    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation

    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)
    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))
    # 1.b) vertical divergence
    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation
    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary

    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. duₕ -= If2c(fω¹² × fu³)
    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))
    cp = @. pressure(cρθ)
    @. duₕ -= hgrad(cp) / cρ
    cK = @. (norm(cuvw)^2) / 2
    @. duₕ -= hgrad(cK + Φ)

    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= (vgradc2f(cK) + ∇Φ)

    # 3) ρθ
    @. dρθ -= hdiv(cuvw * cρθ)
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end




function rhs_remainder!(dY, Y, p, t)
    @info "Remainder part"
    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????



    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    
    # 2) Momentum equation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(
            Geometry.Covariant123Vector.(Ic2f.(cuₕ)),
        ) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    
    @. duₕ -= If2c(fω¹² × fu³)
    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))
    cp = @. pressure(cρθ)
    @. duₕ -= hgrad(cp) / cρ
    cK = @. (norm(cuvw)^2) / 2
    @. duₕ -= hgrad(cK + Φ)

    
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    
    @. dw -= vgradc2f(cK)
    


    # 3) ρθ
    @. dρθ -= hdiv(cuvw * cρθ)
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))


    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end



function rhs_implicit!(dY, Y, p, t)
    @info "Implicit part"
    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ # ρθ on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρθ .= 0 .* cρθ

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation

    

    # 1.b) vertical divergence
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # TODO implicit
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    cp = @. pressure(cρθ)
    @. dw -= vgradc2f(cp) / Ic2f(cρ)
    # @. dw -= R_d/cv_d * Ic2f(Π(cρθ)) * vgradc2f(cρθ) / Ic2f(cρ)
    @. dw -= ∇Φ

    # 3) ρθ
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))

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
    @info "construct Wfact!"
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
        
        @info "start solving Tri-diag"
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

        @info "finish solving Tri-diag"
        if transform
            x .*= dtγ
        end
    end
end





# set up function space
function sphere_3D(
    R = 6.4e6,
    zlim = (0, 30.0e3),
    helem = 4,
    zelem = 15,
    npoly = 5,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end



# temporary FieldVector broadcast and fill patches that speeds up solves by 2-3x
import Base: copyto!, fill!
using Base.Broadcast: Broadcasted, broadcasted, BroadcastStyle
transform_broadcasted(bc::Broadcasted{Fields.FieldVectorStyle}, symb, axes) =
    Broadcasted(bc.f, map(arg -> transform_broadcasted(arg, symb, axes), bc.args), axes)
transform_broadcasted(fv::Fields.FieldVector, symb, axes) =
    parent(getproperty(fv, symb))
transform_broadcasted(x, symb, axes) = x
@inline function Base.copyto!(
    dest::Fields.FieldVector,
    bc::Broadcasted{Fields.FieldVectorStyle},
)
    for symb in propertynames(dest)
        p = parent(getproperty(dest, symb))
        copyto!(p, transform_broadcasted(bc, symb, axes(p)))
    end
    return dest
end
function Base.fill!(a::Fields.FieldVector, x)
    for symb in propertynames(a)
        fill!(parent(getproperty(a, symb)), x)
    end
    return a
end