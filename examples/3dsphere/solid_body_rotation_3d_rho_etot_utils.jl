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
    Operators
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

global_logger(TerminalLogger())

const R = 6.4e6 # radius
const Ω = 7.2921e-5 # Earth rotation (radians / sec)
const z_top = 3.0e4 # height position of the model top
const grav = 9.8 # gravitational constant
const p_0 = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)

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

# pressure
function pressure(ρ, e_tot, normuvw, z)
    I = e_tot - gravitational_potential(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end



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

# initial conditions for density and total energy
function init_sbr_thermo(z)

    p = p_0 * exp(-z / H)
    ρ = 1 / R_d / T_0 * p

    T = p / ρ / R_d

    e = cv_d * (T - T_tri) + gravitational_potential(z)
    ρe_tot = ρ * e

    return (ρ = ρ, ρe_tot = ρe_tot)
end

function rhs!(dY, Y, p, t)

    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dY.Yc.ρe_tot

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    # hdiv = Operators.Divergence()
    # hwdiv = Operators.Divergence()
    # hgrad = Operators.Gradient()
    # hwgrad = Operators.Gradient()
    # hcurl = Operators.Curl()
    # hwcurl = Operators.Curl() # Operator.WeakCurl()

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))


    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

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

    ce_tot = @. cρe_tot / cρ
    cp = @. pressure(cρ, ce_tot, norm(cuvw), c_coords.z)
    cE = @. (norm(cuvw)^2) / 2 + Φ

    @. duₕ -= hgrad(cp) / cρ
    @. duₕ -= hgrad(cE)

    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= vgradc2f(cE)

    # 3) total energy


    @. dρe_tot -= hdiv(cuvw * (cρe_tot + cp))
    @. dρe_tot -= vdivf2c(Ic2f(cuₕ * (cρe_tot + cp)))
    @. dρe_tot -= vdivf2c(fw * Ic2f(cρe_tot + cp))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY

end


function rhs_remainder!(dY, Y, p, t)

    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dY.Yc.ρe_tot

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot


    # uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), cuₕ)
    # w_phy = Geometry.transform.(Ref(Geometry.WAxis()), fw)
    # @info "maximum vertical velocity is w, u_h", maximum(abs.(w_phy.components.data.:1)), maximum(abs.(uₕ_phy.components.data.:1)), maximum(abs.(uₕ_phy.components.data.:2))


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

    ce_tot = @. cρe_tot / cρ
    cp = @. pressure(cρ, ce_tot, norm(cuvw), c_coords.z)
    cK = @. (norm(cuvw)^2) / 2

    @. duₕ -= hgrad(cp) / cρ
    @. duₕ -= hgrad(cK + Φ)

    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    # @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= vgradc2f(cK)

    # 3) total energy

    @. dρe_tot -= hdiv(cuvw * (cρe_tot+cp))
    @. dρe_tot -= vdivf2c(Ic2f(cuₕ * (cρe_tot+cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY

end


function rhs_implicit!(dY, Y, p, t)
    @unpack P, Φ, ∇Φ = p

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dY.Yc.ρe_tot

    # uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), cuₕ)
    # w_phy = Geometry.transform.(Ref(Geometry.WAxis()), fw)
    # @info "maximum vertical velocity is w, u_h", maximum(abs.(w_phy.components.data.:1)), maximum(abs.(uₕ_phy.components.data.:1)), maximum(abs.(uₕ_phy.components.data.:2))




    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot

    # hyperdiffusion not needed in SBR

    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    ce_tot = @. cρe_tot / cρ
    cp = @. pressure(cρ, ce_tot, norm(cuvw), c_coords.z)
    @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= ∇Φ

    # 3) total energy
    @. dρe_tot -= vdivf2c(fw * Ic2f(cρe_tot + cp))


    return dY

end

include("solid_body_rotation_3d_implicit.jl")