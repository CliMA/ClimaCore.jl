using Test
using LinearAlgebra, UnPack

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

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# test specifications
const test_name = "baroclinic_wave"

# parameters
const R = 6.371229e6 # radius
const grav = 9.80616 # gravitational constant
const Ω = 7.29212e-5 # Earth rotation (radians / sec)
const R_d = 287.0 # R dry (gas constant / mol mass dry air)
const κ = 2 / 7 # kappa
const γ = 1.4 # heat capacity ratio
const cp_d = R_d / κ # heat capacity at constant pressure
const cv_d = cp_d - R_d # heat capacity at constant volume
const p_0 = 1.0e5 # reference pressure
const k = 3
const T_e = 310 # temperature at the equator
const T_p = 240 # temperature at the pole
const T_0 = 0.5 * (T_e + T_p)
const T_tri = 273.16 # triple point temperature
const Γ = 0.005
const A = 1 / Γ
const B = (T_0 - T_p) / T_0 / T_p
const C = 0.5 * (k + 2) * (T_e - T_p) / T_e / T_p
const b = 2
const H = R_d * T_0 / grav
const z_t = 15.0e3
const λ_c = 20.0
const ϕ_c = 40.0
const d_0 = R / 6
const V_p = 1.0

τ_z_1(z) = exp(Γ * z / T_0)
τ_z_2(z) = 1 - 2 * (z / b / H)^2
τ_z_3(z) = exp(-(z / b / H)^2)
τ_1(z) = 1 / T_0 * τ_z_1(z) + B * τ_z_2(z) * τ_z_3(z)
τ_2(z) = C * τ_z_2(z) * τ_z_3(z)
τ_int_1(z) = A * (τ_z_1(z) - 1) + B * z * τ_z_3(z)
τ_int_2(z) = C * z * τ_z_3(z)
F_z(z) = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
I_T(ϕ) = cosd(ϕ)^k - k / (k + 2) * (cosd(ϕ))^(k + 2)
temp(ϕ, z) = (τ_1(z) - τ_2(z) * I_T(ϕ))^(-1)
pres(ϕ, z) = p_0 * exp(-grav / R_d * (τ_int_1(z) - τ_int_2(z) * I_T(ϕ)))
r(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
U(ϕ, z) =
    grav * k / R * τ_int_2(z) * temp(ϕ, z) * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
u(ϕ, z) = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U(ϕ, z))
v(ϕ, z) = 0.0
c3(λ, ϕ) = cos(π * r(λ, ϕ) / 2 / d_0)^3
s1(λ, ϕ) = sin(π * r(λ, ϕ) / 2 / d_0)
cond(λ, ϕ) = (0 < r(λ, ϕ) < d_0) * (r(λ, ϕ) != R * pi)
if test_name == "baroclinic_wave"
    δu(λ, ϕ, z) =
        -16 * V_p / 3 / sqrt(3) *
        F_z(z) *
        c3(λ, ϕ) *
        s1(λ, ϕ) *
        (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
        sin(r(λ, ϕ) / R) * cond(λ, ϕ)
    δv(λ, ϕ, z) =
        16 * V_p / 3 / sqrt(3) *
        F_z(z) *
        c3(λ, ϕ) *
        s1(λ, ϕ) *
        cosd(ϕ_c) *
        sind(λ - λ_c) / sin(r(λ, ϕ) / R) * cond(λ, ϕ)
    const κ₄ = 1.0e17 # m^4/s
elseif test_name == "balanced_flow"
    δu(λ, ϕ, z) = 0.0
    δv(λ, ϕ, z) = 0.0
    const κ₄ = 0.0
end
uu(λ, ϕ, z) = u(ϕ, z) + δu(λ, ϕ, z)
uv(λ, ϕ, z) = v(ϕ, z) + δv(λ, ϕ, z)

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
const hwdiv = Operators.WeakDivergence()
const hgrad = Operators.Gradient()
const hwgrad = Operators.WeakGradient()
const hcurl = Operators.Curl()
const hwcurl = Operators.WeakCurl()
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
function initial_condition(ϕ, λ, z)
    ρ = pres(ϕ, z) / R_d / temp(ϕ, z)
    e =
        cv_d * (temp(ϕ, z) - T_tri) +
        gravitational_potential(z) +
        (uu(λ, ϕ, z)^2 + uv(λ, ϕ, z)^2) / 2
    ρe = ρ * e

    return (ρ = ρ, ρe_tot = ρe)
end

# initial conditions for velocity
function initial_condition_velocity(local_geometry)
    coord = local_geometry.coordinates
    ϕ = coord.lat
    λ = coord.long
    z = coord.z
    return Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.UVVector(uu(λ, ϕ, z), uv(λ, ϕ, z)),
        local_geometry,
    )
end

function rhs!(dY, Y, parameters, t)
    @unpack P, Φ, ∇Φ = parameters

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dYc = dY.Yc
    dρ = dYc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dYc.ρe_tot
    z = c_coords.z

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot

    # 0) compute hyperviscosity coefficients

    χe_tot = @. dρe_tot = hwdiv(hgrad(cρe_tot / cρ))
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(dρe_tot)
    Spaces.weighted_dss!(duₕ)

    @. dρe_tot = -κ₄ * hwdiv(cρ * hgrad(χe_tot))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(χuₕ))),
            )
        )

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
    cp = @. pressure(cρ, ce_tot, norm(cuvw), z)
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

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(duₕ)
    Spaces.weighted_dss!(dw)

    return dY

end


function rhs_remainder!(dY, Y, parameters, t)

    @unpack P, Φ, ∇Φ = parameters

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dYc = dY.Yc
    dρ = dYc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dYc.ρe_tot
    z = c_coords.z

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot

    # 0) compute hyperviscosity coefficients

    χe_tot = @. dρe_tot = hwdiv(hgrad(cρe_tot / cρ))
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(dρe_tot)
    Spaces.weighted_dss!(duₕ)

    @. dρe_tot = -κ₄ * hwdiv(cρ * hgrad(χe_tot))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(χuₕ))),
            )
        )

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
    cp = @. pressure(cρ, ce_tot, norm(cuvw), z)
    cK = @. (norm(cuvw)^2) / 2

    @. duₕ -= hgrad(cp) / cρ
    @. duₕ -= hgrad(cK + Φ)

    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. dw -= vgradc2f(cK)

    # 3) total energy

    @. dρe_tot -= hdiv(cuvw * (cρe_tot + cp))
    @. dρe_tot -= vdivf2c(Ic2f(cuₕ * (cρe_tot + cp)))

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(duₕ)
    Spaces.weighted_dss!(dw)

    return dY

end


function rhs_implicit!(dY, Y, parameters, t)
    @unpack P, Φ, ∇Φ = parameters

    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe_tot = Y.Yc.ρe_tot # total energy on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe_tot = dY.Yc.ρe_tot
    z = c_coords.z

    dρ .= 0 .* cρ
    dw .= 0 .* fw
    duₕ .= 0 .* cuₕ
    dρe_tot .= 0 .* cρe_tot

    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    ce_tot = @. cρe_tot / cρ
    cp = @. pressure(cρ, ce_tot, norm(cuvw), z)
    @. dw -= vgradc2f(cp) / Ic2f(cρ)
    @. dw -= ∇Φ

    # 3) total energy
    @. dρe_tot -= vdivf2c(fw * Ic2f(cρe_tot + cp))

    return dY

end

include("implicit_3d_sphere_utils.jl")
