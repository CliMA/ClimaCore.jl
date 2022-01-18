using Test
using LinearAlgebra

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

# Nonhydrostatic gravity wave
# Reference: https://climate.ucdavis.edu/pubs/UJ2012JCP.pdf Section 5.4

const R = 6.37122e6 # radius
const grav = 9.8 # gravitational constant
const Ω = 0.0 # Earth rotation (radians / sec)
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const κ = 2 / 7 # kappa
const γ = 1.4 # heat capacity ratio
const cp_d = R_d / κ # heat capacity at constant pressure
const cv_d = cp_d - R_d # heat capacity at constant volume
const T_tri = 273.16 # triple point temperature
const N = 0.01 # Brunt-Vaisala frequency
const S = grav^2 / cp_d / N^2
const T_0 = 300 # isothermal atmospheric temperature
const Δθ = 10.0 # maximum potential temperature perturbation
const R_t = R / 3 # width of the perturbation
const L_z = 20.0e3 # vertial wave length of the perturbation
const p_0 = 1.0e5 # reference pressure
const λ_c = 180.0 # center longitude of the cosine bell
const ϕ_c = 0.0 # center latitude of the cosine bell

# set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3),
    helem = 4,
    zelem = 12,
    npoly = 4,
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

Φ(z) = grav * z

function pressure(ρ, e, normuvw, z)
    I = e - Φ(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end

# set up 3D domain - spherical shell
hv_center_space, hv_face_space = sphere_3D(R, (0, 10.0e3), 5, 5, 4)

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

r(λ, ϕ) = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))

function initial_condition(ϕ, λ, z)
    rd = r(λ, ϕ)
    if rd < R_t
        s = 0.5 * (1 + cos(pi * rd / R_t))
    else
        s = 0.0
    end
    p = p_0 * (1 - S / T_0 + S / T_0 * exp(-N^2 * z / grav))^(cp_d / R_d)
    θ = T_0 * exp(N^2 * z / grav) + Δθ * s * sin(2 * pi * z / L_z)
    T = θ * (p / p_0)^κ
    ρ = p / R_d / T
    e = cv_d * (T - T_tri) + grav * z
    ρe = ρ * e

    return (ρ = ρ, ρe = ρe)
end

# Coriolis
const f =
    @. Geometry.Contravariant3Vector(Geometry.WVector(2 * Ω * sind(coords.lat)))

Yc = map(coord -> initial_condition(coord.lat, coord.long, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

function rhs!(dY, Y, _, t)
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe = Y.Yc.ρe # scalar on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()

    dρ .= 0 .* cρ

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients

    χe = @. dρe = hwdiv(hgrad(cρe / cρ))
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) - Geometry.Covariant12Vector(
            hwcurl(Geometry.Covariant3Vector(hcurl(cuₕ))),
        )

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    κ₄ = 1.0e17 # m^4/s
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) - Geometry.Covariant12Vector(
                hwcurl(Geometry.Covariant3Vector(hcurl(χuₕ))),
            )
        )

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
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
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    ce = @. cρe / cρ
    cp = @. pressure(cρ, ce, norm(cuvw), coords.z)

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuvw)^2) / 2 + Φ(coords.z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y)
rhs!(dYdt, Y, nothing, 0.0)

# run!
using OrdinaryDiffEq
# Solve the ODE
time_end = 600
dt = 3
prob = ODEProblem(rhs!, Y, (0.0, time_end))
sol = solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

@info "Solution L₂ norm at time t = 0: ", norm(Y.Yc.ρe)
@info "Solution L₂ norm at time t = $(time_end): ", norm(sol.u[end].Yc.ρe)

# TODO: visualization artifacts

# ENV["GKSwstype"] = "nul"
# using ClimaCorePlots, Plots
# Plots.GRBackend()
# dirname = "nonhydrostatic_gravity_wave"
# path = joinpath(@__DIR__, "output", dirname)
# mkpath(path)

# function linkfig(figpath, alt = "")
#     # buildkite-agent upload figpath
#     # link figure in logs if we are running on CI
#     if get(ENV, "BUILDKITE", "") == "true"
#         artifact_url = "artifact://$figpath"
#         print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
#     end
# end
