# Dry rising thermal bubble in 2D: a warm perturbation in a neutrally stratified
# atmosphere becomes buoyant and rolls up into a mushroom cloud. A standard
# check that the nonhydrostatic solver handles strong local buoyancy while
# conserving mass and energy. Written in vector-invariant form with total energy
# as the prognostic thermodynamic variable.
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using StaticArrays, IntervalSets, LinearAlgebra

import ClimaComms
ClimaComms.@import_required_backends
include("plane_utils.jl")

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space =
    hvspace_2D((-500, 500), (0, 1000); xelem = 10, zelem = 40)


geopotential(z) = grav * z

# Reference:
# https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml,
# Section 5a Prognostic thermodynamic variable: Total Energy
function init_dry_rising_bubble_2d(x, z)
    x_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    e = cv_d * (T - T_0) + g * z
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_dry_rising_bubble_2d(coord.x, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant1Vector(0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

energy_0 = sum(Y.Yc.ρe)
mass_0 = sum(Y.Yc.ρ)

function rhs_invariant!(dY, Y, _, t)

    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρe = Y.Yc.ρe

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.Divergence{Operators.WeakForm}()
    hgrad = Operators.Gradient()
    hwgrad = Operators.Gradient{Operators.WeakForm}()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    dρ .= 0 .* cρ

    cw = If2c.(fw)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    ce = @. cρe / cρ
    cI = @. ce - geopotential(z) - (norm(cuw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT

    h_tot = @. ce + cp / cρ # Total enthalpy at cell centers

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    χe = @. dρe = hwdiv(hgrad(h_tot)) # we store χe in dρe
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))

    Spaces.weighted_dss!(dρe)
    Spaces.weighted_dss!(duₕ)

    κ₄ = 100.0 # m^4/s
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ = -κ₄ * (hwgrad(hdiv(χuₕ)))

    # 1) Mass conservation
    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hwdiv.(cρ .* (cuw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # 1.c) vertical upwinding
    third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.Extrapolate(1),
        top = Operators.Extrapolate(1),
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
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    fω¹² = hcurl.(fw)
    fω¹² .+= vcurlc2f.(cuₕ)

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant1Vector.(Geometry.Covariant13Vector.(Ic2f.(cuₕ)))
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant13Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuw)^2) / 2 + geopotential(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hwdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c((Ic2f(cρ) * third_order_upwind_c2f(fw, (cρe + cp) / cρ)))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))


    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
import ClimaTimeSteppers as CTS
Δt = 0.04
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs_invariant!),
    Y,
    (0.0, 1200.0),
    nothing,
)
integrator = CTS.init(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:10.0:1200.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev CTS.solve!(integrator)

using Test
# The domain is closed, so mass and total energy must be conserved (measured
# drift: 1e-15 and 4e-13), and the warm bubble must rise at a few m/s
# (measured max|w| = 3.4) — neither stay still nor blow up.
@test abs(sum(sol.u[end].Yc.ρ) - sum(sol.u[1].Yc.ρ)) / sum(sol.u[1].Yc.ρ) < 1e-12
@test abs(sum(sol.u[end].Yc.ρe) - sum(sol.u[1].Yc.ρe)) / sum(sol.u[1].Yc.ρe) < 1e-10
@testset "bubble rise speed" begin
    w = maximum(abs, parent(Geometry.WVector.(sol.u[end].w)))
    @test 1 < w < 10
end

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "bubble_2d_invariant_rhoe"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol.u
    Plots.plot(Geometry.WVector.(Geometry.Covariant13Vector.(If2c.(u.w))))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(Geometry.UVector.(Geometry.Covariant13Vector.(u.uₕ)))
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

# post-processing
Es = [sum(u.Yc.ρe) for u in sol.u]
Mass = [sum(u.Yc.ρ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy_cons.png"),
)
Plots.png(
    Plots.plot((Mass .- mass_0) ./ mass_0),
    joinpath(path, "mass_cons.png"),
)

include(joinpath(@__DIR__, "../..", "example_utils.jl")) # linkfig

linkfig(
    relpath(joinpath(path, "energy_cons.png"), joinpath(@__DIR__, "../../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass_cons.png"), joinpath(@__DIR__, "../../..")),
    "Mass",
)
