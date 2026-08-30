# Advection over topography: an energy perturbation carried by a prescribed
# horizontal wind across a sinusoidal hill, in an isothermal atmosphere with
# gravity switched off (geopotential ≡ 0). Isolates the terrain-following metric terms from
# buoyancy effects, so errors from the coordinate transformation show up on
# their own.
using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaComms
ClimaComms.@import_required_backends
include("plane_utils.jl")

import ClimaCore:
    ClimaCore,
    slab,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Operators,
    Hypsography
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


function warp_surface(coord)
    # sin²(x) form ground elevation
    x = Geometry.component(coord, 1)
    FT = eltype(x)
    hc = FT(500.0)
    h = hc * FT(sin(π * x / 25000)^2)
    return h
end


# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space =
    hvspace_2D(
        (0, 25000),
        (0, 25000);
        xelem = 30,
        zelem = 30,
        warp_fn = warp_surface,
    )



geopotential(z) = 0.0

function init_advection_test(x, z)
    cv_d = C_v
    p_0 = MSLP
    # auxiliary quantities
    T = T_0
    p = p_0
    ρ = p / R_d / T # density
    e = cv_d * (T - T_0) + (sin(x * π / 12500))^2 / 2
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end
function init_velocity_profile(x, z)
    u = abs(sin(x * π / 12500))
    return Geometry.UVector.(u)
end

coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_advection_test(coord.x, coord.z), coords)
uₕ = map(coord -> init_velocity_profile(coord.x, coord.z), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
uₕ = Geometry.Covariant1Vector.(uₕ)
Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
# The flow at the sloped lower boundary must be tangent to it: contravariant
# u³ = g³¹u₁ + g³³u₃ = 0, that is u₃ = -g³¹u₁/g³³. That is a constraint on the
# state, not a tendency, so it is applied where the timestepper allows the
# state to be changed — after every stage, next to the DSS — rather than
# inside `rhs_invariant!`, which would mutate the stage value the integrator
# is still combining.
function project_surface_w!(Y)
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    face_level = Fields.level(
        Fields.local_geometry_field(hv_face_space),
        ClimaCore.Utilities.half,
    )
    u₁_bc = Fields.level(Ic2f.(Y.uₕ), ClimaCore.Utilities.half)
    gⁱʲ = face_level.gⁱʲ
    g13 = gⁱʲ.components.data.:3
    g33 = gⁱʲ.components.data.:9
    u₃_bc =
        Geometry.Covariant3Vector.(
            -1 .* g13 .* u₁_bc.components.data.:1 ./ g33,
        )
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. Y.w = apply_boundary_w(Y.w)
    return Y
end

Spaces.weighted_dss!(Yc)
Spaces.weighted_dss!(uₕ)
Spaces.weighted_dss!(w)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)
project_surface_w!(Y)

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

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Neumann condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )

    fω¹² = hcurl.(fw)
    fω¹² .+= vcurlc2f.(cuₕ)

    cω¹² = hcurl.(cw)
    cω¹² .+= If2c.(vcurlc2f.(cuₕ))

    # cross product
    fu =
        Geometry.Covariant13Vector.(Ic2f.(cuₕ)) .+
        Geometry.Covariant13Vector.(fw)
    fu¹² = Geometry.project.(Ref(Geometry.Contravariant1Axis()), fu)
    fu³ = Geometry.project.(Ref(Geometry.Contravariant3Axis()), fu)

    cu = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)
    cu³ = Geometry.project.(Ref(Geometry.Contravariant3Axis()), cu)
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    #@. duₕ -= If2c(fω¹²) × If2c(fu³)
    @. duₕ -= cω¹² × cu³

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. norm(cu)^2 / 2 + geopotential(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy
    @. dρe -= hwdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    # `w` at the surface is fixed by the free-slip terrain constraint applied
    # above, so the vertical momentum equation must not drive it off that
    # constraint.
    apply_boundary_dw = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Covariant3Vector(0.0)),
    )
    @. dw = apply_boundary_dw(dw)

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)
    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
import ClimaTimeSteppers as CTS
# Δx ≈ 210 m and the sound speed is ≈ 340 m/s, so Δt = 0.5 s sits right at the
# acoustic CFL limit and eventually goes unstable; halving it is comfortably
# inside the limit for the three-stage SSP scheme.
Δt = 0.25
# A `FieldVector` may hold non-`Field` entries, which need no DSS.
_dss!(x::Fields.Field) = Spaces.weighted_dss!(x)
_dss!(::Any) = nothing
function dss!(Y, parameters, t)
    foreach(_dss!, Fields._values(Y))
    project_surface_w!(Y)
end
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = rhs_invariant!, dss!),
    Y,
    (0.0, 15000.0),
    nothing,
)
integrator = CTS.init(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = Δt,
    saveat = collect(0.0:30.0:15000.0),
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev CTS.solve!(integrator)

# Mass and total energy must be conserved (measured drift: 8e-13 and 2e-9),
# and with gravity off, w can only be what the terrain forces: u·∇h ≈ 0.1 m/s
# (measured 0.089). Larger values mean the surface condition is leaking.
@test abs(sum(sol.u[end].Yc.ρ) - sum(sol.u[1].Yc.ρ)) / sum(sol.u[1].Yc.ρ) < 1e-10
@test abs(sum(sol.u[end].Yc.ρe) - sum(sol.u[1].Yc.ρe)) / sum(sol.u[1].Yc.ρe) < 1e-7
@test maximum(abs, parent(Geometry.WVector.(sol.u[end].w))) < 0.5

ENV["GKSwstype"] = "nul"
import Plots, ClimaCorePlots
Plots.GRBackend()

dir = "iso_channel_2d"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρe)
end
Plots.mp4(anim, joinpath(path, "total_energy.mp4"), fps = 20)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "density.mp4"), fps = 20)

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
