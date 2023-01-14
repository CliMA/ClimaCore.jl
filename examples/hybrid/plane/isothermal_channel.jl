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

function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    xelem = 30,
    zelem = 30,
    npoly = 4,
    warp_fn = warp_surface,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]);
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = xelem)
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)
    z_surface = warp_fn.(Fields.coordinate_field(horzspace))
    hv_face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )
    hv_center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    return (hv_center_space, hv_face_space)
end

# set up 2D domain - doubly periodic box
hv_center_space, hv_face_space = hvspace_2D((0, 25000), (0, 25000))

const MSLP = 1e5 # mean sea level pressure
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature


Φ(z) = 0.0

function init_advection_test(x, z)
    cp_d = C_p
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
# ==========
u₁_bc = Fields.level(Ic2f.(uₕ), ClimaCore.Utilities.half)
gⁱʲ =
    Fields.level(
        Fields.local_geometry_field(hv_face_space),
        ClimaCore.Utilities.half,
    ).gⁱʲ
g13 = gⁱʲ.components.data.:3
g11 = gⁱʲ.components.data.:1
g33 = gⁱʲ.components.data.:4
u₃_bc = Geometry.Covariant3Vector.(-1 .* g13 .* u₁_bc.components.data.:1 ./ g33)
apply_boundary_w =
    Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
@. w = apply_boundary_w(w)
# ==========
Spaces.weighted_dss!(Yc)
Spaces.weighted_dss!(uₕ)
Spaces.weighted_dss!(w)
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
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    dρ .= 0 .* cρ

    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)

    # Enforce no flux boundary condition on bottom `w`
    u₁_bc = Fields.level(Ic2f.(cuₕ), ClimaCore.Utilities.half)
    gⁱʲ =
        Fields.level(
            Fields.local_geometry_field(hv_face_space),
            ClimaCore.Utilities.half,
        ).gⁱʲ
    g13 = gⁱʲ.components.data.:3
    g11 = gⁱʲ.components.data.:1
    g33 = gⁱʲ.components.data.:4
    u₃_bc =
        Geometry.Covariant3Vector.(-1 .* g13 .* u₁_bc.components.data.:1 ./ g33)
    apply_boundary_w =
        Operators.SetBoundaryOperator(bottom = Operators.SetValue(u₃_bc))
    @. fw = apply_boundary_w(w)

    cw = If2c.(fw)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    ce = @. cρe / cρ
    cI = @. ce - Φ(z) - (norm(cuw)^2) / 2
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
    dρ .-= hdiv.(cρ .* (cuw))

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
    cu¹ = Geometry.project.(Ref(Geometry.Contravariant1Axis()), cu)
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

    cE = @. (norm(cuₕ)^2 + If2c(norm(fw)^2)) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy
    @. dρe -= hdiv(cuw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cρ))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)
    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
using OrdinaryDiffEq
Δt = 0.5
prob = ODEProblem(rhs_invariant!, Y, (0.0, 15000.0))
integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 30.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

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

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "energy_cons.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "mass_cons.png"), joinpath(@__DIR__, "../..")),
    "Mass",
)
