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
import ClimaCore.Domains.Geometry: Cartesian2DPoint
using ClimaCore.Geometry

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

helem = 75
velem = 10 # Use 20 if results are poor.
npoly = 4

# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 10,
    velem = 50,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(-0)..Geometry.YPoint{FT}(0),
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh = Meshes.EquispacedRectangleMesh(horzdomain, helem, 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# set up rhs!
hv_center_space, hv_face_space =
    hvspace_2D((-150000, 150000), (0, 10000), helem, velem, npoly)

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

function pressure(ρθ)
    if ρθ >= 0
        return MSLP * (R_d * ρθ / MSLP)^γ
    else
        return NaN
    end
end

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_inertial_gravity_wave(x, z)
    p_0 = MSLP
    g = grav
    cp_d = C_p
    x_c = 0.
    θ_0 = 300.
    Δθ = 0.01
    A = 5000.
    H = 10000.
    NBr = 0.01
    S = NBr * NBr / g

    p = p_0 * (1 - g / (cp_d * θ_0 * S) * (1 - exp(-S * z)))^(cp_d / R_d)
    θ = θ_0 * exp(z * S) + Δθ * sin(pi * z / H) / (1 + ((x - x_c) / A)^2)
    ρ = p / ((p / p_0)^(R_d / cp_d) * R_d * θ)
    ρθ = ρ * θ

    return (ρ = ρ, ρθ = ρθ, ρuₕ = ρ * Geometry.Cartesian1Vector(0.0))
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space);
face_coords = Fields.coordinate_field(hv_face_space);
Yc = map(coord -> init_inertial_gravity_wave(coord.x, coord.z), coords);
ρw = map(coord -> Geometry.Cartesian3Vector(0.0), face_coords);
Y = Fields.FieldVector(Yc = Yc, ρw = ρw);

function rhs!(dY, Y, _, t)
    ρw = Y.ρw
    Yc = Y.Yc
    dYc = dY.Yc
    dρw = dY.ρw

    # spectral horizontal operators
    hdiv = Operators.Divergence()

    # vertical FD operators with BC's
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
    )
    uvdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
        top = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
    )
    If_bc = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    )
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Ic = Operators.InterpolateF2C()
    ∂ = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    ∂f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )

    uₕ = @. Yc.ρuₕ / Yc.ρ
    w = @. ρw / If(Yc.ρ)
    p = @. pressure(Yc.ρθ)

    # density
    @. dYc.ρ = -∂(ρw)
    @. dYc.ρ -= hdiv(Yc.ρuₕ)

    # potential temperature
    @. dYc.ρθ = -(∂(ρw * If(Yc.ρθ / Yc.ρ)))
    @. dYc.ρθ -= hdiv(uₕ * Yc.ρθ)

    # horizontal momentum
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
            @SMatrix [1.0]
        ),
    )
    @. dYc.ρuₕ = -hdiv(Yc.ρuₕ ⊗ uₕ + p * Ih)
    @. dYc.ρuₕ -= uvdivf2c(ρw ⊗ If_bc(uₕ))

    # vertical momentum
    @. dρw = B(
        Geometry.transform(
            Geometry.Cartesian3Axis(),
            -(∂f(p)) - If(Yc.ρ) * ∂f(Φ(coords.z)),
        ) - vvdivc2f(Ic(ρw ⊗ w)),
    )
    uₕf = @. If_bc(Yc.ρuₕ / Yc.ρ) # requires boundary conditions
    @. dρw -= hdiv(uₕf ⊗ ρw)

    Spaces.weighted_dss!(dYc)
    Spaces.weighted_dss!(dρw)
    return dY
end

using OrdinaryDiffEq
Δt = 1.5
tspan = (0., 1000.)
prob = ODEProblem(rhs!, Y, tspan)
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10.,
    progress = true,
    progress_steps = 1,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "inertial_gravity_wave"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# post-processing
import Plots
θ_ref = 300. .* exp.(coords.z .* (0.01 * 0.01 / grav))
anim = Plots.@animate for u in sol.u
    Plots.plot(u.Yc.ρθ ./ u.Yc.ρ .- θ_ref, clim = (-0.002, 0.012))
end
Plots.mp4(anim, joinpath(path, "wave_Δθ_explicit.mp4"), fps = 20)
anim = Plots.@animate for u in sol.u
    Plots.plot(pressure.(u.Yc.ρθ) .- pressure.(u.Yc.ρ .* θ_ref), clim = (0., 3.))
end
Plots.mp4(anim, joinpath(path, "wave_Δp_explicit.mp4"), fps = 20)
