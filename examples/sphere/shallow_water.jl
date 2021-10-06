push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const R = 6.37122e6
const α = 0.0 # angel between the north pole and the center of the northern panel
const Ω = 7.292e-5
const u0 = 2 * pi * R / (12 * 86400)
const g = 9.80616
const h0 = 2.94e4 / g
const D₄ = 1.0e16

ne = 4
Nq = 4

domain = Domains.SphereDomain(R)
mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), ne)
grid_topology = Topologies.Grid2DTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

coords = Fields.coordinate_field(space)
const J = Fields.Field(space.local_geometry.J, space)

f = map(Fields.local_geometry_field(space)) do local_geometry
    coord = local_geometry.coordinates
    ϕ = coord.lat
    λ = coord.long

    f = 2 * Ω * (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))

    # technically this should be a WVector, but since we are only in a 2D space,
    # WVector, Contravariant3Vector, Covariant3Vector are all equivalent
    # this _won't_ be true in 3D however!
    Geometry.Contravariant3Vector(f)
end

function init_state(local_geometry)
    coord = local_geometry.coordinates

    ϕ = coord.lat
    λ = coord.long

    # set initial state
    h =
        h0 -
        (R * Ω * u0 + u0^2 / 2) / g *
        (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))^2
    uλ = u0 * (cosd(α) * cosd(ϕ) + sind(α) * cosd(λ) * sind(ϕ))
    uϕ = -u0 * sind(α) * sind(λ)

    u = Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.UVVector(uλ, uϕ),
        local_geometry,
    )

    return (h = h, u = u)
end

function rhs!(dydt, y, f, t)

    sdiv = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    # compute hyperviscosity first
    @. dydt.h = wdiv(grad(y.h))
    @. dydt.u =
        wgrad(sdiv(y.u)) -
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))

    Spaces.weighted_dss!(dydt)

    @. dydt.h = -D₄ * wdiv(grad(dydt.h))
    @. dydt.u =
        -D₄ * (
            wgrad(sdiv(dydt.u)) - Geometry.Covariant12Vector(
                wcurl(Geometry.Covariant3Vector(curl(dydt.u))),
            )
        )

    # add in pieces
    @. begin
        dydt.h += -wdiv(y.h * y.u)
        dydt.u +=
            -grad(g * y.h + norm(y.u)^2 / 2) +
            Geometry.Covariant12Vector((J * (y.u × (f + curl(y.u)))))
    end
    Spaces.weighted_dss!(dydt)
    return dydt
end

y0 = init_state.(Fields.local_geometry_field(space))
dydt = similar(y0)
rhs!(dydt, y0, f, 0.0)

# Solve the ODE operator
T = 86400 * 5
dt = 30 * 60
prob = ODEProblem(rhs!, y0, (0.0, T), f)
sol = solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

@info "Solution norm at T = 0: ", norm(y0.h)
@info "Solution norm at T = $(T): ", norm(sol.u[end].h)
@info "L2 error at T = $(T): ", norm(sol.u[end].h .- y0.h)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg_sphere_shallowwater"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

Plots.png(Plots.plot(sol.u[end].h .- y0.h), joinpath(path, "error.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/error.png", "Absolute error in height")
