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

# This example solves the shallow-water equations on a cubed-sphere manifold.
# This file contains two test cases:
# - One, called "steady_state", reproduces Test Case 2 in Williamson et al,
#   "A standard test set for numerical approximations to the shallow water
#   equations in spherical geometry", 1992. This test case gives the steady-state
#   solution to the non-linear shallow water equations. It consists of solid
#   body rotation or zonal flow with the corresponding geostrophic height field.
#   This can be run with an angle α that represents the angle between the north
#   pole and the center of the top cube panel.
# - The other one, called "mountain", reproduces Test Case 5 in the same
#   reference paper. It represents a zonal flow over an isolated mountain,
#   where the governing equations describe a global steady-state nonlinear
#   zonal geostrophic flow, with a corresponding geostrophic height field over
#   a non-uniform reference surface h_s.

# Physical parameters needed
const R = 6.37122e6
const Ω = 7.292e-5
const g = 9.80616
const D₄ = 1.0e16 # hyperdiffusion coefficient
const test_name = get(ARGS, 1, "steady_state") # default test case to run
const test_angle_name = get(ARGS, 2, "alpha0") # default test case to run
const steady_state_test_name = "steady_state"
const mountain_test_name = "mountain"
const alpha0_test_name = "alpha0"
const alpha45_test_name = "alpha45"

# Test-specific physical parameters
if test_angle_name == alpha45_test_name
    const α = 45.0
else # default test case, α = 0.0
    const α = 0.0
end

if test_name == mountain_test_name
    const u0 = 20.0
    const h0 = 5960
    const a = 20.0 # radius of conical mountain
    const λc = 90.0 # center of mountain long coord, shifted by 180 compared to the paper, because our λ ∈ [-180, 180] (in the paper it was 270, with λ ∈ [0, 360])
    const ϕc = 30.0 # center of mountain lat coord
    const h_s0 = 2e3
else # default case, steady-state test case
    const u0 = 2 * pi * R / (12 * 86400)
    const h0 = 2.94e4 / g
end

# Plot variables and auxiliary function
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()
dirname = "cg_sphere_shallowwater_$(test_name)"
dirname = "$(dirname)_$(test_angle_name)"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    # Buildkite-agent upload figpath
    # Link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

# Set up discretization
ne = 8
Nq = 4

domain = Domains.SphereDomain(R)
mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), ne)
grid_topology = Topologies.Grid2DTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

coords = Fields.coordinate_field(space)
const J = Fields.Field(space.local_geometry.J, space)

# Definition of Coriolis force field
f = map(Fields.local_geometry_field(space)) do local_geometry

    coord = local_geometry.coordinates
    ϕ = coord.lat
    λ = coord.long

    f = 2 * Ω * (-cosd(λ) * cosd(ϕ) * sind(α) + sind(ϕ) * cosd(α))

    # Technically this should be a WVector, but since we are only in a 2D space,
    # WVector, Contravariant3Vector, Covariant3Vector are all equivalent.
    # This _won't_ be true in 3D however!
    Geometry.Contravariant3Vector(f)
end

# Definition of bottom surface topography field
if test_name == mountain_test_name # define the non-uniform reference surface h_s
    h_s = map(Fields.coordinate_field(space)) do coord
        ϕ = coord.lat
        λ = coord.long
        r = sqrt(min(a^2, (λ - λc)^2 + (ϕ - ϕc)^2)) # positive branch
        h_s = h_s0 * (1 - r / a)
    end
else
    h_s = zeros(space)
end

function init_state(local_geometry)
    coord = local_geometry.coordinates

    ϕ = coord.lat
    λ = coord.long

    # Set initial state
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

function rhs!(dydt, y, parameters, t)
    f = parameters.f
    h_s = parameters.h_s

    div = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
    wcurl = Operators.WeakCurl()

    # Compute hyperviscosity first
    @. dydt.h = wdiv(grad(y.h))
    @. dydt.u = wgrad(div(y.u)) - Geometry.Covariant12Vector(wcurl(curl(y.u)))

    Spaces.weighted_dss!(dydt)

    @. dydt.h = -D₄ * wdiv(grad(dydt.h))
    @. dydt.u =
        -D₄ *
        (wgrad(div(dydt.u)) - Geometry.Covariant12Vector(wcurl(curl(dydt.u))))

    # Add in pieces
    @. begin
        dydt.h += -wdiv(y.h * y.u)
        dydt.u +=
            -grad(g * (y.h + h_s) + norm(y.u)^2 / 2) +
            Geometry.Covariant12Vector((J * (y.u × (f + curl(y.u)))))
    end
    Spaces.weighted_dss!(dydt)
    return dydt
end

# Set initial condition
y0 = init_state.(Fields.local_geometry_field(space))

# Set up RHS function
dydt = similar(y0)
rhs!(dydt, y0, (f = f, h_s = h_s), 0.0)

# Solve the ODE operator
T = 86400 * 2
dt = 10 * 60
prob = ODEProblem(rhs!, y0, (0.0, T), (f = f, h_s = h_s))
sol = solve(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

@info "Test case: $(test_name)"
if test_name == steady_state_test_name
    @info "  with α: $(test_angle_name)"
end
@info "Solution L₂ norm at T = 0: ", norm(y0.h)
@info "Solution L₂ norm at T = $(T): ", norm(sol.u[end].h)
@info "Fluid volume at T = 0: ", sum(y0.h)
@info "Fluid volume at T = $(T): ", sum(sol.u[end].h)

if test_name == steady_state_test_name # In this case, we use the IC as the reference exact solution
    @info "L2 error at T = $(T): ", norm(sol.u[end].h .- y0.h)
    Plots.png(Plots.plot(sol.u[end].h .- y0.h), joinpath(path, "error.png"))
    linkfig(
        relpath(joinpath(path, "error.png"), joinpath(@__DIR__, "../..")),
        "Absolute error in height",
    )
else # In this case, we only plot the latest output of the dynamic problem
    Plots.png(Plots.plot(sol.u[end].h), joinpath(path, "final_height.png"))
    linkfig(
        relpath(
            joinpath(path, "final_height.png"),
            joinpath(@__DIR__, "../.."),
        ),
        "Height field at the final time step",
    )
end
