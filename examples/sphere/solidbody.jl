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



FT = Float64
const R = FT(6.37122e6)
const h0 = 1000.0
const r0 = R / 3
const center = Geometry.LatLongPoint(0.0, 270.0)

ne = 4
Nq = 4

domain = Domains.SphereDomain(R)
mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), ne)
grid_topology = Topologies.Grid2DTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)


coords = Fields.coordinate_field(space)


phi = map(coords) do coord
    n_coord = Geometry.components(Geometry.Cartesian123Point(coord))
    n_center = Geometry.components(Geometry.Cartesian123Point(center))
    # https://en.wikipedia.org/wiki/Great-circle_distance
    rd = R * atan(norm(n_coord × n_center), dot(n_coord, n_center))
    if rd < r0
        h0 / 2 * (1 + cospi(rd / r0))
    else
        0.0
    end
end


u = map(coords) do coord
    α0 = 0.0
    u0 = 2 * pi * R / 12
    θ = coord.lat
    λ = coord.long

    u = u0 * (cosd(α0) * cosd(θ) + sind(α0) * cosd(λ) * sind(θ))
    v = -u0 * sind(α0) * sind(λ)
    Geometry.UVVector(u, v)
end

div = Operators.Divergence()
divu = div.(u)
