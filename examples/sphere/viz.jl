push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets


ne = 16
Nq = 4

domain = Domains.SphereDomain(10.0)
mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), ne)
grid_topology = Topologies.Grid2DTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)


# unfortunately we can'y do this with Plots.jl directly:
# https://github.com/JuliaPlots/Plots.jl/issues/3480

using PlotlyJS

coords = Fields.coordinate_field(space)
X = Geometry.Cartesian123Point.(coords)
I, J, K = Spaces.triangulate(space)

mesh = mesh3d(
    x = vec(parent(X.x1)),
    y = vec(parent(X.x2)),
    z = vec(parent(X.x3)),
    i = I,
    j = J,
    k = K,
    intensity = vec(parent(coords.long)),
)


dirname = "viz"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)
savefig(plot(mesh), joinpath(path, "long.html"))

#=
function cart(uv::Geometry.UVVector, coord::Geometry.LatLongPoint)
    ϕ = coord.lat
    λ = coord.long
    G = @SMatrix [
        -sind(λ) -sind(ϕ) * cosd(λ);
         cosd(λ) -sind(ϕ) * sind(λ);
         0       cosd(ϕ)
    ]
    G * Geometry.components(uv)
end

V = cart.(uv, coords)

cone(
    x=vec(parent(X.x1)),
    y=vec(parent(X.x2)),
    z=vec(parent(X.x3)),
    u=vec(parent(map(v->v[1], V))),
    v=vec(parent(map(v->v[2], V))),
    w=vec(parent(map(v->v[3], V))),
    sizemode="absolute",
    sizeref=2,
    anchor="tip"
)
=#
