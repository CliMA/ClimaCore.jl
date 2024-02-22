using Test
using ClimaCore: Geometry, Domains, Meshes, Topologies

# need to make sure mesh objects with different arrays but same contents give identical topologies
# https://github.com/CliMA/ClimaCore.jl/issues/1592

domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(10.0);
    boundary_tags = (:bottom, :top),
)
mesh1 = Meshes.IntervalMesh(domain, [Geometry.ZPoint(Float64(i)) for i in 0:10])
mesh2 = Meshes.IntervalMesh(domain, [Geometry.ZPoint(Float64(i)) for i in 0:10])

@test mesh1 !== mesh2
@test mesh1 == mesh2
@test isequal(mesh1, mesh2)
@test hash(mesh1) == hash(mesh2)

topology1 = Topologies.IntervalTopology(mesh1)
topology2 = Topologies.IntervalTopology(mesh2)

@test topology1 === topology2
