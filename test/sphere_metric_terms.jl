using LinearAlgebra, IntervalSets, UnPack
import ClimaCore: Domains, Topologies, Meshes, Spaces
import ClimaCore.Meshes: EquiangularSphereWarp, Mesh2D

using Test

@testset "Surface area of a sphere" begin
    FT = Float64
    radius = FT(3)
    ne = 4
    Nq = 4
    Nqh = 7
    domain = Domains.SphereDomain(radius)
    mesh = Mesh2D(domain, EquiangularSphereWarp(), ne)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D_sphere(grid_topology, quad)

    @test sum(ones(space)) ≈ 4pi * radius^2 rtol = 1e-3
end
