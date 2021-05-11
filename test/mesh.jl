using Test
using StaticArrays
import ClimateMachineCore: slab, Domains, Topologies, Meshes
import ClimateMachineCore.Geometry: Cartesian2DPoint

@testset "1×1 domain mesh" begin
    domain = Domains.RectangleDomain(
        x1min = -3.0,
        x1max = 5.0,
        x2min = -2.0,
        x2max = 8.0,
        x1periodic = false,
        x2periodic = false,
    )
    discretiation = Domains.EquispacedRectangleDiscretization(domain, 1, 1)
    grid_topology = Topologies.GridTopology(discretiation)

    quad = Meshes.Quadratures.GLL{4}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)

    mesh = Meshes.Mesh2D(grid_topology, quad)

    array = getfield(mesh.coordinates, :array)
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(mesh.coordinates, 1)
    @test coord_slab[1, 1] ≈ Cartesian2DPoint(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Cartesian2DPoint(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Cartesian2DPoint(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Cartesian2DPoint(5.0, 8.0)

    local_geometry_slab = slab(mesh.local_geometry, 1)
    for i in 1:4, j in 1:4
        @test local_geometry_slab[i, j].∂ξ∂x ≈ @SMatrix [2/8 0; 0 2/10]
        @test local_geometry_slab[i, j].J ≈ (10 / 2) * (8 / 2)
        @test local_geometry_slab[i, j].WJ ≈
              (10 / 2) * (8 / 2) * weights[i] * weights[j]
        @test local_geometry_slab[i, j].invM * local_geometry_slab[i, j].WJ ≈ 1
    end
end
