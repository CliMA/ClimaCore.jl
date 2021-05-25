using Test
using StaticArrays, IntervalSets
import ClimateMachineCore: slab, Domains, Topologies, Meshes
import ClimateMachineCore.Geometry: Cartesian2DPoint

@testset "1×1 domain mesh" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = false,
    )
    discretization = Domains.EquispacedRectangleDiscretization(domain, 1, 1)
    grid_topology = Topologies.GridTopology(discretization)

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
        if i in (1, 4)
            @test 2 *
                  local_geometry_slab[i, j].invM *
                  local_geometry_slab[i, j].WJ ≈ 1
        else
            @test local_geometry_slab[i, j].invM *
                  local_geometry_slab[i, j].WJ ≈ 1
        end
    end

    @test length(mesh.boundary_surface_geometries) == 2
    @test keys(mesh.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(mesh.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(mesh.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end
