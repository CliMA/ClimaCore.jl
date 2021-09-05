using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimaCore: slab, Domains, Meshes, Topologies, Spaces
import ClimaCore.Geometry
import ClimaCore.Domains.Geometry: Cartesian2DPoint

@testset "1d domain space" begin
    domain = Domains.RectangleDomain(
        -3..5,
        0..0,
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace1D(grid_topology, quad)

    coord_data = Spaces.coordinates_data(space)
    @test eltype(coord_data) == Geometry.XPoint{Float64}

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 1, 1)
    coord_slab = slab(Spaces.coordinates_data(space), 1)
    @test coord_slab[1] == Geometry.XPoint(-3.0)
    @test coord_slab[4] == Geometry.XPoint(5.0)

    local_geometry_slab = slab(space.local_geometry, 1)
    dss_weights_slab = slab(space.dss_weights, 1)


    for i in 1:4
        @test Geometry.components(local_geometry_slab[i].∂x∂ξ) ≈
              @SMatrix [8 / 2]
        @test Geometry.components(local_geometry_slab[i].∂ξ∂x) ≈
              @SMatrix [2 / 8]
        @test local_geometry_slab[i].J ≈ (8 / 2)
        @test local_geometry_slab[i].WJ ≈ (8 / 2) * weights[i]
        if i in (1, 4)
            @test dss_weights_slab[i] ≈ 1 / 2
        else
            @test dss_weights_slab[i] ≈ 1
        end
    end

end


@testset "1×1 domain space" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = false,
        x2boundary = (:south, :north),
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = parent(Spaces.coordinates_data(space))
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(Spaces.coordinates_data(space), 1)
    @test coord_slab[1, 1] ≈ Cartesian2DPoint(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Cartesian2DPoint(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Cartesian2DPoint(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Cartesian2DPoint(5.0, 8.0)

    local_geometry_slab = slab(space.local_geometry, 1)
    dss_weights_slab = slab(space.dss_weights, 1)


    for i in 1:4, j in 1:4
        @test Geometry.components(local_geometry_slab[i, j].∂x∂ξ) ≈
              @SMatrix [8/2 0; 0 10/2]
        @test Geometry.components(local_geometry_slab[i, j].∂ξ∂x) ≈
              @SMatrix [2/8 0; 0 2/10]
        @test local_geometry_slab[i, j].J ≈ (10 / 2) * (8 / 2)
        @test local_geometry_slab[i, j].WJ ≈
              (10 / 2) * (8 / 2) * weights[i] * weights[j]
        if i in (1, 4)
            @test dss_weights_slab[i, j] ≈ 1 / 2
        else
            @test dss_weights_slab[i, j] ≈ 1
        end
    end

    @test length(space.boundary_surface_geometries) == 2
    @test keys(space.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(space.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(space.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end
