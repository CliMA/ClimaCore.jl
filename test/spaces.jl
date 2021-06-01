using Test
using StaticArrays, IntervalSets, LinearAlgebra

import ClimateMachineCore: slab, Domains, Meshes, Topologies, Spaces
import ClimateMachineCore.Domains.Geometry: Cartesian2DPoint

@testset "1×1 domain space" begin
    domain = Domains.RectangleDomain(
        -3..5,
        -2..8,
        x1periodic = true,
        x2periodic = false,
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, 1, 1)
    grid_topology = Topologies.GridTopology(mesh)

    quad = Spaces.Quadratures.GLL{4}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)

    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    array = getfield(space.coordinates, :array)
    @test size(array) == (4, 4, 2, 1)
    coord_slab = slab(space.coordinates, 1)
    @test coord_slab[1, 1] ≈ Cartesian2DPoint(-3.0, -2.0)
    @test coord_slab[4, 1] ≈ Cartesian2DPoint(5.0, -2.0)
    @test coord_slab[1, 4] ≈ Cartesian2DPoint(-3.0, 8.0)
    @test coord_slab[4, 4] ≈ Cartesian2DPoint(5.0, 8.0)

    local_geometry_slab = slab(space.local_geometry, 1)
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

    @test length(space.boundary_surface_geometries) == 2
    @test keys(space.boundary_surface_geometries) == (:south, :north)
    @test sum(parent(space.boundary_surface_geometries.north.sWJ)) ≈ 8
    @test parent(space.boundary_surface_geometries.north.normal)[1, :, 1] ≈
          [0.0, 1.0]
end

@testset "Column FiniteDifferenceSpace" begin
    for FT in (Float32, Float64)
        a = FT(0.0)
        b = FT(1.0)
        n = 10
        cs = Spaces.FaceFiniteDifferenceSpace(a, b, n)
        @test cs.cent.Δh[1] ≈ FT(1 / 10)
        @test cs.face.Δh[1] ≈ FT(1 / 10)
        @test cs.cent == Spaces.coords(cs, Spaces.CellCent())
        @test cs.face == Spaces.coords(cs, Spaces.CellFace())
        # n_cells = n_cells_real + 2*n_cells_ghost
        # n_faces = n_cells + 1 = (10 + 2*1) + 1
        @test Spaces.interior_face_range(cs) == 2:12
        @test Spaces.interior_cent_range(cs) == 2:11
        if FT === Float64
            @show cs.∇_cent_to_cent
        end
    end
    @test Spaces.n_hat(Spaces.ColumnMin()) == -1
    @test Spaces.binary(Spaces.ColumnMin()) == 0

    @test Spaces.n_hat(Spaces.ColumnMax()) == 1
    @test Spaces.binary(Spaces.ColumnMax()) == 1
end
