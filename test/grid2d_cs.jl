using Test
import ClimaCore: Domains, Meshes, Topologies
import ClimaCore.Geometry: Cartesian2DPoint
using ClimaCore.Meshes:
    equispaced_rectangular_mesh,
    cube_panel_mesh,
    EquidistantSphereMesh,
    EquiangularSphereMesh,
    sphere_mesh
using StaticArrays
using IntervalSets

function cube_panel_topology(ne, ::Type{FT}) where {FT <: AbstractFloat}
    domain = Domains.Unstructured2DDomain{FT}()
    mesh = cube_panel_mesh(ne, FT)
    grid_topology = Topologies.Grid2DTopology(mesh, domain)
    return (domain, mesh, grid_topology)
end

FT = Float64
_, _, grid_topology = cube_panel_topology(1, FT)

@testset "simple cube surface grid opposing face" begin

    @testset "assert correct element numbering" begin
        _, _, grid_topology = cube_panel_topology(1, FT)
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            0,
            1,
        )
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            7,
            1,
        )
    end

    @testset "6 element mesh with 1 element per panel" begin
        _, _, grid_topology = cube_panel_topology(1, FT)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (4, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (2, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (5, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (3, 3, false)

        @test Topologies.opposing_face(grid_topology, 2, 1) == (5, 2, false)
        @test Topologies.opposing_face(grid_topology, 2, 2) == (3, 2, false)
        @test Topologies.opposing_face(grid_topology, 2, 3) == (1, 2, false)
        @test Topologies.opposing_face(grid_topology, 2, 4) == (6, 2, false)

        @test Topologies.opposing_face(grid_topology, 3, 1) == (4, 2, false)
        @test Topologies.opposing_face(grid_topology, 3, 2) == (2, 2, false)
        @test Topologies.opposing_face(grid_topology, 3, 3) == (1, 4, false)
        @test Topologies.opposing_face(grid_topology, 3, 4) == (6, 4, false)

        @test Topologies.opposing_face(grid_topology, 4, 1) == (5, 1, false)
        @test Topologies.opposing_face(grid_topology, 4, 2) == (3, 1, false)
        @test Topologies.opposing_face(grid_topology, 4, 3) == (1, 1, false)
        @test Topologies.opposing_face(grid_topology, 4, 4) == (6, 1, false)

        @test Topologies.opposing_face(grid_topology, 5, 1) == (4, 1, false)
        @test Topologies.opposing_face(grid_topology, 5, 2) == (2, 1, false)
        @test Topologies.opposing_face(grid_topology, 5, 3) == (1, 3, false)
        @test Topologies.opposing_face(grid_topology, 5, 4) == (6, 3, false)

        @test Topologies.opposing_face(grid_topology, 6, 1) == (4, 4, false)
        @test Topologies.opposing_face(grid_topology, 6, 2) == (2, 4, false)
        @test Topologies.opposing_face(grid_topology, 6, 3) == (5, 4, false)
        @test Topologies.opposing_face(grid_topology, 6, 4) == (3, 4, false)
    end

    @testset "24 element mesh with 4 elements per panel" begin
        _, _, grid_topology = cube_panel_topology(2, FT)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (13, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (2, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (17, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (3, 3, false)
    end
end

@testset "cube surface grid interior faces iterator" begin
    @testset "all faces should be interior faces" begin
        _, _, grid_topology = cube_panel_topology(1, FT)
        @test length(Topologies.interior_faces(grid_topology)) ==
              grid_topology.mesh.nfaces
        _, _, grid_topology = cube_panel_topology(2, FT)
        @test length(Topologies.interior_faces(grid_topology)) ==
              grid_topology.mesh.nfaces
    end

    @testset "6 element mesh with 1 element per panel" begin
        _, _, grid_topology = cube_panel_topology(1, FT)
        faces = collect(Topologies.interior_faces(grid_topology))
        @test faces[1] == (1, 3, 5, 3, false)
        @test faces[2] == (1, 4, 3, 3, false)
        @test faces[3] == (5, 4, 6, 3, false)
        @test faces[4] == (6, 4, 3, 4, false)
        @test faces[5] == (1, 1, 4, 3, false)
        @test faces[6] == (1, 2, 2, 3, false)
        @test faces[7] == (4, 4, 6, 1, false)
        @test faces[8] == (6, 2, 2, 4, false)
        @test faces[9] == (4, 1, 5, 1, false)
        @test faces[10] == (5, 2, 2, 1, false)
        @test faces[11] == (4, 2, 3, 1, false)
        @test faces[12] == (3, 2, 2, 2, false)
    end
end


@testset "simple cube surface grid boundary faces iterator" begin
    @testset "24 element mesh with 4 elements per panel (no boundary faces for this topology)" begin
        _, _, grid_topology = cube_panel_topology(2, FT)
        @test length(Topologies.boundary_faces(grid_topology, 1)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 2)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 3)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 4)) == 0
    end
end

@testset "sphere mesh" begin
    @testset "4 elements per edge, equidistant spherical mesh of radius 10" begin
        radius = FT(10)
        mesh = sphere_mesh(4, radius, FT, EquidistantSphereMesh())
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end
    @testset "4 elements per edge, equiangular spherical mesh of radius 10" begin
        radius = FT(10)
        mesh = sphere_mesh(4, radius, FT, EquiangularSphereMesh())
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end
end
