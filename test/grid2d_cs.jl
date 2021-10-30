using Test
import ClimaCore: Domains, Meshes, Topologies, Geometry
using ClimaCore.Meshes:
    equispaced_rectangular_mesh,
    cube_panel_mesh,
    AbstractWarp,
    EquidistantSphereWarp,
    EquiangularSphereWarp,
    NoWarp,
    Mesh2D
using StaticArrays
using IntervalSets

function cube_panel_topology(ne, ::Type{FT}) where {FT <: AbstractFloat}
    domain = Domains.CubePanelDomain{FT}()
    mesh = Mesh2D(domain, ne)
    grid_topology = Topologies.Grid2DTopology(mesh)
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
        @test Topologies.opposing_face(grid_topology, 1, 1) == (4, 4, true)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (3, 1, true)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (2, 1, true)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (5, 1, true)

        @test Topologies.opposing_face(grid_topology, 2, 1) == (1, 3, true)
        @test Topologies.opposing_face(grid_topology, 2, 2) == (3, 4, true)
        @test Topologies.opposing_face(grid_topology, 2, 3) == (6, 2, true)
        @test Topologies.opposing_face(grid_topology, 2, 4) == (5, 2, true)

        @test Topologies.opposing_face(grid_topology, 3, 1) == (1, 2, true)
        @test Topologies.opposing_face(grid_topology, 3, 2) == (4, 3, true)
        @test Topologies.opposing_face(grid_topology, 3, 3) == (6, 3, true)
        @test Topologies.opposing_face(grid_topology, 3, 4) == (2, 2, true)

        @test Topologies.opposing_face(grid_topology, 4, 1) == (5, 4, true)
        @test Topologies.opposing_face(grid_topology, 4, 2) == (6, 4, true)
        @test Topologies.opposing_face(grid_topology, 4, 3) == (3, 2, true)
        @test Topologies.opposing_face(grid_topology, 4, 4) == (1, 1, true)

        @test Topologies.opposing_face(grid_topology, 5, 1) == (1, 4, true)
        @test Topologies.opposing_face(grid_topology, 5, 2) == (2, 4, true)
        @test Topologies.opposing_face(grid_topology, 5, 3) == (6, 1, true)
        @test Topologies.opposing_face(grid_topology, 5, 4) == (4, 1, true)

        @test Topologies.opposing_face(grid_topology, 6, 1) == (5, 3, true)
        @test Topologies.opposing_face(grid_topology, 6, 2) == (2, 3, true)
        @test Topologies.opposing_face(grid_topology, 6, 3) == (3, 3, true)
        @test Topologies.opposing_face(grid_topology, 6, 4) == (4, 2, true)

        # 6 faces, 4 vertices per face, 8 global vertices, so each vertex should be part of 3 elements
        @test Topologies.vertex_coordinates(grid_topology, 1)[1] isa
              Geometry.Cartesian123Point
        # check that all vertices appear as part of 3 elements
        for vert in Topologies.vertices(grid_topology)
            @test length(vert) == 3
        end
    end
    @testset "24 element mesh with 4 elements per panel" begin
        _, _, grid_topology = cube_panel_topology(2, FT)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (13, 4, true)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (2, 4, true)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (3, 1, true)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (17, 1, true)
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
        @test faces[1] == (1, 4, 5, 1, true)
        @test faces[2] == (1, 3, 2, 1, true)
        @test faces[3] == (1, 2, 3, 1, true)
        @test faces[4] == (1, 1, 4, 4, true)
        @test faces[5] == (4, 1, 5, 4, true)
        @test faces[6] == (5, 2, 2, 4, true)
        @test faces[7] == (2, 2, 3, 4, true)
        @test faces[8] == (4, 3, 3, 2, true)
        @test faces[9] == (5, 3, 6, 1, true)
        @test faces[10] == (6, 2, 2, 3, true)
        @test faces[11] == (6, 3, 3, 3, true)
        @test faces[12] == (4, 2, 6, 4, true)
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
    FT = Float64
    @testset "4 elements per edge, equidistant spherical mesh of radius 10; Float type = Float64" begin
        radius = FT(10)
        domain = Domains.SphereDomain(radius)
        mesh = Mesh2D(domain, EquidistantSphereWarp(), 4)
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end
    @testset "4 elements per edge, equiangular spherical mesh of radius 10; Float type = Float64" begin
        radius = FT(10)
        domain = Domains.SphereDomain(radius)
        mesh = Mesh2D(domain, EquiangularSphereWarp(), 4)
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end

    FT = BigFloat
    @testset "4 elements per edge, equidistant spherical mesh of radius 10; Float type = BigFloat" begin
        radius = FT(10)
        domain = Domains.SphereDomain(radius)
        mesh = Mesh2D(domain, EquidistantSphereWarp(), 4)
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end
    @testset "4 elements per edge, equiangular spherical mesh of radius 10; Float type = BigFloat" begin
        radius = FT(10)
        domain = Domains.SphereDomain(radius)
        mesh = Mesh2D(domain, EquiangularSphereWarp(), 4)
        crad = abs.(sqrt.(sum(mesh.coordinates .^ 2, dims = 2)) .- radius)
        @test maximum(crad) ≤ 100 * eps(FT)
    end
end
