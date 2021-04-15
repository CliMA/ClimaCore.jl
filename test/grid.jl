using Test
import ClimateMachineCore.Domains
import ClimateMachineCore.Topologies

function rectangular_grid(n1, n2, x1periodic, x2periodic)
    domain = Domains.RectangleDomain(
        x1min = 0.0,
        x1max = 1.0,
        x2min = 0.0,
        x2max = 1.0,
        x1periodic = x1periodic,
        x2periodic = x2periodic,
    )
    discretiation = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
    grid_topology = Topologies.GridTopology(discretiation)
    return (domain, discretiation, grid_topology)
end

@testset "simple rectangular grid opposing face" begin

    @testset "assert coorect element numbering" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            0,
            1,
        )
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            2,
            1,
        )
    end

    @testset "assert correct face numbering" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            1,
            5,
        )
        @test_throws AssertionError Topologies.opposing_face(
            grid_topology,
            1,
            0,
        )
    end

    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (1, 2, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (1, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (1, 4, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (1, 3, false)

    end
    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, false)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (1, 2, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (1, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (0, 4, false)
    end
    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(1, 1, false, false)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (0, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (0, 2, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (0, 4, false)
    end
    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(2, 2, false, false)
        @test Topologies.opposing_face(grid_topology, 1, 1) == (0, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 2) == (2, 1, false)
        @test Topologies.opposing_face(grid_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(grid_topology, 1, 4) == (3, 3, false)
        @test Topologies.opposing_face(grid_topology, 2, 1) == (1, 2, false)
        @test Topologies.opposing_face(grid_topology, 2, 2) == (0, 2, false)
        @test Topologies.opposing_face(grid_topology, 2, 3) == (0, 3, false)
        @test Topologies.opposing_face(grid_topology, 2, 4) == (4, 3, false)
    end
end

@testset "simple rectangular grid interior faces iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test length(Topologies.interior_faces(grid_topology)) == 2
        faces = collect(Topologies.interior_faces(grid_topology))
        @test faces[1] == (1, 1, 1, 2, false)
        @test faces[2] == (1, 3, 1, 4, false)
    end
    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.interior_faces(grid_topology)) == 1
        faces = collect(Topologies.interior_faces(grid_topology))
        @test faces[1] == (1, 1, 1, 2, false)
    end
    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.interior_faces(grid_topology)) == 0
        faces = collect(Topologies.interior_faces(grid_topology))
        @test isempty(faces)
    end
    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(2, 2, false, false)
        @test length(Topologies.interior_faces(grid_topology)) == 4
        faces = collect(Topologies.interior_faces(grid_topology))
        @test faces[1] == (2, 1, 1, 2, false)
        @test faces[2] == (3, 3, 1, 4, false)
        @test faces[3] == (4, 1, 3, 2, false)
        @test faces[4] == (4, 3, 2, 4, false)
    end
end

@testset "simple rectangular grid boundry faces iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test length(Topologies.boundary_faces(grid_topology, 1)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 2)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 3)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 4)) == 0
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 1)))
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 2)))
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 3)))
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 4)))
    end
    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.boundary_faces(grid_topology, 1)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 2)) == 0
        @test length(Topologies.boundary_faces(grid_topology, 3)) == 1
        @test length(Topologies.boundary_faces(grid_topology, 4)) == 1
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 1)))
        @test isempty(collect(Topologies.boundary_faces(grid_topology, 2)))
        @test collect(Topologies.boundary_faces(grid_topology, 3)) == [(1, 3)]
        @test collect(Topologies.boundary_faces(grid_topology, 4)) == [(1, 4)]
    end
    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.boundary_faces(grid_topology, 1)) == 1
        @test length(Topologies.boundary_faces(grid_topology, 2)) == 1
        @test length(Topologies.boundary_faces(grid_topology, 3)) == 1
        @test length(Topologies.boundary_faces(grid_topology, 4)) == 1
        @test collect(Topologies.boundary_faces(grid_topology, 1)) == [(1, 1)]
        @test collect(Topologies.boundary_faces(grid_topology, 2)) == [(1, 2)]
        @test collect(Topologies.boundary_faces(grid_topology, 3)) == [(1, 3)]
        @test collect(Topologies.boundary_faces(grid_topology, 4)) == [(1, 4)]
    end
    @testset "2×3 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(2, 3, false, false)
        @test length(Topologies.boundary_faces(grid_topology, 1)) == 3
        @test length(Topologies.boundary_faces(grid_topology, 2)) == 3
        @test length(Topologies.boundary_faces(grid_topology, 3)) == 2
        @test length(Topologies.boundary_faces(grid_topology, 4)) == 2

        @test collect(Topologies.boundary_faces(grid_topology, 1)) ==
              [(1, 1), (3, 1), (5, 1)]
        @test collect(Topologies.boundary_faces(grid_topology, 2)) ==
              [(2, 2), (4, 2), (6, 2)]
        @test collect(Topologies.boundary_faces(grid_topology, 3)) ==
              [(1, 3), (2, 3)]
        @test collect(Topologies.boundary_faces(grid_topology, 4)) ==
              [(5, 4), (6, 4)]
    end
end

@testset "simple rectangular grid vertex iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, true)
        @test length(Topologies.vertices(grid_topology)) == 1
        V = collect(Topologies.vertices(grid_topology))
        @test V[1] isa Topologies.Vertex
        @test length(V[1]) == 4
        @test collect(V[1]) == [(1, 1), (1, 2), (1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, grid_topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.vertices(grid_topology)) == 2
        V = collect(Topologies.vertices(grid_topology))
        @test V[1] isa Topologies.Vertex
        @test length(V[1]) == 2
        @test collect(V[1]) == [(1, 1), (1, 2)]
        @test length(V[2]) == 2
        @test collect(V[2]) == [(1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.vertices(grid_topology)) == 4
        V = collect(Topologies.vertices(grid_topology))
        @test V[1] isa Topologies.Vertex
        @test length(V[1]) == 1
        @test collect(V[1]) == [(1, 1)]
        @test length(V[2]) == 1
        @test collect(V[2]) == [(1, 2)]
        @test length(V[3]) == 1
        @test collect(V[3]) == [(1, 3)]
        @test length(V[4]) == 1
        @test collect(V[4]) == [(1, 4)]
    end

    @testset "2×3 element quad mesh with non-periodic boundaries" begin
        _, _, grid_topology = rectangular_grid(2, 3, false, false)
        @test length(Topologies.vertices(grid_topology)) == 3 * 4
        V = collect(Topologies.vertices(grid_topology))
        @test length(V[1]) == 1
        @test collect(V[1]) == [(1, 1)]
    end
end
