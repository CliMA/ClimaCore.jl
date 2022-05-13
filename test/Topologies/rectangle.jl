using Test
import ClimaCore: Domains, Meshes, Topologies
import ClimaCore.Geometry: Geometry
using StaticArrays
using IntervalSets

function rectangular_grid(
    n1,
    n2,
    x1periodic,
    x2periodic;
    x1min = 0.0,
    x1max = 1.0,
    x2min = 0.0,
    x2max = 1.0,
)
    domain = Domains.RectangleDomain(
        Geometry.XPoint(x1min) .. Geometry.XPoint(x1max),
        Geometry.YPoint(x2min) .. Geometry.YPoint(x2max),
        x1periodic = x1periodic,
        x2periodic = x2periodic,
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    return Topologies.Topology2D(mesh)
end

@testset "opposing face tests" begin
    @testset "assert correct element numbering" begin
        topology = rectangular_grid(1, 1, true, true)
        @test_throws BoundsError Topologies.opposing_face(topology, 0, 1)
        @test_throws BoundsError Topologies.opposing_face(topology, 2, 1)
    end

    @testset "assert correct face numbering" begin
        topology = rectangular_grid(1, 1, true, true)
        @test_throws AssertionError Topologies.opposing_face(topology, 1, 5)
        @test_throws AssertionError Topologies.opposing_face(topology, 1, 0)
    end

    @testset "1×1 element quad mesh with all periodic boundries" begin
        topology = rectangular_grid(1, 1, true, true)

        @test Topologies.opposing_face(topology, 1, 1) == (1, 3, true)
        @test Topologies.opposing_face(topology, 1, 2) == (1, 4, true)
        @test Topologies.opposing_face(topology, 1, 3) == (1, 1, true)
        @test Topologies.opposing_face(topology, 1, 4) == (1, 2, true)
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        topology = rectangular_grid(1, 1, true, false)
        @test Topologies.opposing_face(topology, 1, 1) == (0, 1, false)
        @test Topologies.opposing_face(topology, 1, 2) == (1, 4, true)
        @test Topologies.opposing_face(topology, 1, 3) == (0, 2, false)
        @test Topologies.opposing_face(topology, 1, 4) == (1, 2, true)
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(1, 1, false, false)
        @test Topologies.opposing_face(topology, 1, 1) == (0, 3, false)
        @test Topologies.opposing_face(topology, 1, 2) == (0, 2, false)
        @test Topologies.opposing_face(topology, 1, 3) == (0, 4, false)
        @test Topologies.opposing_face(topology, 1, 4) == (0, 1, false)
    end

    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 2, false, false)
        @test Topologies.opposing_face(topology, 1, 1) == (0, 3, false)
        @test Topologies.opposing_face(topology, 1, 2) == (2, 4, true)
        @test Topologies.opposing_face(topology, 1, 3) == (3, 1, true)
        @test Topologies.opposing_face(topology, 1, 4) == (0, 1, false)
        @test Topologies.opposing_face(topology, 2, 1) == (0, 3, false)
        @test Topologies.opposing_face(topology, 2, 2) == (0, 2, false)
        @test Topologies.opposing_face(topology, 2, 3) == (4, 1, true)
        @test Topologies.opposing_face(topology, 2, 4) == (1, 2, true)
    end
end

@testset "neighboring element tests" begin

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        topology = rectangular_grid(1, 1, true, false)
        @test Topologies.local_neighboring_elements(topology, 1) == []
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(1, 1, false, false)
        @test Topologies.local_neighboring_elements(topology, 1) == []
    end

    @testset "2×2 element quad mesh with periodic boundaries" begin
        topology = rectangular_grid(2, 2, true, true)
        @test Topologies.local_neighboring_elements(topology, 1) == [2, 3, 4]
        @test Topologies.local_neighboring_elements(topology, 2) == [1, 3, 4]
        @test Topologies.local_neighboring_elements(topology, 3) == [1, 2, 4]
        @test Topologies.local_neighboring_elements(topology, 4) == [1, 2, 3]
    end

    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 2, false, false)
        @test Topologies.local_neighboring_elements(topology, 1) == [2, 3, 4]
        @test Topologies.local_neighboring_elements(topology, 2) == [1, 3, 4]
        @test Topologies.local_neighboring_elements(topology, 3) == [1, 2, 4]
        @test Topologies.local_neighboring_elements(topology, 4) == [1, 2, 3]
    end
end

@testset "simple rectangular mesh interior faces iterator" begin

    @testset "1×1 element quad mesh with all periodic boundries" begin
        topology = rectangular_grid(1, 1, true, true)
        @test length(Topologies.interior_faces(topology)) == 2
        faces = collect(Topologies.interior_faces(topology))
        @test sort(faces) == sort([(1, 4, 1, 2, true), (1, 3, 1, 1, true)])
    end


    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.interior_faces(topology)) == 1
        faces = collect(Topologies.interior_faces(topology))
        @test faces == [(1, 4, 1, 2, true)]
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.interior_faces(topology)) == 0
        faces = collect(Topologies.interior_faces(topology))
        @test isempty(faces)
    end

    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 2, false, false)
        @test length(Topologies.interior_faces(topology)) == 4
        faces = collect(Topologies.interior_faces(topology))
        @test sort(faces) == sort([
            (2, 4, 1, 2, true),
            (3, 1, 1, 3, true),
            (4, 4, 3, 2, true),
            (4, 1, 2, 3, true),
        ])
    end
end

@testset "simple rectangular mesh boundry faces iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        topology = rectangular_grid(1, 1, true, true)
        @test isempty(Topologies.boundary_tags(topology))
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.boundary_faces(topology, 1)) == 1
        @test length(Topologies.boundary_faces(topology, 2)) == 1
        @test collect(Topologies.boundary_faces(topology, 1)) == [(1, 1)]
        @test collect(Topologies.boundary_faces(topology, 2)) == [(1, 3)]
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.boundary_faces(topology, 1)) == 1
        @test length(Topologies.boundary_faces(topology, 2)) == 1
        @test length(Topologies.boundary_faces(topology, 3)) == 1
        @test length(Topologies.boundary_faces(topology, 4)) == 1
        @test collect(Topologies.boundary_faces(topology, 1)) == [(1, 4)]
        @test collect(Topologies.boundary_faces(topology, 2)) == [(1, 2)]
        @test collect(Topologies.boundary_faces(topology, 3)) == [(1, 1)]
        @test collect(Topologies.boundary_faces(topology, 4)) == [(1, 3)]
        @test Topologies.boundary_tag(topology, :west) == 1
        @test Topologies.boundary_tag(topology, :east) == 2
        @test Topologies.boundary_tag(topology, :south) == 3
        @test Topologies.boundary_tag(topology, :north) == 4
    end

    @testset "2×3 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 3, false, false)
        @test length(Topologies.boundary_faces(topology, 1)) == 3
        @test length(Topologies.boundary_faces(topology, 2)) == 3
        @test length(Topologies.boundary_faces(topology, 3)) == 2
        @test length(Topologies.boundary_faces(topology, 4)) == 2

        @test collect(Topologies.boundary_faces(topology, 1)) ==
              [(1, 4), (3, 4), (5, 4)]
        @test collect(Topologies.boundary_faces(topology, 2)) ==
              [(2, 2), (4, 2), (6, 2)]
        @test collect(Topologies.boundary_faces(topology, 3)) ==
              [(1, 1), (2, 1)]
        @test collect(Topologies.boundary_faces(topology, 4)) ==
              [(5, 3), (6, 3)]
    end
end

@testset "simple rectangular mesh vertex iterator" begin

    @testset "1×1 element quad mesh with all periodic boundries" begin
        topology = rectangular_grid(1, 1, true, true)
        # this has 1 global vertex
        @test length(Topologies.local_vertices(topology)) == 1
        V = collect(Topologies.local_vertices(topology))
        @test V[1] isa Topologies.Vertex
        @test sort(collect(V[1])) == [(1, 1), (1, 2), (1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        topology = rectangular_grid(1, 1, true, false)
        @test length(Topologies.local_vertices(topology)) == 2
        V = collect(Topologies.local_vertices(topology))
        @test V[1] isa Topologies.Vertex
        @test sort(collect(V[1])) == [(1, 1), (1, 2)]
        @test sort(collect(V[2])) == [(1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(1, 1, false, false)
        @test length(Topologies.local_vertices(topology)) == 4
        V = collect(Topologies.local_vertices(topology))
        @test V[1] isa Topologies.Vertex
        @test collect(V[1]) == [(1, 1)]
        @test collect(V[2]) == [(1, 2)]
        @test collect(V[3]) == [(1, 3)]
        @test collect(V[4]) == [(1, 4)]
    end

    @testset "2×3 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 3, false, false)
        @test length(Topologies.local_vertices(topology)) == 3 * 4
        V = collect(Topologies.local_vertices(topology))
        @test length(V[1]) == 1
        @test collect(V[1]) == [(1, 1)]
    end


    @testset "2×3 element quad mesh with periodic boundaries" begin
        topology = rectangular_grid(2, 3, true, true)
        @test length(Topologies.local_vertices(topology)) == 2 * 3
        V = collect(Topologies.local_vertices(topology))
        @test length(V) == 6
        @test sort([sort(collect(vert)) for vert in V]) == sort([
            sort([(1, 1), (2, 2), (5, 4), (6, 3)]),
            sort([(2, 1), (1, 2), (6, 4), (5, 3)]),
            sort([(3, 1), (4, 2), (1, 4), (2, 3)]),
            sort([(4, 1), (3, 2), (2, 4), (1, 3)]),
            sort([(5, 1), (6, 2), (3, 4), (4, 3)]),
            sort([(6, 1), (5, 2), (4, 4), (3, 3)]),
        ])
    end
end

@testset "simple rectangular mesh coordinates" begin


    @testset "1×1 element quad mesh with all periodic boundries" begin
        topology = rectangular_grid(1, 1, true, true)
        c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
        @test c1 == Geometry.XYPoint(0.0, 0.0)
        @test c2 == Geometry.XYPoint(1.0, 0.0)
        @test c3 == Geometry.XYPoint(1.0, 1.0)
        @test c4 == Geometry.XYPoint(0.0, 1.0)
    end

    @testset "1×1 element quad mesh with non-periodic boundries" begin
        topology =
            rectangular_grid(1, 1, false, false; x1min = -1.0, x2min = -1.0)
        c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
        @test c1 == Geometry.XYPoint(-1.0, -1.0)
        @test c2 == Geometry.XYPoint(1.0, -1.0)
        @test c3 == Geometry.XYPoint(1.0, 1.0)
        @test c4 == Geometry.XYPoint(-1.0, 1.0)
    end

    @testset "2×4 element quad mesh with non-periodic boundaries" begin
        topology = rectangular_grid(2, 4, false, false)
        c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
        @test c1 == Geometry.XYPoint(0.0, 0.0)
        @test c2 == Geometry.XYPoint(0.5, 0.0)
        @test c3 == Geometry.XYPoint(0.5, 0.25)
        @test c4 == Geometry.XYPoint(0.0, 0.25)

        c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 8)
        @test c1 == Geometry.XYPoint(0.5, 0.75)
        @test c2 == Geometry.XYPoint(1.0, 0.75)
        @test c3 == Geometry.XYPoint(1.0, 1.0)
        @test c4 == Geometry.XYPoint(0.5, 1.0)
    end


    @testset "check coordinate type accuracy" begin
        topology = rectangular_grid(
            3,
            1,
            false,
            false,
            x1min = big(0.0),
            x1max = big(1.0),
            x2min = big(0.0),
            x2max = big(1.0),
        )
        c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
        @test eltype(c2) == BigFloat
        @test getfield(c2, 1) ≈ big(1.0) / big(3.0) rtol = eps(BigFloat)
        @test getfield(c2, 2) == 0.0
    end
end
