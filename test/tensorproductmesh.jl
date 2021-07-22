using Test
import ClimaCore: Domains, Meshes, Topologies
import ClimaCore.Geometry: Cartesian2DPoint
using StaticArrays
using IntervalSets

function tensorproduct_grid(
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
        x1min..x1max,
        x2min..x2max,
        x1periodic = x1periodic,
        x2periodic = x2periodic,
        x1boundary = x1periodic ? nothing : (:east, :west),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    mesh = Meshes.TensorProductMesh(domain, n1, n2)
    ts_topology = Topologies.TensorProductTopology(mesh)
    return (domain, mesh, ts_topology)
end

@testset "simple tensor-product grid opposing face" begin

    @testset "assert correct element numbering" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test_throws AssertionError Topologies.opposing_face(ts_topology, 0, 1)
        @test_throws AssertionError Topologies.opposing_face(ts_topology, 2, 1)
    end

    @testset "assert correct face numbering" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test_throws AssertionError Topologies.opposing_face(ts_topology, 1, 5)
        @test_throws AssertionError Topologies.opposing_face(ts_topology, 1, 0)
    end

    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test Topologies.opposing_face(ts_topology, 1, 1) == (1, 2, false)
        @test Topologies.opposing_face(ts_topology, 1, 2) == (1, 1, false)
        @test Topologies.opposing_face(ts_topology, 1, 3) == (1, 4, false)
        @test Topologies.opposing_face(ts_topology, 1, 4) == (1, 3, false)
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, false)
        @test Topologies.opposing_face(ts_topology, 1, 1) == (1, 2, false)
        @test Topologies.opposing_face(ts_topology, 1, 2) == (1, 1, false)
        @test Topologies.opposing_face(ts_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(ts_topology, 1, 4) == (0, 4, false)
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, false, false)
        @test Topologies.opposing_face(ts_topology, 1, 1) == (0, 1, false)
        @test Topologies.opposing_face(ts_topology, 1, 2) == (0, 2, false)
        @test Topologies.opposing_face(ts_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(ts_topology, 1, 4) == (0, 4, false)
    end

    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(2, 2, false, false)
        @test Topologies.opposing_face(ts_topology, 1, 1) == (0, 1, false)
        @test Topologies.opposing_face(ts_topology, 1, 2) == (2, 1, false)
        @test Topologies.opposing_face(ts_topology, 1, 3) == (0, 3, false)
        @test Topologies.opposing_face(ts_topology, 1, 4) == (3, 3, false)
        @test Topologies.opposing_face(ts_topology, 2, 1) == (1, 2, false)
        @test Topologies.opposing_face(ts_topology, 2, 2) == (0, 2, false)
        @test Topologies.opposing_face(ts_topology, 2, 3) == (0, 3, false)
        @test Topologies.opposing_face(ts_topology, 2, 4) == (4, 3, false)
    end
end

@testset "simple tensor-product grid interior faces iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test length(Topologies.interior_faces(ts_topology)) == 2
        faces = collect(Topologies.interior_faces(ts_topology))
        @test faces[1] == (1, 1, 1, 2, false)
        @test faces[2] == (1, 3, 1, 4, false)
    end
    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, false)
        @test length(Topologies.interior_faces(ts_topology)) == 1
        faces = collect(Topologies.interior_faces(ts_topology))
        @test faces[1] == (1, 1, 1, 2, false)
    end
    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, false, false)
        @test length(Topologies.interior_faces(ts_topology)) == 0
        faces = collect(Topologies.interior_faces(ts_topology))
        @test isempty(faces)
    end
    @testset "2×2 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(2, 2, false, false)
        @test length(Topologies.interior_faces(ts_topology)) == 4
        faces = collect(Topologies.interior_faces(ts_topology))
        @test faces[1] == (2, 1, 1, 2, false)
        @test faces[2] == (3, 3, 1, 4, false)
        @test faces[3] == (4, 1, 3, 2, false)
        @test faces[4] == (4, 3, 2, 4, false)
    end
end

@testset "simple tensor-product grid boundry faces iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test length(Topologies.boundary_faces(ts_topology, 1)) == 0
        @test length(Topologies.boundary_faces(ts_topology, 2)) == 0
        @test length(Topologies.boundary_faces(ts_topology, 3)) == 0
        @test length(Topologies.boundary_faces(ts_topology, 4)) == 0
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 1)))
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 2)))
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 3)))
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 4)))
    end
    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, false)
        @test length(Topologies.boundary_faces(ts_topology, 1)) == 0
        @test length(Topologies.boundary_faces(ts_topology, 2)) == 0
        @test length(Topologies.boundary_faces(ts_topology, 3)) == 1
        @test length(Topologies.boundary_faces(ts_topology, 4)) == 1
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 1)))
        @test isempty(collect(Topologies.boundary_faces(ts_topology, 2)))
        @test collect(Topologies.boundary_faces(ts_topology, 3)) == [(1, 3)]
        @test collect(Topologies.boundary_faces(ts_topology, 4)) == [(1, 4)]
    end
    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, false, false)
        @test length(Topologies.boundary_faces(ts_topology, 1)) == 1
        @test length(Topologies.boundary_faces(ts_topology, 2)) == 1
        @test length(Topologies.boundary_faces(ts_topology, 3)) == 1
        @test length(Topologies.boundary_faces(ts_topology, 4)) == 1
        @test collect(Topologies.boundary_faces(ts_topology, 1)) == [(1, 1)]
        @test collect(Topologies.boundary_faces(ts_topology, 2)) == [(1, 2)]
        @test collect(Topologies.boundary_faces(ts_topology, 3)) == [(1, 3)]
        @test collect(Topologies.boundary_faces(ts_topology, 4)) == [(1, 4)]
    end
    @testset "2×3 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(2, 3, false, false)
        @test length(Topologies.boundary_faces(ts_topology, 1)) == 3
        @test length(Topologies.boundary_faces(ts_topology, 2)) == 3
        @test length(Topologies.boundary_faces(ts_topology, 3)) == 2
        @test length(Topologies.boundary_faces(ts_topology, 4)) == 2

        @test collect(Topologies.boundary_faces(ts_topology, 1)) ==
              [(1, 1), (3, 1), (5, 1)]
        @test collect(Topologies.boundary_faces(ts_topology, 2)) ==
              [(2, 2), (4, 2), (6, 2)]
        @test collect(Topologies.boundary_faces(ts_topology, 3)) ==
              [(1, 3), (2, 3)]
        @test collect(Topologies.boundary_faces(ts_topology, 4)) ==
              [(5, 4), (6, 4)]
    end
end

@testset "simple rectangular grid vertex iterator" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        @test length(Topologies.vertices(ts_topology)) == 1
        V = collect(Topologies.vertices(ts_topology))
        @test V[1] isa Topologies.Vertex
        @test length(V[1]) == 4
        @test collect(V[1]) == [(1, 1), (1, 2), (1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with 1 periodic boundary" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, false)
        @test length(Topologies.vertices(ts_topology)) == 2
        V = collect(Topologies.vertices(ts_topology))
        @test V[1] isa Topologies.Vertex
        @test length(V[1]) == 2
        @test collect(V[1]) == [(1, 1), (1, 2)]
        @test length(V[2]) == 2
        @test collect(V[2]) == [(1, 3), (1, 4)]
    end

    @testset "1×1 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, false, false)
        @test length(Topologies.vertices(ts_topology)) == 4
        V = collect(Topologies.vertices(ts_topology))
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
        _, _, ts_topology = tensorproduct_grid(2, 3, false, false)
        @test length(Topologies.vertices(ts_topology)) == 3 * 4
        V = collect(Topologies.vertices(ts_topology))
        @test length(V[1]) == 1
        @test collect(V[1]) == [(1, 1)]
    end

    @testset "2×3 element quad mesh with periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(2, 3, true, true)
        @test length(Topologies.vertices(ts_topology)) == 2 * 3
        V = collect(Topologies.vertices(ts_topology))
        @test length(V) == 6
        @test collect(V[1]) == [(1, 1), (2, 2), (5, 3), (6, 4)]
        @test collect(V[2]) == [(2, 1), (1, 2), (6, 3), (5, 4)]
        @test collect(V[3]) == [(3, 1), (4, 2), (1, 3), (2, 4)]
        @test collect(V[4]) == [(4, 1), (3, 2), (2, 3), (1, 4)]
        @test collect(V[5]) == [(5, 1), (6, 2), (3, 3), (4, 4)]
        @test collect(V[6]) == [(6, 1), (5, 2), (4, 3), (3, 4)]
    end
end

@testset "simple rectangular grid coordinates" begin
    @testset "1×1 element quad mesh with all periodic boundries" begin
        _, _, ts_topology = tensorproduct_grid(1, 1, true, true)
        c1, c2, c3, c4 = Topologies.vertex_coordinates(ts_topology, 1)
        @test c1 == Cartesian2DPoint(0.0, 0.0)
        @test c2 == Cartesian2DPoint(1.0, 0.0)
        @test c3 == Cartesian2DPoint(0.0, 1.0)
        @test c4 == Cartesian2DPoint(1.0, 1.0)

        _, _, ts_topology = tensorproduct_grid(
            1,
            1,
            false,
            false;
            x1min = -1.0,
            x1max = 1.0,
            x2min = -1.0,
            x2max = 1.0,
        )
        c1, c2, c3, c4 = Topologies.vertex_coordinates(ts_topology, 1)
        @test c1 == Cartesian2DPoint(-1.0, -1.0)
        @test c2 == Cartesian2DPoint(1.0, -1.0)
        @test c3 == Cartesian2DPoint(-1.0, 1.0)
        @test c4 == Cartesian2DPoint(1.0, 1.0)
    end

    @testset "2×4 element quad mesh with non-periodic boundaries" begin
        _, _, ts_topology = tensorproduct_grid(2, 4, false, false)
        c1, c2, c3, c4 = Topologies.vertex_coordinates(ts_topology, 1)
        @test c1 == Cartesian2DPoint(0.0, 0.0)
        @test c2 == Cartesian2DPoint(0.5, 0.0)
        @test c3 == Cartesian2DPoint(0.0, 0.25)
        @test c4 == Cartesian2DPoint(0.5, 0.25)

        c1, c2, c3, c4 = Topologies.vertex_coordinates(ts_topology, 8)
        @test c1 == Cartesian2DPoint(0.5, 0.75)
        @test c2 == Cartesian2DPoint(1.0, 0.75)
        @test c3 == Cartesian2DPoint(0.5, 1.0)
        @test c4 == Cartesian2DPoint(1.0, 1.0)
    end

    @testset "check coordinate type accuracy" begin
        _, _, ts_topology = tensorproduct_grid(
            3,
            1,
            false,
            false,
            x1min = big(0.0),
            x1max = big(1.0),
            x2min = big(0.0),
            x2max = big(1.0),
        )
        c1, c2, c3, c4 = Topologies.vertex_coordinates(ts_topology, 1)
        @test eltype(c2) == BigFloat
        @test c2.x1 ≈ big(1.0) / big(3.0) rtol = eps(BigFloat)
        @test c2.x2 == 0.0
    end
end
