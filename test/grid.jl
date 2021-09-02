using Test
import ClimaCore: Domains, Meshes, Topologies
import ClimaCore.Geometry: Cartesian2DPoint
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
        x1min..x1max,
        x2min..x2max,
        x1periodic = x1periodic,
        x2periodic = x2periodic,
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    grid_topology = Topologies.GridTopology(mesh)
    return (domain, mesh, grid_topology)
end

function regular_tensorproduct_grid(
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
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    mesh = Meshes.TensorProductMesh(domain, n1, n2)
    r_ts_topology = Topologies.TensorProductTopology(mesh)
    return (domain, mesh, r_ts_topology)
end

function irregular_tensorproduct_grid(
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
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )

    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    nv2 = x2periodic ? n2 : n2 + 1
    range1 = mesh.range1
    range2 = mesh.range2
    elemheight = range2[2] - range2[1]

    coordinates =
        Vector{Cartesian2DPoint{typeof(x1min)}}(undef, (n1 + 1) * (n2 + 1))

    # Iterate mesh vertices
    grid_topo = Topologies.GridTopology(mesh)
    v_iter = Topologies.VertexIterator(grid_topo)

    for v in v_iter
        (i, j) = v.num
        # Shift 0-based indices
        i += 1
        j += 1
        vcoords = Cartesian2DPoint(range1[i], range2[j])
        # If I am on a boundary, don't apply any warping
        if j == 1 || j == nv2
            coordinates[(i - 1) * (n2 + 1) + j] = vcoords
        else # Interior vertex case, shift x2-coord of a fixed amount
            coordinates[(i - 1) * (n2 + 1) + j] =
                Cartesian2DPoint(vcoords.x1, vcoords.x2 + 0.5 * elemheight)
        end
    end
    ts_mesh = Meshes.TensorProductMesh(domain, n1, n2, coordinates)
    i_ts_topology = Topologies.TensorProductTopology(ts_mesh)
    return (domain, ts_mesh, i_ts_topology)
end

@testset "opposing face tests" begin

    _, _, grid_topology = rectangular_grid(1, 1, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, true)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "assert correct element numbering" begin
            @test_throws AssertionError Topologies.opposing_face(topology, 0, 1)
            @test_throws AssertionError Topologies.opposing_face(topology, 2, 1)
        end

        @testset "assert correct face numbering" begin
            @test_throws AssertionError Topologies.opposing_face(topology, 1, 5)
            @test_throws AssertionError Topologies.opposing_face(topology, 1, 0)
        end

        @testset "1×1 element quad mesh with all periodic boundries" begin
            @test Topologies.opposing_face(topology, 1, 1) == (1, 2, false)
            @test Topologies.opposing_face(topology, 1, 2) == (1, 1, false)
            @test Topologies.opposing_face(topology, 1, 3) == (1, 4, false)
            @test Topologies.opposing_face(topology, 1, 4) == (1, 3, false)
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, true, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with 1 periodic boundary" begin
            _, _, topology = rectangular_grid(1, 1, true, false)
            @test Topologies.opposing_face(topology, 1, 1) == (1, 2, false)
            @test Topologies.opposing_face(topology, 1, 2) == (1, 1, false)
            @test Topologies.opposing_face(topology, 1, 3) == (0, 3, false)
            @test Topologies.opposing_face(topology, 1, 4) == (0, 4, false)
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with non-periodic boundaries" begin
            @test Topologies.opposing_face(topology, 1, 1) == (0, 1, false)
            @test Topologies.opposing_face(topology, 1, 2) == (0, 2, false)
            @test Topologies.opposing_face(topology, 1, 3) == (0, 3, false)
            @test Topologies.opposing_face(topology, 1, 4) == (0, 4, false)
        end
    end

    _, _, grid_topology = rectangular_grid(2, 2, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 2, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "2×2 element quad mesh with non-periodic boundaries" begin
            @test Topologies.opposing_face(topology, 1, 1) == (0, 1, false)
            @test Topologies.opposing_face(topology, 1, 2) == (2, 1, false)
            @test Topologies.opposing_face(topology, 1, 3) == (0, 3, false)
            @test Topologies.opposing_face(topology, 1, 4) == (3, 3, false)
            @test Topologies.opposing_face(topology, 2, 1) == (1, 2, false)
            @test Topologies.opposing_face(topology, 2, 2) == (0, 2, false)
            @test Topologies.opposing_face(topology, 2, 3) == (0, 3, false)
            @test Topologies.opposing_face(topology, 2, 4) == (4, 3, false)
        end
    end
end

@testset "simple rectangular mesh interior faces iterator" begin

    _, _, grid_topology = rectangular_grid(1, 1, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, true)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with all periodic boundries" begin
            @test length(Topologies.interior_faces(topology)) == 2
            faces = collect(Topologies.interior_faces(topology))
            @test faces[1] == (1, 1, 1, 2, false)
            @test faces[2] == (1, 3, 1, 4, false)
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, true, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with 1 periodic boundary" begin
            @test length(Topologies.interior_faces(topology)) == 1
            faces = collect(Topologies.interior_faces(topology))
            @test faces[1] == (1, 1, 1, 2, false)
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.interior_faces(topology)) == 0
            faces = collect(Topologies.interior_faces(topology))
            @test isempty(faces)
        end
    end

    _, _, grid_topology = rectangular_grid(2, 2, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 2, false, false)
    _, _, i_ts_topology = irregular_tensorproduct_grid(2, 2, false, false)
    topologies = (grid_topology, r_ts_topology, i_ts_topology)
    for topology in topologies
        @testset "2×2 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.interior_faces(topology)) == 4
            faces = collect(Topologies.interior_faces(topology))
            @test faces[1] == (2, 1, 1, 2, false)
            @test faces[2] == (3, 3, 1, 4, false)
            @test faces[3] == (4, 1, 3, 2, false)
            @test faces[4] == (4, 3, 2, 4, false)
        end
    end
end

@testset "simple rectangular mesh boundry faces iterator" begin
    _, _, grid_topology = rectangular_grid(1, 1, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, true)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with all periodic boundries" begin
            @test length(Topologies.boundary_faces(topology, 1)) == 0
            @test length(Topologies.boundary_faces(topology, 2)) == 0
            @test length(Topologies.boundary_faces(topology, 3)) == 0
            @test length(Topologies.boundary_faces(topology, 4)) == 0
            @test isempty(collect(Topologies.boundary_faces(topology, 1)))
            @test isempty(collect(Topologies.boundary_faces(topology, 2)))
            @test isempty(collect(Topologies.boundary_faces(topology, 3)))
            @test isempty(collect(Topologies.boundary_faces(topology, 4)))
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, true, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with 1 periodic boundary" begin
            @test length(Topologies.boundary_faces(topology, 1)) == 0
            @test length(Topologies.boundary_faces(topology, 2)) == 0
            @test length(Topologies.boundary_faces(topology, 3)) == 1
            @test length(Topologies.boundary_faces(topology, 4)) == 1
            @test isempty(collect(Topologies.boundary_faces(topology, 1)))
            @test isempty(collect(Topologies.boundary_faces(topology, 2)))
            @test collect(Topologies.boundary_faces(topology, 3)) == [(1, 3)]
            @test collect(Topologies.boundary_faces(topology, 4)) == [(1, 4)]
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.boundary_faces(topology, 1)) == 1
            @test length(Topologies.boundary_faces(topology, 2)) == 1
            @test length(Topologies.boundary_faces(topology, 3)) == 1
            @test length(Topologies.boundary_faces(topology, 4)) == 1
            @test collect(Topologies.boundary_faces(topology, 1)) == [(1, 1)]
            @test collect(Topologies.boundary_faces(topology, 2)) == [(1, 2)]
            @test collect(Topologies.boundary_faces(topology, 3)) == [(1, 3)]
            @test collect(Topologies.boundary_faces(topology, 4)) == [(1, 4)]
            @test Topologies.boundary_tag(topology, :west) == 1
            @test Topologies.boundary_tag(topology, :east) == 2
            @test Topologies.boundary_tag(topology, :south) == 3
            @test Topologies.boundary_tag(topology, :north) == 4
        end
    end

    _, _, grid_topology = rectangular_grid(2, 3, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 3, false, false)
    _, _, i_ts_topology = irregular_tensorproduct_grid(2, 3, false, false)
    topologies = (grid_topology, r_ts_topology, i_ts_topology)
    for topology in topologies
        @testset "2×3 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.boundary_faces(topology, 1)) == 3
            @test length(Topologies.boundary_faces(topology, 2)) == 3
            @test length(Topologies.boundary_faces(topology, 3)) == 2
            @test length(Topologies.boundary_faces(topology, 4)) == 2

            @test collect(Topologies.boundary_faces(topology, 1)) ==
                  [(1, 1), (3, 1), (5, 1)]
            @test collect(Topologies.boundary_faces(topology, 2)) ==
                  [(2, 2), (4, 2), (6, 2)]
            @test collect(Topologies.boundary_faces(topology, 3)) ==
                  [(1, 3), (2, 3)]
            @test collect(Topologies.boundary_faces(topology, 4)) ==
                  [(5, 4), (6, 4)]
        end
    end
end

@testset "simple rectangular mesh vertex iterator" begin

    _, _, grid_topology = rectangular_grid(1, 1, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, true)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with all periodic boundries" begin
            @test length(Topologies.vertices(topology)) == 1
            V = collect(Topologies.vertices(topology))
            @test V[1] isa Topologies.Vertex
            @test length(V[1]) == 4
            @test collect(V[1]) == [(1, 1), (1, 2), (1, 3), (1, 4)]
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, true, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with 1 periodic boundary" begin
            @test length(Topologies.vertices(topology)) == 2
            V = collect(Topologies.vertices(topology))
            @test V[1] isa Topologies.Vertex
            @test length(V[1]) == 2
            @test collect(V[1]) == [(1, 1), (1, 2)]
            @test length(V[2]) == 2
            @test collect(V[2]) == [(1, 3), (1, 4)]
        end
    end

    _, _, grid_topology = rectangular_grid(1, 1, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.vertices(topology)) == 4
            V = collect(Topologies.vertices(topology))
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
    end

    _, _, grid_topology = rectangular_grid(2, 3, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 3, false, false)
    _, _, i_ts_topology = irregular_tensorproduct_grid(2, 3, false, false)
    topologies = (grid_topology, r_ts_topology, i_ts_topology)
    for topology in topologies
        @testset "2×3 element quad mesh with non-periodic boundaries" begin
            @test length(Topologies.vertices(topology)) == 3 * 4
            V = collect(Topologies.vertices(topology))
            @test length(V[1]) == 1
            @test collect(V[1]) == [(1, 1)]
        end
    end

    _, _, grid_topology = rectangular_grid(2, 3, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 3, true, true)
    _, _, i_ts_topology = irregular_tensorproduct_grid(2, 3, true, true)
    topologies = (grid_topology, r_ts_topology, i_ts_topology)
    for topology in topologies
        @testset "2×3 element quad mesh with periodic boundaries" begin
            @test length(Topologies.vertices(topology)) == 2 * 3
            V = collect(Topologies.vertices(topology))
            @test length(V) == 6
            @test collect(V[1]) == [(1, 1), (2, 2), (5, 3), (6, 4)]
            @test collect(V[2]) == [(2, 1), (1, 2), (6, 3), (5, 4)]
            @test collect(V[3]) == [(3, 1), (4, 2), (1, 3), (2, 4)]
            @test collect(V[4]) == [(4, 1), (3, 2), (2, 3), (1, 4)]
            @test collect(V[5]) == [(5, 1), (6, 2), (3, 3), (4, 4)]
            @test collect(V[6]) == [(6, 1), (5, 2), (4, 3), (3, 4)]
        end
    end
end

@testset "simple rectangular mesh coordinates" begin

    _, _, grid_topology = rectangular_grid(1, 1, true, true)
    _, _, r_ts_topology = regular_tensorproduct_grid(1, 1, true, true)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with all periodic boundries" begin
            c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
            @test c1 == Cartesian2DPoint(0.0, 0.0)
            @test c2 == Cartesian2DPoint(1.0, 0.0)
            @test c3 == Cartesian2DPoint(0.0, 1.0)
            @test c4 == Cartesian2DPoint(1.0, 1.0)

        end
    end

    _, _, grid_topology = rectangular_grid(
        1,
        1,
        false,
        false;
        x1min = -1.0,
        x1max = 1.0,
        x2min = -1.0,
        x2max = 1.0,
    )
    _, _, r_ts_topology = regular_tensorproduct_grid(
        1,
        1,
        false,
        false;
        x1min = -1.0,
        x1max = 1.0,
        x2min = -1.0,
        x2max = 1.0,
    )
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "1×1 element quad mesh with non-periodic boundries" begin
            c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
            @test c1 == Cartesian2DPoint(-1.0, -1.0)
            @test c2 == Cartesian2DPoint(1.0, -1.0)
            @test c3 == Cartesian2DPoint(-1.0, 1.0)
            @test c4 == Cartesian2DPoint(1.0, 1.0)
        end
    end

    _, _, grid_topology = rectangular_grid(2, 4, false, false)
    _, _, r_ts_topology = regular_tensorproduct_grid(2, 4, false, false)
    topologies = (grid_topology, r_ts_topology)
    for topology in topologies
        @testset "2×4 element quad mesh with non-periodic boundaries" begin
            c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
            @test c1 == Cartesian2DPoint(0.0, 0.0)
            @test c2 == Cartesian2DPoint(0.5, 0.0)
            @test c3 == Cartesian2DPoint(0.0, 0.25)
            @test c4 == Cartesian2DPoint(0.5, 0.25)

            c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 8)
            @test c1 == Cartesian2DPoint(0.5, 0.75)
            @test c2 == Cartesian2DPoint(1.0, 0.75)
            @test c3 == Cartesian2DPoint(0.5, 1.0)
            @test c4 == Cartesian2DPoint(1.0, 1.0)
        end
    end
    _, _, i_ts_topology = irregular_tensorproduct_grid(2, 4, false, false)
    @testset "2×4 element quad mesh with non-periodic boundaries" begin
        c1, c2, c3, c4 = Topologies.vertex_coordinates(i_ts_topology, 1)
        @test c1 == Cartesian2DPoint(0.0, 0.0)
        @test c2 == Cartesian2DPoint(0.5, 0.0)
        @test c3 == Cartesian2DPoint(0.0, 0.375)
        @test c4 == Cartesian2DPoint(0.5, 0.375)

        c1, c2, c3, c4 = Topologies.vertex_coordinates(i_ts_topology, 8)
        @test c1 == Cartesian2DPoint(0.5, 0.875)
        @test c2 == Cartesian2DPoint(1.0, 0.875)
        @test c3 == Cartesian2DPoint(0.5, 1.0)
        @test c4 == Cartesian2DPoint(1.0, 1.0)
    end

    _, _, grid_topology = rectangular_grid(
        3,
        1,
        false,
        false,
        x1min = big(0.0),
        x1max = big(1.0),
        x2min = big(0.0),
        x2max = big(1.0),
    )
    _, _, r_ts_topology = regular_tensorproduct_grid(
        3,
        1,
        false,
        false,
        x1min = big(0.0),
        x1max = big(1.0),
        x2min = big(0.0),
        x2max = big(1.0),
    )

    _, _, i_ts_topology = irregular_tensorproduct_grid(
        3,
        1,
        false,
        false,
        x1min = big(0.0),
        x1max = big(1.0),
        x2min = big(0.0),
        x2max = big(1.0),
    )
    topologies = (grid_topology, r_ts_topology, i_ts_topology)
    for topology in topologies
        @testset "check coordinate type accuracy" begin
            c1, c2, c3, c4 = Topologies.vertex_coordinates(topology, 1)
            @test eltype(c2) == BigFloat
            @test c2.x1 ≈ big(1.0) / big(3.0) rtol = eps(BigFloat)
            @test c2.x2 == 0.0
        end
    end
end
