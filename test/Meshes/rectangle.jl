using ClimaCore: Domains, Meshes, Geometry
using Test
using SparseArrays


function rectangular_mesh(
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
        Domains.IntervalDomain(
            Geometry.XPoint(x1min),
            Geometry.XPoint(x1max);
            periodic = x1periodic,
            boundary_names = boundary =
                x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(x2min),
            Geometry.YPoint(x2max);
            periodic = x2periodic,
            boundary_names = boundary =
                x2periodic ? nothing : (:south, :north),
        ),
    )
    return Meshes.RectilinearMesh(domain, n1, n2)
end

meshes = [
    rectangular_mesh(1, 1, true, true),
    rectangular_mesh(1, 1, true, false),
    rectangular_mesh(1, 1, false, true),
    rectangular_mesh(1, 1, false, false),
    rectangular_mesh(2, 3, true, true),
    rectangular_mesh(2, 3, true, false),
    rectangular_mesh(2, 3, false, true),
    rectangular_mesh(2, 3, false, false),
]

@testset "elements" begin
    @test Meshes.elements(rectangular_mesh(1, 1, true, true)) ==
          CartesianIndices((1, 1))
    @test Meshes.elements(rectangular_mesh(2, 3, true, true)) ==
          CartesianIndices((2, 3))
end

@testset "opposing face" begin
    for mesh in meshes
        for elem in Meshes.elements(mesh)
            for face in 1:4
                if Meshes.is_boundary_face(mesh, elem, face)
                    if face == 1
                        @test Meshes.boundary_face_name(mesh, elem, face) ==
                              :south
                    elseif face == 2
                        @test Meshes.boundary_face_name(mesh, elem, face) ==
                              :east
                    elseif face == 3
                        @test Meshes.boundary_face_name(mesh, elem, face) ==
                              :north
                    elseif face == 4
                        @test Meshes.boundary_face_name(mesh, elem, face) ==
                              :west
                    end
                else
                    (opelem, opface, reversed) =
                        Meshes.opposing_face(mesh, elem, face)
                    @test Meshes.opposing_face(mesh, opelem, opface) ==
                          (elem, face, reversed)
                end
            end
        end
    end
end

@testset "shared vertices" begin
    for mesh in meshes
        for elem in Meshes.elements(mesh)
            for vert in 1:4
                shared_verts = collect(Meshes.SharedVertices(mesh, elem, vert))
                @test 1 <= length(shared_verts) <= 4
                sort!(shared_verts)

                for (velem, vvert) in shared_verts
                    @test sort!(
                        collect(Meshes.SharedVertices(mesh, velem, vvert)),
                    ) == shared_verts
                end
            end
        end
    end
end


@testset "vertex coordinates consistent" begin
    for mesh in meshes
        for elem in Meshes.elements(mesh)
            for vert in 1:4
                coord = Meshes.coordinates(mesh, elem, vert)
                @test 0 <= coord.x <= 1
                @test 0 <= coord.y <= 1
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    vcoord = Meshes.coordinates(mesh, velem, vvert)
                    @test mod(coord.x - vcoord.x, 1) == 0
                    @test mod(coord.y - vcoord.y, 1) == 0
                end
            end
        end
    end
end


@testset "vertex coordinate warping / unwarping" begin
    domain = Domains.SphereDomain(5.0)
    for mesh in meshes
        for elem in Meshes.elements(mesh)
            for (ξ1, ξ2) in [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, -0.5)]
                coord = Meshes.coordinates(mesh, elem, (ξ1, ξ2))
                celem, (cξ1, cξ2) = Meshes.containing_element(mesh, coord)
                @test celem == elem
                @test cξ1 ≈ ξ1 atol = 100eps()
                @test cξ2 ≈ ξ2 atol = 100eps()
            end

            for vert in 1:4
                coord = Meshes.coordinates(mesh, elem, vert)
                # containing_element should be round trip to give the same coordinates
                celem, (cξ1, cξ2) = Meshes.containing_element(mesh, coord)
                @test celem in Meshes.elements(mesh)
                @test -1 <= cξ1 <= 1
                @test -1 <= cξ2 <= 1
                @test cξ1 ≈ 1 || cξ1 ≈ -1
                @test cξ2 ≈ 1 || cξ2 ≈ -1
                @test Meshes.coordinates(mesh, celem, (cξ1, cξ2)) ≈ coord
            end
        end
    end
end
