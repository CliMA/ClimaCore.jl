using ClimaCore: Domains, Meshes
using Test
using SparseArrays


@testset "opposing face" begin
    mesh = Meshes.EquiangularCubedSphereMesh(Domains.SphereDomain(1.0), 3)
    for elem in Meshes.elements(mesh)
        for face = 1:4
            (opelem, opface, reversed) = Meshes.opposing_face(mesh, elem, face)
            @test Meshes.opposing_face(mesh, opelem, opface) == (elem, face, reversed)
        end
    end
end

@testset "shared vertices" begin
    mesh = Meshes.EquiangularCubedSphereMesh(Domains.SphereDomain(1.0), 3)
    for elem in Meshes.elements(mesh)
        x,y,panel = elem.I
        for vert = 1:4
            iscorner =
                vert == 1 && x == 1 && y == 1 ||
                vert == 2 && x == 3 && y == 1 ||
                vert == 3 && x == 3 && y == 3 ||
                vert == 4 && x == 1 && y == 3
            shared_verts = collect(Meshes.SharedVertices(mesh, elem, vert))
            @test length(shared_verts) == (iscorner ? 3 : 4)
            sort!(shared_verts)

            for (velem, vvert) in shared_verts
                @test sort!(collect(Meshes.SharedVertices(mesh, velem, vvert))) == shared_verts
            end
        end
    end

end

@testset "vertex coordinates" begin
    domain = Domains.SphereDomain(5.0)
    for mesh in [
        Meshes.EquiangularCubedSphereMesh(domain, 3),
        Meshes.EquidistantCubedSphereMesh(domain, 3),
        Meshes.ConformalCubedSphereMesh(domain, 3),
    ]
        for elem in Meshes.elements(mesh)
            x,y,panel = elem.I
            for vert = 1:4
                coord = Meshes.coordinates(mesh, elem, vert)
                for (velem, vvert) in Meshes.SharedVertices(mesh, elem, vert)
                    @test Meshes.coordinates(mesh, velem, vvert) ≈ coord
                end
            end
        end
    end
end


@testset "face connectivity matrix" begin
    mesh = Meshes.EquiangularCubedSphereMesh(Domains.SphereDomain(1.0), 3)
    M = Meshes.face_connectivity_matrix(mesh)
    @test nnz(M) == 54 * 4 # every element is connected to 4 others
    # check each panel
    S = M[1:9,1:9]
    @test nnz(S) == 2*(6+6)
    for panel = 2:6
        @test S == M[panel*9-8:panel*9,panel*9-8:panel*9]
    end
end


@testset "vertex connectivity matrix" begin
    mesh = Meshes.EquiangularCubedSphereMesh(Domains.SphereDomain(1.0), 3)
    M = Meshes.vertex_connectivity_matrix(mesh)

    @test nnz(M) == 6*(
        4*(3 + 2 + 2) + # corners
        4*(5 + 3) + # edges
        1*8 # center
    )

    # check each panel
    S = M[1:9,1:9]
    @test nnz(S) == (
        4*3 + # corners
        4*5 + # edges
        1*8 # center
    )
    for panel = 2:6
        @test S == M[panel*9-8:panel*9,panel*9-8:panel*9]
    end

end


