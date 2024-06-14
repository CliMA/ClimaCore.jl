#=
julia --project
using Revise; include(joinpath("test", "Meshes", "cubedsphere.jl"))
=#
using ClimaCore: Geometry, Domains, Meshes
using Test
using StaticArrays, SparseArrays, LinearAlgebra, ForwardDiff


@testset "opposing face" begin
    mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(1.0), 3)
    for elem in Meshes.elements(mesh)
        for face in 1:4
            (opelem, opface, reversed) = Meshes.opposing_face(mesh, elem, face)
            @test Meshes.opposing_face(mesh, opelem, opface) ==
                  (elem, face, reversed)
        end
    end
end

@testset "shared vertices" begin
    mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(1.0), 3)
    for elem in Meshes.elements(mesh)
        x, y, panel = elem.I
        for vert in 1:4
            iscorner =
                vert == 1 && x == 1 && y == 1 ||
                vert == 2 && x == 3 && y == 1 ||
                vert == 3 && x == 3 && y == 3 ||
                vert == 4 && x == 1 && y == 3
            shared_verts = collect(Meshes.SharedVertices(mesh, elem, vert))
            @test length(shared_verts) == (iscorner ? 3 : 4)
            sort!(shared_verts)

            for (velem, vvert) in shared_verts
                @test sort!(
                    collect(Meshes.SharedVertices(mesh, velem, vvert)),
                ) == shared_verts
            end
        end
    end
end

@testset "panel mappings" begin
    for coord1 in [
        Geometry.Cartesian123Point(1.0, 0.0, 0.0),
        Geometry.Cartesian123Point(1.0, sqrt(0.5), 0.0),
        Geometry.Cartesian123Point(1.0, 0.0, sqrt(0.5)),
        Geometry.Cartesian123Point(1.0, sqrt(0.5), -sqrt(0.5)),
    ]
        for panel in 1:6
            coord = Meshes.to_panel(panel, coord1)
            @test Meshes.containing_panel(coord) == panel
            @test Meshes.from_panel(panel, coord) == coord1
        end
    end
end

@testset "vertex coordinates consistent" begin
    for FT in [Float32, Float64, BigFloat]
        domain = Domains.SphereDomain(FT(5))
        if FT == Float64
            meshes = [
                Meshes.EquiangularCubedSphere(domain, 3),
                Meshes.EquidistantCubedSphere(domain, 3),
                Meshes.ConformalCubedSphere(domain, 3),
            ]
        else
            meshes = [
                Meshes.EquiangularCubedSphere(domain, 3),
                Meshes.EquidistantCubedSphere(domain, 3),
            ]
        end
        for mesh in meshes
            for elem in Meshes.elements(mesh)
                for vert in 1:4
                    coord = Meshes.coordinates(mesh, elem, vert)
                    @test coord isa Geometry.Cartesian123Point{FT}
                    @test norm(Geometry.components(coord)) ≈ 5 rtol = 3eps(FT)
                    for (velem, vvert) in
                        Meshes.SharedVertices(mesh, elem, vert)
                        @test Meshes.coordinates(mesh, velem, vvert) ≈ coord rtol =
                            10eps(FT)
                    end
                end
            end
            @test Meshes.element_horizontal_length_scale(mesh) ≈
                  sqrt((4pi * 5^2) / Meshes.nelements(mesh))
        end
    end
end


@testset "vertex coordinate warping / unwarping" begin
    for FT in [Float32, Float64, BigFloat]
        domain = Domains.SphereDomain(FT(5))
        if FT == Float64
            meshes = [
                Meshes.EquiangularCubedSphere(domain, 3),
                Meshes.EquidistantCubedSphere(domain, 3),
                Meshes.ConformalCubedSphere(domain, 3),
                Meshes.EquiangularCubedSphere(domain, 3, Meshes.IntrinsicMap()),
                Meshes.EquidistantCubedSphere(domain, 3, Meshes.IntrinsicMap()),
                Meshes.ConformalCubedSphere(domain, 3, Meshes.IntrinsicMap()),
            ]
        else
            meshes = [
                Meshes.EquiangularCubedSphere(domain, 3),
                Meshes.EquidistantCubedSphere(domain, 3),
                Meshes.EquiangularCubedSphere(domain, 3, Meshes.IntrinsicMap()),
                Meshes.EquidistantCubedSphere(domain, 3, Meshes.IntrinsicMap()),
            ]
        end
        for mesh in meshes
            for elem in Meshes.elements(mesh)
                for ξ in [
                    FT.(SVector(0.0, 0.0)),
                    FT.(SVector(0.0, 0.5)),
                    FT.(SVector(0.5, 0.0)),
                    FT.(SVector(0.5, -0.5)),
                ]
                    coord = Meshes.coordinates(mesh, elem, ξ)
                    @test coord isa Geometry.Cartesian123Point{FT}
                    @test norm(Geometry.components(coord)) ≈ 5 rtol = 3eps(FT)
                    @test Meshes.containing_element(mesh, coord) == elem
                    @test Meshes.reference_coordinates(mesh, elem, coord) ≈ ξ atol =
                        20eps(FT)
                end

                for vert in 1:4
                    coord = Meshes.coordinates(mesh, elem, vert)
                    # containing_element should be round trip to give the same coordinates
                    celem = Meshes.containing_element(mesh, coord)
                    @test celem in Meshes.elements(mesh)
                    cξ = Meshes.reference_coordinates(mesh, celem, coord)
                    @test Geometry.components(
                        Meshes.coordinates(mesh, celem, cξ),
                    ) ≈ Geometry.components(coord) rtol = 3eps(FT)

                    cξ = Meshes.reference_coordinates(mesh, elem, coord)
                    @test cξ[1] ≈ (vert in (1, 4) ? -1 : +1)
                    @test cξ[2] ≈ (vert in (1, 2) ? -1 : +1)
                end
            end
        end
    end
end

@testset "handling of zero signs: even-numbered elements" begin
    # this doesn't work for conformal mappings
    domain = Domains.SphereDomain(5.0)
    meshes = [
        Meshes.EquiangularCubedSphere(domain, 4),
        Meshes.EquidistantCubedSphere(domain, 4),
        Meshes.EquiangularCubedSphere(domain, 4, Meshes.IntrinsicMap()),
        Meshes.EquidistantCubedSphere(domain, 4, Meshes.IntrinsicMap()),
    ]
    for mesh in meshes
        @test Meshes.coordinates(mesh, CartesianIndex(2, 1, 1), 2).x2 === -0.0
        @test Meshes.coordinates(mesh, CartesianIndex(2, 1, 1), 3).x2 === -0.0
        @test Meshes.coordinates(mesh, CartesianIndex(3, 1, 1), 1).x2 === +0.0
        @test Meshes.coordinates(mesh, CartesianIndex(3, 1, 1), 4).x2 === +0.0

        @test Meshes.coordinates(
            mesh,
            CartesianIndex(2, 1, 1),
            SVector(1.0, 0.5),
        ).x2 === -0.0
        @test Meshes.coordinates(
            mesh,
            CartesianIndex(3, 1, 1),
            SVector(-1.0, 0.5),
        ).x2 === +0.0

        @test Meshes.coordinates(mesh, CartesianIndex(1, 2, 1), 3).x3 === -0.0
        @test Meshes.coordinates(mesh, CartesianIndex(1, 2, 1), 4).x3 === -0.0
        @test Meshes.coordinates(mesh, CartesianIndex(1, 3, 1), 1).x3 === +0.0
        @test Meshes.coordinates(mesh, CartesianIndex(1, 3, 1), 2).x3 === +0.0

        @test Meshes.coordinates(
            mesh,
            CartesianIndex(1, 2, 1),
            SVector(0.5, 1.0),
        ).x3 === -0.0
        @test Meshes.coordinates(
            mesh,
            CartesianIndex(1, 3, 1),
            SVector(0.5, -1.0),
        ).x3 === +0.0

        # derivative handling
        M = ForwardDiff.jacobian(SVector(-1.0, 1.0)) do ξ
            Geometry.components(
                Meshes.coordinates(mesh, CartesianIndex(1, 2, 3), ξ),
            )
        end
        @test M[1, 1] < 0
        @test abs(M[2, 1]) ≤ eps(Float64)
        @test M[3, 1] > 0
        @test abs(M[1, 2]) ≤ eps(Float64)
        @test M[2, 2] < 0
        @test abs(M[3, 2]) ≤ eps(Float64)

        M = ForwardDiff.jacobian(SVector(-1.0, -1.0)) do ξ
            Geometry.components(
                Meshes.coordinates(mesh, CartesianIndex(1, 3, 3), ξ),
            )
        end
        @test M[1, 1] < 0
        @test abs(M[2, 1]) ≤ eps(Float64)
        @test M[3, 1] > 0
        @test abs(M[1, 2]) ≤ eps(Float64)
        @test M[2, 2] < 0
        @test abs(M[3, 2]) ≤ eps(Float64)
    end
end

@testset "handling of zero signs: odd-numbered elements" begin
    # currently this only works when using the IntrinsicMap
    domain = Domains.SphereDomain(5.0)
    meshes = [
        Meshes.EquiangularCubedSphere(domain, 3, Meshes.IntrinsicMap()),
        Meshes.EquidistantCubedSphere(domain, 3, Meshes.IntrinsicMap()),
    ]
    for mesh in meshes
        @test Meshes.coordinates(
            mesh,
            CartesianIndex(2, 1, 1),
            SVector(-0.0, 0.5),
        ).x2 === -0.0
        @test Meshes.coordinates(
            mesh,
            CartesianIndex(2, 1, 1),
            SVector(+0.0, 0.5),
        ).x2 === +0.0

        @test Meshes.coordinates(
            mesh,
            CartesianIndex(1, 2, 1),
            SVector(0.5, -0.0),
        ).x3 === -0.0
        @test Meshes.coordinates(
            mesh,
            CartesianIndex(1, 2, 1),
            SVector(0.5, +0.0),
        ).x3 === +0.0

        # derivative handling
        M = ForwardDiff.jacobian(SVector(-1.0, -0.0)) do ξ
            Geometry.components(
                Meshes.coordinates(mesh, CartesianIndex(1, 2, 3), ξ),
            )
        end
        @test M[1, 1] < 0
        @test M[2, 1] == 0
        @test M[3, 1] > 0
        @test M[1, 2] == 0
        @test M[2, 2] < 0
        @test M[3, 2] == 0
        M = ForwardDiff.jacobian(SVector(-1.0, +0.0)) do ξ
            Geometry.components(
                Meshes.coordinates(mesh, CartesianIndex(1, 2, 3), ξ),
            )
        end
        @test M[1, 1] < 0
        @test M[2, 1] == 0
        @test M[3, 1] > 0
        @test M[1, 2] == 0
        @test M[2, 2] < 0
        @test M[3, 2] == 0
    end
end

@testset "face connectivity matrix" begin
    mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(1.0), 3)
    M = Meshes.face_connectivity_matrix(mesh)
    @test nnz(M) == 54 * 4 # every element is connected to 4 others
    # check each panel
    S = M[1:9, 1:9]
    @test nnz(S) == 2 * (6 + 6)
    for panel in 2:6
        @test S == M[(panel * 9 - 8):(panel * 9), (panel * 9 - 8):(panel * 9)]
    end
end


@testset "vertex connectivity matrix" begin
    mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(1.0), 3)
    M = Meshes.vertex_connectivity_matrix(mesh)

    @test nnz(M) == 6 * (
        4 * (3 + 2 + 2) + # corners
        4 * (5 + 3) + # edges
        1 * 8 # center
    )

    # check each panel
    S = M[1:9, 1:9]
    @test nnz(S) == (
        4 * 3 + # corners
        4 * 5 + # edges
        1 * 8 # center
    )
    for panel in 2:6
        @test S == M[(panel * 9 - 8):(panel * 9), (panel * 9 - 8):(panel * 9)]
    end

end
