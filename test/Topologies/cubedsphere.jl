using ClimaCore: Geometry, Domains, Meshes, Topologies
using Test


@testset "neighboring element tests" begin
    @testset "1 element across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 1),
        )
        @test Topologies.local_neighboring_elements(topology, 1) == [2, 3, 5, 6]
    end
    @testset "2 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 2),
        )
        @test Topologies.local_neighboring_elements(topology, 1) ==
              [2, 3, 4, 19, 20, 23, 24]
        @test Topologies.local_neighboring_elements(topology, 2) ==
              [1, 3, 4, 5, 7, 23, 24]
        @test Topologies.local_neighboring_elements(topology, 3) ==
              [1, 2, 4, 9, 11, 19, 20]
        @test Topologies.local_neighboring_elements(topology, 4) ==
              [1, 2, 3, 5, 7, 9, 11]
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        @test Topologies.local_neighboring_elements(topology, 5) ==
              [1, 2, 3, 4, 6, 7, 8, 9]
    end
end

@testset "interior faces iterator" begin
    @testset "1 element across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 1),
        )
        @test length(Topologies.interior_faces(topology)) == 12
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        @test length(Topologies.interior_faces(topology)) == 6 * 2 * 6 + 12 * 3
    end
end

@testset "boundaries" begin
    @testset "1 element across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 1),
        )
        @test isempty(Topologies.boundary_tags(topology))
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        @test isempty(Topologies.boundary_tags(topology))
    end
end

@testset "boundaries" begin
    @testset "1 element across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 1),
        )
        length(Topologies.local_vertices(topology)) == 8
        for uvert in Topologies.local_vertices(topology)
            @test length(uvert) == 3
        end
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        length(Topologies.local_vertices(topology)) == 8 + 12 * 2 + 6 * 4
        for uvert in Topologies.local_vertices(topology)
            @test length(uvert) in (3, 4)
        end
    end
end
