using ClimaCore: Geometry, Domains, Meshes, Topologies
using Test


@testset "neighboring element tests" begin
    @testset "1 element across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 1),
        )
        @test Topologies.neighboring_elements(topology, 1) ==
              [6, 2, 3, 5, 0, 0, 0, 0]
    end
    @testset "2 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 2),
        )
        @test Topologies.neighboring_elements(topology, 1) ==
              [23, 2, 3, 20, 24, 4, 19, 0]
        @test Topologies.neighboring_elements(topology, 2) ==
              [24, 5, 4, 1, 0, 7, 3, 23]
        @test Topologies.neighboring_elements(topology, 3) ==
              [1, 4, 11, 19, 2, 9, 0, 20]
        @test Topologies.neighboring_elements(topology, 4) ==
              [2, 7, 9, 3, 5, 0, 11, 1]
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        @test Topologies.neighboring_elements(topology, 5) ==
              [2, 6, 8, 4, 3, 9, 7, 1]
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
