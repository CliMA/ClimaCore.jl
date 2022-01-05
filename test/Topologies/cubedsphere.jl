using ClimaCore: Geometry, Domains, Meshes, Topologies
using Test




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
        length(Topologies.vertices(topology)) == 8
        for uvert in Topologies.vertices(topology)
            @test length(uvert) == 3
        end
    end
    @testset "3 elements across each panel" begin
        topology = Topologies.Topology2D(
            Meshes.EquiangularCubedSphere(Domains.SphereDomain(5.0), 3),
        )
        length(Topologies.vertices(topology)) == 8 + 12 * 2 + 6 * 4
        for uvert in Topologies.vertices(topology)
            @test length(uvert) in (3, 4)
        end
    end
end
