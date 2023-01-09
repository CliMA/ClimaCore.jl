using LinearAlgebra, IntervalSets, UnPack
using ClimaComms
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, Operators, Fields

using Test

using StaticArrays, LinearAlgebra

function rotational_field(space, α0 = 45.0)
    coords = Fields.coordinate_field(space)
    map(coords) do coord
        ϕ = coord.lat
        λ = coord.long
        uu = (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end
end

@testset "Spherical geometry properties" begin
    # test different combinations of odd/even to ensure pole is correctly
    # handled
    for Ne in (4, 5), Nq in (4, 5)
        FT = Float64
        radius = FT(3)

        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, Ne)
        grid_topology =
            Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
        quad = Spaces.Quadratures.GLL{Nq}()
        space = Spaces.SpectralElementSpace2D(grid_topology, quad)

        @test sum(ones(space)) ≈ 4pi * radius^2 rtol = 1e-3

        for α in [0.0, 45.0, 90.0]
            div = Operators.Divergence()
            u = rotational_field(space, α)
            divu = Spaces.weighted_dss!(div.(u))
            @test norm(divu) < 1e-2

            # test dss on UVcoordinates
            uu = Spaces.weighted_dss!(copy(u))
            @test norm(uu .- u) < 1e-14

            uᵢ = Geometry.transform.(Ref(Geometry.Covariant12Axis()), u)
            uuᵢ = Spaces.weighted_dss!(copy(uᵢ))
            @test norm(uuᵢ .- uᵢ) < 1e-14

            uⁱ = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), u)
            uuⁱ = Spaces.weighted_dss!(copy(uⁱ))
            @test norm(uuⁱ .- uⁱ) < 1e-14
        end
    end
end
