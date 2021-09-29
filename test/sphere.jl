using LinearAlgebra, IntervalSets, UnPack
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, Operators, Fields
import ClimaCore.Meshes: EquiangularSphereWarp, Mesh2D

using Test



using StaticArrays, LinearAlgebra
function rotational_field(space, axis::Geometry.LatLongPoint)
    n_axis = Geometry.components(Geometry.Cartesian123Point(axis))
    coords = Fields.coordinate_field(space)
    map(coords) do coord
        n_coord = Geometry.components(Geometry.Cartesian123Point(coord))
        u_cart = n_axis × n_coord
        θ = coord.lat
        λ = coord.long
        F = @SMatrix [
            -sind(λ) cosd(λ) 0
            0 0 1/cosd(θ)
        ]
        uv = F * u_cart
        if abs(θ) ≈ 90
            Geometry.UVVector(u_cart[1], u_cart[2])
        else

            Geometry.UVVector(uv...)
        end
    end
end

@testset "Spherical geometry properties" begin
    FT = Float64
    radius = FT(3)
    ne = 4
    Nq = 4
    Nqh = 7
    domain = Domains.SphereDomain(radius)
    mesh = Mesh2D(domain, EquiangularSphereWarp(), ne)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    @test sum(ones(space)) ≈ 4pi * radius^2 rtol = 1e-3


    div = Operators.Divergence()
    u = rotational_field(space, Geometry.LatLongPoint(45.0, 45.0))
    divu = Spaces.weighted_dss!(div.(u))
    @test norm(divu) < 1e-2

end
