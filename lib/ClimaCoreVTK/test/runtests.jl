using Test
using ClimaCoreVTK
using IntervalSets
import ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Fields


dir = mktempdir()

@testset "sphere" begin
    R = 6.37122e6

    domain = Domains.SphereDomain(R)
    mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), 6)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{5}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    u = map(coords) do coord
        u0 = 20.0
        α0 = 45.0
        ϕ = coord.lat
        λ = coord.long

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        Geometry.UVVector(uu, uv)
    end

    writevtk(joinpath(dir, "sphere"), (coords = coords, u = u))
end

@testset "rectangle" begin

    domain = Domains.RectangleDomain(
        Geometry.XPoint(0)..Geometry.XPoint(2π),
        Geometry.YPoint(0)..Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )

    n1, n2 = 2, 2
    Nq = 4
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    coords = Fields.coordinate_field(space)
    sinxy = map(coords) do coord
        sin(coord.x + coord.y)
    end
    u = map(coords) do coord
        Geometry.UVVector(-coord.y, coord.x)
    end

    writevtk(joinpath(dir, "plane"), (sinxy = sinxy, u = u))

end
