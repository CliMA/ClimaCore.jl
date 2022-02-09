using Test

import ClimaCore

if haskey(ENV, "CI")
end
OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "static spectral element 2D cubed-sphere" begin
    R = 6.37122e6

    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, 6)
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    quad = ClimaCore.Spaces.Quadratures.GLL{5}()
    space = ClimaCore.Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = ClimaCore.Fields.coordinate_field(space)

    u = map(coords) do coord
        u0 = 20.0
        α0 = 45.0
        ϕ = coord.lat
        λ = coord.long

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        ClimaCore.Geometry.UVVector(uu, uv)
    end
    # TODO: add tests
end

@testset "static spectral element rectangle 2D" begin
    domain = ClimaCore.Domains.RectangleDomain(
        ClimaCore.Geometry.XPoint(0) .. ClimaCore.Geometry.XPoint(2π),
        ClimaCore.Geometry.YPoint(0) .. ClimaCore.Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )
    n1, n2 = 2, 2
    Nq = 4
    mesh = ClimaCore.Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology = ClimaCore.Topologies.Topology2D(mesh)
    #quad = ClimaCore.Spaces.Quadratures.GLL{Nq}()
    quad = ClimaCore.Spaces.Quadratures.ClosedUniform{Nq + 1}()
    space = ClimaCore.Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = ClimaCore.Fields.coordinate_field(space)

    sinxy = map(coords) do coord
        cos(coord.x + coord.y)
    end

    # TODO: add tests
end
