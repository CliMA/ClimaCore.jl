ENV["GKSwstype"] = "nul"
using Test
using IntervalSets

import Plots

import ClimaCore
import ClimaCorePlots

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "spectral element 2D cubed-sphere" begin
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

    field_fig = Plots.plot(u.components.data.:1)
    @test field_fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "2D_cubed_sphere_field.png")
    Plots.png(field_fig, fig_png)
    @test isfile(fig_png)
end

@testset "spectral element rectangle 2D" begin
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

    space_fig = Plots.plot(space)
    @test space_fig !== nothing

    sinxy = map(coords) do coord
        cos(coord.x + coord.y)
    end

    field_fig = Plots.plot(sinxy)
    @test field_fig !== nothing

    space_png = joinpath(OUTPUT_DIR, "2D_rectangle_space.png")
    field_png = joinpath(OUTPUT_DIR, "2D_rectangle_field.png")
    Plots.png(space_fig, space_png)
    Plots.png(field_fig, field_png)
    @test isfile(space_png)
    @test isfile(field_png)
end

@testset "hybrid finite difference / spectral element 2D" begin
    FT = Float64
    helem = 10
    velem = 40
    npoly = 4

    vertdomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint{FT}(0),
        ClimaCore.Geometry.ZPoint{FT}(1000);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = ClimaCore.Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.XPoint{FT}(-500) ..
        ClimaCore.Geometry.XPoint{FT}(500),
        periodic = true,
    )
    horzmesh = ClimaCore.Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = ClimaCore.Topologies.IntervalTopology(horzmesh)

    quad = ClimaCore.Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = ClimaCore.Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_center_space,
    )
    hv_face_space =
        ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    coords = ClimaCore.Fields.coordinate_field(hv_center_space)

    xcoords_fig = Plots.plot(coords.x)
    @test xcoords_fig !== nothing

    zcoords_fig = Plots.plot(coords.z)
    @test zcoords_fig !== nothing

    xcoords_png = joinpath(OUTPUT_DIR, "hybrid_xcoords_center_field.png")
    zcoords_png = joinpath(OUTPUT_DIR, "hybrid_zcoords_center_field.png")
    Plots.png(xcoords_fig, xcoords_png)
    Plots.png(zcoords_fig, zcoords_png)
    @test isfile(xcoords_png)
    @test isfile(zcoords_png)
end
