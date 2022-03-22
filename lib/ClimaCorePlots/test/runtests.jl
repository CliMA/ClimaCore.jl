ENV["GKSwstype"] = "nul"
using Test
using IntervalSets

import Plots

import ClimaCore
import ClimaCorePlots

OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))

@testset "spectral element 1D" begin
    domain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.XPoint(0.0) .. ClimaCore.Geometry.XPoint(π),
        boundary_tags = (:left, :right),
    )
    mesh = ClimaCore.Meshes.IntervalMesh(domain; nelems = 5)
    grid_topology = ClimaCore.Topologies.IntervalTopology(mesh)
    quad = ClimaCore.Spaces.Quadratures.GLL{5}()
    space = ClimaCore.Spaces.SpectralElementSpace1D(grid_topology, quad)
    coords = ClimaCore.Fields.coordinate_field(space)

    u = sin.(π .* coords.x)

    field_fig = Plots.plot(u)
    @test field_fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "1D_field.png")
    Plots.png(field_fig, fig_png)
    @test isfile(fig_png)
end

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

@testset "spectral element 3D extruded cubed-sphere" begin
    FT = Float64
    R = 6.37122e6
    velem = 40

    horz_domain = ClimaCore.Domains.SphereDomain(R)
    horz_mesh = ClimaCore.Meshes.EquiangularCubedSphere(horz_domain, 6)
    horz_grid_topology = ClimaCore.Topologies.Topology2D(horz_mesh)
    quad = ClimaCore.Spaces.Quadratures.GLL{5}()
    horz_space =
        ClimaCore.Spaces.SpectralElementSpace2D(horz_grid_topology, quad)

    vertdomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint{FT}(0),
        ClimaCore.Geometry.ZPoint{FT}(1000);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = ClimaCore.Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(vertmesh)

    hv_center_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
        horz_space,
        vert_center_space,
    )
    coords = ClimaCore.Fields.coordinate_field(hv_center_space)

    u = map(coords) do coord
        u0 = 20.0
        α0 = 45.0
        ϕ = coord.lat
        λ = coord.long
        z = coord.z

        uu = u0 * (cosd(α0) * cosd(ϕ) + sind(α0) * cosd(λ) * sind(ϕ))
        uv = -u0 * sind(α0) * sind(λ)
        uw = z
        ClimaCore.Geometry.UVWVector(uu, uv, uw)
    end

    v_field_level3_fig = Plots.plot(u.components.data.:2, level = 3)
    @test v_field_level3_fig !== nothing

    w_field_level10_fig = Plots.plot(u.components.data.:3, level = 10)
    @test w_field_level10_fig !== nothing

    v_field_level3_fig_png =
        joinpath(OUTPUT_DIR, "3D_cubed_sphere_v_field_level3.png")
    Plots.png(v_field_level3_fig, v_field_level3_fig_png)
    @test isfile(v_field_level3_fig_png)

    w_field_level10_fig_png =
        joinpath(OUTPUT_DIR, "3D_cubed_sphere_w_field_level10.png")
    Plots.png(w_field_level10_fig, w_field_level10_fig_png)
    @test isfile(w_field_level10_fig_png)
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

@testset "hybrid finite difference / spectral element 3D" begin
    FT = Float64
    xelem = 10
    yelem = 5
    velem = 40
    npoly = 4

    vertdomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint{FT}(0),
        ClimaCore.Geometry.ZPoint{FT}(1000);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = ClimaCore.Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(vertmesh)

    xdomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.XPoint{FT}(-500) ..
        ClimaCore.Geometry.XPoint{FT}(500),
        periodic = true,
    )
    ydomain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.YPoint{FT}(-100) ..
        ClimaCore.Geometry.YPoint{FT}(100),
        periodic = true,
    )

    horzdomain = ClimaCore.Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = ClimaCore.Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology = ClimaCore.Topologies.Topology2D(horzmesh)

    quad = ClimaCore.Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = ClimaCore.Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_center_space,
    )
    hv_face_space =
        ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    coords = ClimaCore.Fields.coordinate_field(hv_center_space)

    xcoords_fig = Plots.plot(coords.x, slice = (:, 0.0, :))
    @test xcoords_fig !== nothing

    ycoords_fig = Plots.plot(coords.y, slice = (0.0, :, :))
    @test ycoords_fig !== nothing

    xzcoords_fig = Plots.plot(coords.z, slice = (:, 0.0, :))
    @test xzcoords_fig !== nothing

    yzcoords_fig = Plots.plot(coords.z, slice = (0.0, :, :))
    @test yzcoords_fig !== nothing

    xcoords_png = joinpath(OUTPUT_DIR, "hybrid_xcoords_center_field.png")
    ycoords_png = joinpath(OUTPUT_DIR, "hybrid_ycoords_center_field.png")
    xzcoords_png = joinpath(OUTPUT_DIR, "hybrid_xzcoords_center_field.png")
    yzcoords_png = joinpath(OUTPUT_DIR, "hybrid_yzcoords_center_field.png")

    Plots.png(xcoords_fig, xcoords_png)
    Plots.png(ycoords_fig, ycoords_png)
    Plots.png(xzcoords_fig, xzcoords_png)
    Plots.png(yzcoords_fig, yzcoords_png)

    @test isfile(xcoords_png)
    @test isfile(ycoords_png)
    @test isfile(xzcoords_png)
    @test isfile(yzcoords_png)
end
