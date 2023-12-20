using Test
using ClimaComms
using IntervalSets

import ClimaCore
using ClimaCoreMakie, Makie

using GLMakie
OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))
@show OUTPUT_DIR

@testset "spectral element 2D cubed-sphere" begin
    R = 6.37122e6

    domain = ClimaCore.Domains.SphereDomain(R)
    mesh = ClimaCore.Meshes.EquiangularCubedSphere(domain, 6)
    grid_topology = ClimaCore.Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        mesh,
    )
    quad = ClimaCore.Quadratures.GLL{5}()
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

    fig = ClimaCoreMakie.fieldheatmap(u.components.data.:1)
    @test fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "2D_cubed_sphere.png")
    GLMakie.save(fig_png, fig)
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
    grid_topology = ClimaCore.Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        mesh,
    )
    #quad = ClimaCore.Quadratures.GLL{Nq}()
    quad = ClimaCore.Quadratures.ClosedUniform{Nq + 1}()
    space = ClimaCore.Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = ClimaCore.Fields.coordinate_field(space)

    sinxy = map(coords) do coord
        cos(coord.x + coord.y)
    end

    fig = ClimaCoreMakie.fieldheatmap(sinxy)
    @test fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "2D_rectangle.png")
    GLMakie.save(fig_png, fig)
    @test isfile(fig_png)
end

@testset "extruded" begin

    domain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.XPoint(-1000.0),
        ClimaCore.Geometry.XPoint(1000.0),
        periodic = true,
    )
    mesh = ClimaCore.Meshes.IntervalMesh(domain, nelems = 32)
    topology = ClimaCore.Topologies.IntervalTopology(mesh)
    quad = ClimaCore.Quadratures.GLL{5}()
    horz_space = ClimaCore.Spaces.SpectralElementSpace1D(topology, quad)
    horz_coords = ClimaCore.Fields.coordinate_field(horz_space)

    z_surface = map(horz_coords) do coord
        100 * exp(-(coord.x / 100)^2)
    end

    vert_domain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint(0.0),
        ClimaCore.Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )
    vert_mesh = ClimaCore.Meshes.IntervalMesh(vert_domain, nelems = 36)
    vert_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(vert_mesh)
    vert_face_space =
        ClimaCore.Spaces.FaceFiniteDifferenceSpace(vert_center_space)


    face_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
        horz_space,
        vert_face_space,
        ClimaCore.Hypsography.LinearAdaption(z_surface),
    )

    fcoords = ClimaCore.Fields.coordinate_field(face_space)

    f = Figure()
    gaa = f[1, 1] = GridLayout()
    gab = f[1, 2] = GridLayout()
    gba = f[2, 1] = GridLayout()
    gbb = f[2, 2] = GridLayout()

    paa = fieldcontourf!(Axis(gaa[1, 1]), fcoords.x)
    Colorbar(gaa[1, 2], paa)
    pab = fieldheatmap!(Axis(gab[1, 1]), fcoords.x)
    Colorbar(gab[1, 2], pab)
    plot(fcoords.z)

    center_space =
        ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)

    ccoords = ClimaCore.Fields.coordinate_field(center_space)
    pba = fieldcontourf!(Axis(gba[1, 1]), ccoords.x)
    Colorbar(gba[1, 2], pba)
    pbb = fieldheatmap!(Axis(gbb[1, 1]), ccoords.x)
    Colorbar(gbb[1, 2], pbb)

    fig_png = joinpath(OUTPUT_DIR, "extruded_topology.png")
    GLMakie.save(fig_png, f)
    @test isfile(fig_png)

end
