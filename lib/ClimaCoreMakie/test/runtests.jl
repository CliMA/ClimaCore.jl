using Test
using ClimaComms
using IntervalSets

import ClimaCore
import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Quadratures,
    Spaces,
    Fields,
    Geometry,
    Hypsography
using ClimaCoreMakie, Makie

using GLMakie
OUTPUT_DIR = mkpath(get(ENV, "CI_OUTPUT_DIR", tempname()))
@show OUTPUT_DIR

@testset "spectral element 2D cubed-sphere" begin
    R = 6.37122e6
    context = ClimaComms.SingletonCommsContext()

    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, 6)
    grid_topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{5}()
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

    fig = ClimaCoreMakie.fieldheatmap(u.components.data.:1)
    @test fig !== nothing

    fig_png = joinpath(OUTPUT_DIR, "2D_cubed_sphere.png")
    GLMakie.save(fig_png, fig)
    @test isfile(fig_png)
end

@testset "spectral element rectangle 2D" begin
    domain = Domains.RectangleDomain(
        Geometry.XPoint(0) .. Geometry.XPoint(2π),
        Geometry.YPoint(0) .. Geometry.YPoint(2π),
        x1periodic = true,
        x2periodic = true,
    )

    n1, n2 = 2, 2
    Nq = 4
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    grid_topology =
        Topologies.Topology2D(ClimaComms.SingletonCommsContext(), mesh)
    #quad = Quadratures.GLL{Nq}()
    quad = Quadratures.ClosedUniform{Nq + 1}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

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

    context = ClimaComms.context()
    domain = Domains.IntervalDomain(
        Geometry.XPoint(-1000.0),
        Geometry.XPoint(1000.0),
        periodic = true,
    )
    mesh = Meshes.IntervalMesh(domain, nelems = 32)
    topology = Topologies.IntervalTopology(mesh)
    quad = Quadratures.GLL{5}()
    horz_space = Spaces.SpectralElementSpace1D(topology, quad)
    horz_coords = Fields.coordinate_field(horz_space)

    z_surface = map(horz_coords) do coord
        Geometry.ZPoint(100 * exp(-(coord.x / 100)^2))
    end

    vert_domain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_names = (:bottom, :top),
    )
    vert_mesh = Meshes.IntervalMesh(vert_domain, nelems = 36)
    vtopology = Topologies.IntervalTopology(context, vert_mesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vtopology)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vert_center_space)


    face_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horz_space,
        vert_face_space,
        Hypsography.LinearAdaption(z_surface),
    )

    fcoords = Fields.coordinate_field(face_space)

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

    center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(face_space)

    ccoords = Fields.coordinate_field(center_space)
    pba = fieldcontourf!(Axis(gba[1, 1]), ccoords.x)
    Colorbar(gba[1, 2], pba)
    pbb = fieldheatmap!(Axis(gbb[1, 1]), ccoords.x)
    Colorbar(gbb[1, 2], pbb)

    fig_png = joinpath(OUTPUT_DIR, "extruded_topology.png")
    GLMakie.save(fig_png, f)
    @test isfile(fig_png)

end
