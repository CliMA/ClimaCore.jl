
using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Operators
using IntervalSets
using Test

@testset "3D box" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(vertmesh)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint(-500.0) .. Geometry.XPoint(500.0),
        Geometry.YPoint(-500.0) .. Geometry.YPoint(500.0),
        x1periodic = true,
        x2periodic = true,
    )

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.RectilinearMesh(horzdomain, 10, 10)
    horztopology = Topologies.Topology2D(horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)


    coords = Fields.coordinate_field(hv_center_space)

    xpts = range(Geometry.XPoint(-500.0), Geometry.XPoint(500.0), length = 21)
    ypts = range(Geometry.YPoint(-500.0), Geometry.YPoint(500.0), length = 21)
    zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)


    interp_x = Operators.interpolate_array(coords.x, xpts, ypts, zpts)
    @test interp_x ≈ [x.x for x in xpts, y in ypts, z in zpts]

    interp_y = Operators.interpolate_array(coords.y, xpts, ypts, zpts)
    @test interp_y ≈ [y.y for x in xpts, y in ypts, z in zpts]

    interp_z = Operators.interpolate_array(coords.z, xpts, ypts, zpts)
    @test interp_z[:, :, 2:(end - 1)] ≈
          [z.z for x in xpts, y in ypts, z in zpts[2:(end - 1)]]
    @test interp_z[:, :, 1] ≈
          [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
    @test interp_z[:, :, end] ≈
          [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts, y in ypts]
end
