
using ClimaComms
using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Fields, Remapping
using IntervalSets
using Test

device = ClimaComms.CPUDevice()

@testset "2D extruded" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint(-500.0) .. Geometry.XPoint(500.0),
        periodic = true,
    )

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = 10)
    horztopology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        horzmesh,
    )
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)


    coords = Fields.coordinate_field(hv_center_space)

    xpts = range(Geometry.XPoint(-500.0), Geometry.XPoint(500.0), length = 21)
    zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)


    interp_x = Remapping.interpolate_array(coords.x, xpts, zpts)
    @test interp_x ≈ [x.x for x in xpts, z in zpts]

    interp_z = Remapping.interpolate_array(coords.z, xpts, zpts)
    @test interp_z[:, 2:(end - 1)] ≈ [z.z for x in xpts, z in zpts[2:(end - 1)]]
    @test interp_z[:, 1] ≈ [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts]
    @test interp_z[:, end] ≈ [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts]

end


@testset "3D box" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint(-500.0) .. Geometry.XPoint(500.0),
        Geometry.YPoint(-500.0) .. Geometry.YPoint(500.0),
        x1periodic = true,
        x2periodic = true,
    )

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.RectilinearMesh(horzdomain, 10, 10)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(device),
        horzmesh,
    )
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)


    coords = Fields.coordinate_field(hv_center_space)

    xpts = range(Geometry.XPoint(-500.0), Geometry.XPoint(500.0), length = 21)
    ypts = range(Geometry.YPoint(-500.0), Geometry.YPoint(500.0), length = 21)
    zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)


    interp_x = Remapping.interpolate_array(coords.x, xpts, ypts, zpts)
    @test interp_x ≈ [x.x for x in xpts, y in ypts, z in zpts]

    interp_y = Remapping.interpolate_array(coords.y, xpts, ypts, zpts)
    @test interp_y ≈ [y.y for x in xpts, y in ypts, z in zpts]

    interp_z = Remapping.interpolate_array(coords.z, xpts, ypts, zpts)
    @test interp_z[:, :, 2:(end - 1)] ≈
          [z.z for x in xpts, y in ypts, z in zpts[2:(end - 1)]]
    @test interp_z[:, :, 1] ≈
          [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
    @test interp_z[:, :, end] ≈
          [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts, y in ypts]
end

@testset "3D box - space filling curve" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint(-500.0) .. Geometry.XPoint(500.0),
        Geometry.YPoint(-500.0) .. Geometry.YPoint(500.0),
        x1periodic = true,
        x2periodic = true,
    )

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.RectilinearMesh(horzdomain, 10, 10)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(device),
        horzmesh,
        Topologies.spacefillingcurve(horzmesh),
    )
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)


    coords = Fields.coordinate_field(hv_center_space)

    xpts = range(Geometry.XPoint(-500.0), Geometry.XPoint(500.0), length = 21)
    ypts = range(Geometry.YPoint(-500.0), Geometry.YPoint(500.0), length = 21)
    zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)


    interp_x = Remapping.interpolate_array(coords.x, xpts, ypts, zpts)
    @test interp_x ≈ [x.x for x in xpts, y in ypts, z in zpts]

    interp_y = Remapping.interpolate_array(coords.y, xpts, ypts, zpts)
    @test interp_y ≈ [y.y for x in xpts, y in ypts, z in zpts]

    interp_z = Remapping.interpolate_array(coords.z, xpts, ypts, zpts)
    @test interp_z[:, :, 2:(end - 1)] ≈
          [z.z for x in xpts, y in ypts, z in zpts[2:(end - 1)]]
    @test interp_z[:, :, 1] ≈
          [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
    @test interp_z[:, :, end] ≈
          [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts, y in ypts]
end


@testset "3D sphere" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.SphereDomain(1e6)

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 6)
    horztopology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(device),
        horzmesh,
    )
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)


    coords = Fields.coordinate_field(hv_center_space)

    longpts = range(
        Geometry.LongPoint(-180.0),
        Geometry.LongPoint(180.0),
        length = 21,
    )
    latpts =
        range(Geometry.LatPoint(-80.0), Geometry.LatPoint(80.0), length = 21)
    zpts = range(Geometry.ZPoint(0.0), Geometry.ZPoint(1000.0), length = 21)


    interp_sin_long =
        Remapping.interpolate_array(sind.(coords.long), longpts, latpts, zpts)
    @test interp_sin_long ≈
          [sind(x.long) for x in longpts, y in latpts, z in zpts] rtol = 0.01

    interp_lat = Remapping.interpolate_array(coords.lat, longpts, latpts, zpts)
    @test interp_lat ≈ [y.lat for x in longpts, y in latpts, z in zpts] rtol = 0.01

    interp_z = Remapping.interpolate_array(coords.z, longpts, latpts, zpts)

    @test interp_z[:, :, 2:(end - 1)] ≈
          [z.z for x in longpts, y in latpts, z in zpts[2:(end - 1)]]
    @test interp_z[:, :, 1] ≈
          [1000.0 * (0 / 30 + 1 / 30) / 2 for x in longpts, y in latpts]
    @test interp_z[:, :, end] ≈
          [1000.0 * (29 / 30 + 30 / 30) / 2 for x in longpts, y in latpts]
end
