using Logging
using Test
using IntervalSets

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    Remapping,
    Hypsography
using ClimaComms
const context = ClimaComms.context()
const pid, nprocs = ClimaComms.init(context)
const device = ClimaComms.device()

# log output only from root process
logger_stream = ClimaComms.iamroot(context) ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

if !(device isa ClimaComms.CUDADevice)
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

        xpts = range(-500.0, 500.0, length = 21)
        zpts = range(0.0, 1000.0, length = 21)
        hcoords = [Geometry.XPoint(x) for x in xpts]
        zcoords = [Geometry.ZPoint(z) for z in zpts]

        remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space)

        interp_x = Remapping.interpolate(remapper, coords.x)
        if ClimaComms.iamroot(context)
            @test interp_x ≈ [x for x in xpts, z in zpts]
        end

        interp_z = Remapping.interpolate(remapper, coords.z)
        expected_z = [z for x in xpts, z in zpts]
        if ClimaComms.iamroot(context)
            @test interp_z[:, 2:(end - 1)] ≈ expected_z[:, 2:(end - 1)]
            @test interp_z[:, 1] ≈
                  [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts]
            @test interp_z[:, end] ≈
                  [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts]
        end

        # With hypsography
        z_surface = cosd.(Fields.coordinate_field(horzspace).x) .+ 20
        hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_center_space,
            Hypsography.LinearAdaption(z_surface),
        )

        coords = Fields.coordinate_field(hv_center_space)

        remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space;)

        function ref_z(x, z)
            z_surface = cosd(x) + 20
            z_top = vertdomain.coord_max.z
            return (z - z_surface) / (1 - z_surface / z_top)
        end

        val_or_NaN(x, z, val) = ref_z(x, z) >= 0 ? val : NaN

        interp_x = Remapping.interpolate(remapper, coords.x; physical_z = true)
        if ClimaComms.iamroot(context)
            @test interp_x ≈ [val_or_NaN(x, z, x) for x in xpts, z in zpts] nans = true
        end
        interp_z = Remapping.interpolate(remapper, coords.z; physical_z = true)
        expected_z = [val_or_NaN(x, z, z) for x in xpts, z in zpts]
        if ClimaComms.iamroot(context)
            @test interp_z[:, 2:(end - 1)] ≈ expected_z[:, 2:(end - 1)] nans =
                true

            z_bottom = 1000.0 * (0 / 30 + 1 / 30) / 2
            @test interp_z[:, 1] ≈
                  [val_or_NaN(x, z_bottom, z_bottom) for x in xpts] nans = true

            z_top = 1000.0 * (29 / 30 + 30 / 30) / 2
            @test interp_z[:, end] ≈ [val_or_NaN(x, z_top, z_top) for x in xpts] rtol = 0.02 nans = true
        end
    end
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

    xpts = range(-500.0, 500.0, length = 21)
    ypts = range(-500.0, 500.0, length = 21)
    zpts = range(0.0, 1000.0, length = 21)
    hcoords = [Geometry.XYPoint(x, y) for x in xpts, y in ypts]
    zcoords = [Geometry.ZPoint(z) for z in zpts]

    remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space)

    interp_x = Remapping.interpolate(remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ [x for x in xpts, y in ypts, z in zpts]
    end

    interp_y = Remapping.interpolate(remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test interp_y ≈ [y for x in xpts, y in ypts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, y in ypts, z in zpts]
    if ClimaComms.iamroot(context)
        @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)]
        @test interp_z[:, :, 1] ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
        @test interp_z[:, :, end] ≈
              [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts, y in ypts]
    end
    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(hcoords, nothing, horiz_space)

    coords = Fields.coordinate_field(horiz_space)

    interp_x = Remapping.interpolate(horiz_remapper, coords.x)
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test interp_x ≈ [x for x in xpts, y in ypts]
    end

    interp_y = Remapping.interpolate(horiz_remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test interp_y ≈ [y for x in xpts, y in ypts]
    end

    if !(device isa ClimaComms.CUDADevice)
        # With hypsography
        z_surface =
            cosd.(Fields.coordinate_field(horzspace).x) .+
            cosd.(Fields.coordinate_field(horzspace).y) .+ 20
        hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_center_space,
            Hypsography.LinearAdaption(z_surface),
        )

        coords = Fields.coordinate_field(hv_center_space)

        remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space;)

        function ref_z(x, y, z)
            z_surface = cosd(x) + cosd(y) + 20
            z_top = vertdomain.coord_max.z
            return (z - z_surface) / (1 - z_surface / z_top)
        end

        val_or_NaN(x, y, z, val) = ref_z(x, y, z) >= 0 ? val : NaN

        interp_x = Remapping.interpolate(remapper, coords.x; physical_z = true)
        if ClimaComms.iamroot(context)
            @test interp_x ≈
                  [val_or_NaN(x, y, z, x) for x in xpts, y in ypts, z in zpts] nans =
                true
        end

        interp_y = Remapping.interpolate(remapper, coords.y; physical_z = true)
        if ClimaComms.iamroot(context)
            @test interp_y ≈
                  [val_or_NaN(x, y, z, y) for x in xpts, y in ypts, z in zpts] nans =
                true
        end

        interp_z = Remapping.interpolate(remapper, coords.z; physical_z = true)
        expected_z =
            [val_or_NaN(x, y, z, z) for x in xpts, y in ypts, z in zpts]
        if ClimaComms.iamroot(context)
            @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)] nans =
                true

            z_bottom = 1000.0 * (0 / 30 + 1 / 30) / 2
            @test interp_z[:, :, 1] ≈ [
                val_or_NaN(x, y, z_bottom, z_bottom) for x in xpts, y in ypts
            ] nans = true

            z_top = 1000.0 * (29 / 30 + 30 / 30) / 2
            @test interp_z[:, :, end] ≈
                  [val_or_NaN(x, y, z_top, z_top) for x in xpts, y in ypts] rtol =
                0.02 nans = true
        end
    end
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

    xpts = range(-500.0, 500.0, length = 21)
    ypts = range(-500.0, 500.0, length = 21)
    zpts = range(0.0, 1000.0, length = 21)
    hcoords = [Geometry.XYPoint(x, y) for x in xpts, y in ypts]
    zcoords = [Geometry.ZPoint(z) for z in zpts]

    remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space)

    interp_x = Remapping.interpolate(remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ [x for x in xpts, y in ypts, z in zpts]
    end

    interp_y = Remapping.interpolate(remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test interp_y ≈ [y for x in xpts, y in ypts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, y in ypts, z in zpts]
    if ClimaComms.iamroot(context)
        @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)]
        @test interp_z[:, :, 1] ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
        @test interp_z[:, :, end] ≈
              [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts, y in ypts]
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(hcoords, nothing, horiz_space)

    coords = Fields.coordinate_field(horiz_space)

    interp_x = Remapping.interpolate(horiz_remapper, coords.x)
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test interp_x ≈ [x for x in xpts, y in ypts]
    end

    interp_y = Remapping.interpolate(horiz_remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test interp_y ≈ [y for x in xpts, y in ypts]
    end

    if !(device isa ClimaComms.CUDADevice)
        # With hypsography
        z_surface =
            cosd.(Fields.coordinate_field(horzspace).x) .+
            cosd.(Fields.coordinate_field(horzspace).y) .+ 20
        hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_center_space,
            Hypsography.LinearAdaption(z_surface),
        )

        coords = Fields.coordinate_field(hv_center_space)

        remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space;)

        function ref_z(x, y, z)
            z_surface = cosd(x) + cosd(y) + 20
            z_top = vertdomain.coord_max.z
            return (z - z_surface) / (1 - z_surface / z_top)
        end

        val_or_NaN(x, y, z, val) = ref_z(x, y, z) >= 0 ? val : NaN

        interp_x = Remapping.interpolate(remapper, coords.x; physical_z = true)
        if ClimaComms.iamroot(context)
            @test interp_x ≈
                  [val_or_NaN(x, y, z, x) for x in xpts, y in ypts, z in zpts] nans =
                true
        end

        interp_y = Remapping.interpolate(remapper, coords.y; physical_z = true)
        if ClimaComms.iamroot(context)
            @test interp_y ≈
                  [val_or_NaN(x, y, z, y) for x in xpts, y in ypts, z in zpts] nans =
                true
        end

        interp_z = Remapping.interpolate(remapper, coords.z; physical_z = true)
        expected_z =
            [val_or_NaN(x, y, z, z) for x in xpts, y in ypts, z in zpts]
        if ClimaComms.iamroot(context)
            @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)] nans =
                true

            z_bottom = 1000.0 * (0 / 30 + 1 / 30) / 2
            @test interp_z[:, :, 1] ≈ [
                val_or_NaN(x, y, z_bottom, z_bottom) for x in xpts, y in ypts
            ] nans = true

            z_top = 1000.0 * (29 / 30 + 30 / 30) / 2
            @test interp_z[:, :, end] ≈
                  [val_or_NaN(x, y, z_top, z_top) for x in xpts, y in ypts] rtol =
                0.02 nans = true
        end
    end
end

@testset "3D sphere" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(ClimaComms.device()),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.SphereDomain(1e6)

    quad = Spaces.Quadratures.GLL{4}()
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, 6)
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    longpts = range(-120.0, 120.0, 21)
    latpts = range(-80.0, 80.0, 21)
    zpts = range(0.0, 1000.0, 21)
    hcoords =
        [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
    zcoords = [Geometry.ZPoint(z) for z in zpts]

    remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space)

    coords = Fields.coordinate_field(hv_center_space)

    interp_sin_long = Remapping.interpolate(remapper, sind.(coords.long))
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈
              [sind(x) for x in longpts, y in latpts, z in zpts] rtol = 0.01
    end

    interp_sin_lat = Remapping.interpolate(remapper, sind.(coords.lat))
    if ClimaComms.iamroot(context)
        @test interp_sin_lat ≈
              [sind(y) for x in longpts, y in latpts, z in zpts] rtol = 0.01
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in longpts, y in latpts, z in zpts]
    if ClimaComms.iamroot(context)
        @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)]
        @test interp_z[:, :, 1] ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in longpts, y in latpts]
        @test interp_z[:, :, end] ≈
              [1000.0 * (29 / 30 + 30 / 30) / 2 for x in longpts, y in latpts]
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(hcoords, nothing, horiz_space)

    coords = Fields.coordinate_field(horiz_space)

    interp_sin_long = Remapping.interpolate(horiz_remapper, sind.(coords.long))
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ [sind(x) for x in longpts, y in latpts] rtol = 0.01
    end

    interp_sin_lat = Remapping.interpolate(horiz_remapper, sind.(coords.lat))
    if ClimaComms.iamroot(context)
        @test interp_sin_lat ≈ [sind(y) for x in longpts, y in latpts] rtol = 0.01
    end

    if !(device isa ClimaComms.CUDADevice)
        # With hypsography
        #
        # When we have hypsography, we should allow for larger error because elements close to
        # the surface are not reconstructed well
        z_surface =
            cosd.(Fields.coordinate_field(horzspace).lat) .+
            cosd.(Fields.coordinate_field(horzspace).long) .+ 20
        hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_center_space,
            Hypsography.LinearAdaption(z_surface),
        )

        coords = Fields.coordinate_field(hv_center_space)

        remapper = Remapping.Remapper(hcoords, zcoords, hv_center_space)

        function ref_z(x, y, z)
            z_surface = cosd(x) + cosd(y) + 20
            z_top = vertdomain.coord_max.z
            return (z - z_surface) / (1 - z_surface / z_top)
        end

        val_or_NaN(x, y, z, val) = ref_z(x, y, z) >= 0 ? val : NaN

        interp_sin_long = Remapping.interpolate(
            remapper,
            sind.(coords.long);
            physical_z = true,
        )

        if ClimaComms.iamroot(context)
            @test interp_sin_long ≈ [
                val_or_NaN(x, y, z, sind(x)) for x in longpts, y in latpts,
                z in zpts
            ] rtol = 0.06 nans = true
        end

        interp_sin_lat = Remapping.interpolate(
            remapper,
            sind.(coords.lat);
            physical_z = true,
        )
        if ClimaComms.iamroot(context)
            # We have to add atol because we have an element that is identically zero
            @test interp_sin_lat ≈ [
                val_or_NaN(x, y, z, sind(y)) for x in longpts, y in latpts,
                z in zpts
            ] rtol = 0.01 nans = true atol = 1e-15
        end

        interp_z = Remapping.interpolate(remapper, coords.z, physical_z = true)

        FT = Spaces.undertype(hv_center_space)

        expected_z = [z for x in longpts, y in latpts, z in zpts]

        if ClimaComms.iamroot(context)
            @test interp_z[:, :, 2:(end - 1)] ≈ expected_z[:, :, 2:(end - 1)]

            z_bottom = 1000.0 * (0 / 30 + 1 / 30) / 2
            @test interp_z[:, :, 1] ≈ [
                val_or_NaN(x, y, z_bottom, z_bottom) for x in longpts,
                y in latpts
            ] nans = true

            z_top = 1000.0 * (29 / 30 + 30 / 30) / 2
            @test interp_z[:, :, end] ≈
                  [val_or_NaN(x, y, z_top, z_top) for x in longpts, y in latpts] nans =
                true rtol = 0.02
        end
    end
end
