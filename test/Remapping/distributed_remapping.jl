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
    Quadratures,
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

@testset "Utils" begin
    # batched_ranges(num_fields, buffer_length)
    @test Remapping.batched_ranges(1, 1) == [1:1]
    @test Remapping.batched_ranges(1, 2) == [1:1]
    @test Remapping.batched_ranges(2, 2) == [1:2]
    @test Remapping.batched_ranges(3, 2) == [1:2, 3:3]
end

on_gpu = device isa ClimaComms.CUDADevice
broken = true

if !on_gpu
    @testset "2D extruded" begin
        vertdomain = Domains.IntervalDomain(
            Geometry.ZPoint(0.0),
            Geometry.ZPoint(1000.0);
            boundary_names = (:bottom, :top),
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

        quad = Quadratures.GLL{4}()
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

        remapper = Remapping.Remapper(
            hv_center_space,
            hcoords,
            zcoords,
            buffer_length = 2,
        )

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

        # Remapping two fields
        interp_xx = Remapping.interpolate(remapper, [coords.x, coords.x])
        if ClimaComms.iamroot(context)
            @test interp_x ≈ interp_xx[:, :, 1]
            @test interp_x ≈ interp_xx[:, :, 2]
        end

        # Remapping three fields (more than the buffer length)
        interp_xxx =
            Remapping.interpolate(remapper, [coords.x, coords.x, coords.x])
        if ClimaComms.iamroot(context)
            @test interp_x ≈ interp_xxx[:, :, 1]
            @test interp_x ≈ interp_xxx[:, :, 2]
            @test interp_x ≈ interp_xxx[:, :, 3]
        end

        # Remapping in-place one field
        if !broken
            dest = zeros(21, 21)
            Remapping.interpolate!(dest, remapper, coords.x)
            if ClimaComms.iamroot(context)
                @test interp_x ≈ dest
            end
        end

        # Two fields
        dest = zeros(21, 21, 2)
        Remapping.interpolate!(dest, remapper, [coords.x, coords.x])
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest[:, :, 1]
            @test interp_x ≈ dest[:, :, 2]
        end

        # Three fields (more than buffer length)
        if !broken
            dest = zeros(21, 21, 3)
            Remapping.interpolate!(
                dest,
                remapper,
                [coords.x, coords.x, coords.x],
            )
            if ClimaComms.iamroot(context)
                @test interp_x ≈ dest[:, :, 1]
                @test interp_x ≈ dest[:, :, 2]
                @test interp_x ≈ dest[:, :, 3]
            end
        end
    end
end

@testset "3D box" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_names = (:bottom, :top),
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

    quad = Quadratures.GLL{4}()
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

    remapper =
        Remapping.Remapper(hv_center_space, hcoords, zcoords, buffer_length = 2)

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

    # Remapping two fields
    interp_xy = Remapping.interpolate(remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xy[:, :, :, 1]
        @test interp_y ≈ interp_xy[:, :, :, 2]
    end
    # Remapping three fields (more than the buffer length)
    interp_xyx = Remapping.interpolate(remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xyx[:, :, :, 1]
        @test interp_y ≈ interp_xyx[:, :, :, 2]
        @test interp_x ≈ interp_xyx[:, :, :, 3]
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    remapper_1field =
        on_gpu ? Remapping.Remapper(hv_center_space, hcoords, zcoords) :
        remapper
    dest = zeros(21, 21, 21)
    Remapping.interpolate!(dest, remapper_1field, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = zeros(21, 21, 21, 2)
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 21, 3)
        Remapping.interpolate!(dest, remapper, [coords.x, coords.y, coords.x])
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest[:, :, :, 1]
            @test interp_y ≈ dest[:, :, :, 2]
            @test interp_x ≈ dest[:, :, :, 3]
        end
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

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

    # Two fields
    interp_xy = Remapping.interpolate(horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_xy[:, :, 1] ≈ interp_x
        @test interp_xy[:, :, 2] ≈ interp_y
    end

    # Three fields
    interp_xyx =
        Remapping.interpolate(horiz_remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_xyx[:, :, 1] ≈ interp_x
        @test interp_xyx[:, :, 2] ≈ interp_y
        @test interp_xyx[:, :, 3] ≈ interp_x
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    if !broken
        dest = zeros(21, 21)
        Remapping.interpolate!(dest, remapper_1field, coords.x)
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest
        end
    end

    # Two fields
    dest = zeros(21, 21, 2)
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 3)
        Remapping.interpolate!(
            dest,
            horiz_remapper,
            [coords.x, coords.y, coords.x],
        )
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest[:, :, 1]
            @test interp_y ≈ dest[:, :, 2]
            @test interp_x ≈ dest[:, :, 3]
        end
    end
end


@testset "3D box - space filling curve" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_names = (:bottom, :top),
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

    quad = Quadratures.GLL{4}()
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

    remapper =
        Remapping.Remapper(hv_center_space, hcoords, zcoords, buffer_length = 2)

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

    # Remapping two fields
    interp_xy = Remapping.interpolate(remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xy[:, :, :, 1]
        @test interp_y ≈ interp_xy[:, :, :, 2]
    end
    # Remapping three fields (more than the buffer length)
    interp_xyx = Remapping.interpolate(remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xyx[:, :, :, 1]
        @test interp_y ≈ interp_xyx[:, :, :, 2]
        @test interp_x ≈ interp_xyx[:, :, :, 3]
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    remapper_1field =
        on_gpu ? Remapping.Remapper(hv_center_space, hcoords, zcoords) :
        remapper
    dest = zeros(21, 21, 21)
    Remapping.interpolate!(dest, remapper_1field, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = zeros(21, 21, 21, 2)
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 21, 3)
        Remapping.interpolate!(dest, remapper, [coords.x, coords.y, coords.x])
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest[:, :, :, 1]
            @test interp_y ≈ dest[:, :, :, 2]
            @test interp_x ≈ dest[:, :, :, 3]
        end
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

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

    # Two fields
    interp_xy = Remapping.interpolate(horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_xy[:, :, 1] ≈ interp_x
        @test interp_xy[:, :, 2] ≈ interp_y
    end

    # Three fields
    interp_xyx =
        Remapping.interpolate(horiz_remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_xyx[:, :, 1] ≈ interp_x
        @test interp_xyx[:, :, 2] ≈ interp_y
        @test interp_xyx[:, :, 3] ≈ interp_x
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    if !broken
        dest = zeros(21, 21)
        Remapping.interpolate!(dest, remapper_1field, coords.x)
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest
        end
    end

    # Two fields
    dest = zeros(21, 21, 2)
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 3)
        Remapping.interpolate!(
            dest,
            horiz_remapper,
            [coords.x, coords.y, coords.x],
        )
        if ClimaComms.iamroot(context)
            @test interp_x ≈ dest[:, :, 1]
            @test interp_y ≈ dest[:, :, 2]
            @test interp_x ≈ dest[:, :, 3]
        end
    end
end

@testset "3D sphere" begin
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(0.0),
        Geometry.ZPoint(1000.0);
        boundary_names = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 30)
    verttopo = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(ClimaComms.device()),
        vertmesh,
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(verttopo)

    horzdomain = Domains.SphereDomain(1e6)

    quad = Quadratures.GLL{4}()
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

    remapper =
        Remapping.Remapper(hv_center_space, hcoords, zcoords, buffer_length = 2)

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

    # Remapping two fields
    interp_long_lat =
        Remapping.interpolate(remapper, [sind.(coords.long), sind.(coords.lat)])
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ interp_long_lat[:, :, :, 1]
        @test interp_sin_lat ≈ interp_long_lat[:, :, :, 2]
    end
    # Remapping three fields (more than the buffer length)
    interp_long_lat_long = Remapping.interpolate(
        remapper,
        [sind.(coords.long), sind.(coords.lat), sind.(coords.long)],
    )
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ interp_long_lat_long[:, :, :, 1]
        @test interp_sin_lat ≈ interp_long_lat_long[:, :, :, 2]
        @test interp_sin_long ≈ interp_long_lat_long[:, :, :, 3]
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    remapper_1field =
        on_gpu ? Remapping.Remapper(hv_center_space, hcoords, zcoords) :
        remapper
    dest = zeros(21, 21, 21)
    Remapping.interpolate!(dest, remapper_1field, sind.(coords.long))
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ dest
    end

    # Two fields
    dest = zeros(21, 21, 21, 2)
    Remapping.interpolate!(
        dest,
        remapper,
        [sind.(coords.long), sind.(coords.lat)],
    )
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ dest[:, :, :, 1]
        @test interp_sin_lat ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 21, 3)
        Remapping.interpolate!(
            dest,
            remapper,
            [sind.(coords.long), sind.(coords.lat), sind.(coords.long)],
        )
        if ClimaComms.iamroot(context)
            @test interp_sin_long ≈ dest[:, :, :, 1]
            @test interp_sin_lat ≈ dest[:, :, :, 2]
            @test interp_sin_long ≈ dest[:, :, :, 3]
        end
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

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

    # Two fields
    interp_sin_long_lat = Remapping.interpolate(
        horiz_remapper,
        [sind.(coords.long), sind.(coords.lat)],
    )
    if ClimaComms.iamroot(context)
        @test interp_sin_long_lat[:, :, 1] ≈ interp_sin_long
        @test interp_sin_long_lat[:, :, 2] ≈ interp_sin_lat
    end

    # Three fields
    interp_sin_long_lat_long = Remapping.interpolate(
        horiz_remapper,
        [sind.(coords.long), sind.(coords.lat), sind.(coords.long)],
    )
    if ClimaComms.iamroot(context)
        @test interp_sin_long_lat_long[:, :, 1] ≈ interp_sin_long
        @test interp_sin_long_lat_long[:, :, 2] ≈ interp_sin_lat
        @test interp_sin_long_lat_long[:, :, 3] ≈ interp_sin_long
    end

    # Remapping in-place one field
    #
    # We have to change remapper for GPU to make sure it works for when have have only one
    # field
    if !broken
        dest = zeros(21, 21)
        Remapping.interpolate!(dest, remapper_1field, sind.(coords.long))
        if ClimaComms.iamroot(context)
            @test interp_sin_long ≈ dest
        end
    end

    # Two fields
    dest = zeros(21, 21, 2)
    Remapping.interpolate!(
        dest,
        horiz_remapper,
        [sind.(coords.long), sind.(coords.lat)],
    )
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ dest[:, :, 1]
        @test interp_sin_lat ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    if !broken
        dest = zeros(21, 21, 3)
        Remapping.interpolate!(
            dest,
            horiz_remapper,
            [sind.(coords.long), sind.(coords.lat), sind.(coords.long)],
        )
        if ClimaComms.iamroot(context)
            @test interp_sin_long ≈ dest[:, :, 1]
            @test interp_sin_lat ≈ dest[:, :, 2]
            @test interp_sin_long ≈ dest[:, :, 3]
        end
    end
end
