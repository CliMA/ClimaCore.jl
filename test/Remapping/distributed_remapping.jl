#=
julia --project
using Revise; include("test/Remapping/distributed_remapping.jl")
=#
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
    Hypsography,
    CommonSpaces
using ClimaComms
ClimaComms.@import_required_backends
const context = ClimaComms.context()
const pid, nprocs = ClimaComms.init(context)
const device = ClimaComms.device()
ArrayType = ClimaComms.array_type(device)

# log output only from root process
logger_stream = ClimaComms.iamroot(context) ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

with_mpi = context isa ClimaComms.MPICommsContext

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

    remapper =
        Remapping.Remapper(hv_center_space, hcoords, zcoords, buffer_length = 2)

    interp_x = Remapping.interpolate(remapper, coords.x)
    interp_x2 = Remapping.interpolate(coords.x, hcoords, zcoords)
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, z in zpts]
        @test Array(interp_x2) ≈ [x for x in xpts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, z in zpts]
    if ClimaComms.iamroot(context)
        @test Array(interp_z[:, 2:(end - 1)]) ≈ expected_z[:, 2:(end - 1)]
        @test Array(interp_z[:, 1]) ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts]
        @test Array(interp_z[:, end]) ≈
              [1000.0 * (29 / 30 + 30 / 30) / 2 for x in xpts]
    end

    # Remapping two fields
    interp_xx = Remapping.interpolate(remapper, [coords.x, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xx[:, :, 1]
        @test interp_x ≈ interp_xx[:, :, 2]
    end

    # Remapping three fields (more than the buffer length)
    interp_xxx = Remapping.interpolate(remapper, [coords.x, coords.x, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ interp_xxx[:, :, 1]
        @test interp_x ≈ interp_xxx[:, :, 2]
        @test interp_x ≈ interp_xxx[:, :, 3]
    end

    # Remapping in-place one field
    dest = ArrayType(zeros(21, 21))
    Remapping.interpolate!(dest, remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 2))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_x ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 3))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.x, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_x ≈ dest[:, :, 2]
        @test interp_x ≈ dest[:, :, 3]
    end

    # Convenience interpolate (just checking the shape, correctness of the
    # interpolation is checked elsewhere)
    interp_conv = Remapping.interpolate(coords.x)
    if ClimaComms.iamroot(context)
        @test size(interp_conv) == (180, 30)
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
    horztopology = Topologies.Topology2D(context, horzmesh)
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
    interp_x2 = Remapping.interpolate(coords.x, hcoords, zcoords)
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts, z in zpts]
        @test Array(interp_x2) ≈ [x for x in xpts, y in ypts, z in zpts]
    end

    interp_y = Remapping.interpolate(remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, y in ypts, z in zpts]
    if ClimaComms.iamroot(context)
        @test Array(interp_z[:, :, 2:(end - 1)]) ≈ expected_z[:, :, 2:(end - 1)]
        @test Array(interp_z[:, :, 1]) ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
        @test Array(interp_z[:, :, end]) ≈
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
    dest = ArrayType(zeros(21, 21, 21))
    Remapping.interpolate!(dest, remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 21, 2))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 21, 3))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
        @test interp_x ≈ dest[:, :, :, 3]
    end

    # Convenience interpolate (just checking the shape, correctness of the
    # interpolation is checked elsewhere)
    interp_conv = Remapping.interpolate(coords.x; zresolution = 50)
    if ClimaComms.iamroot(context)
        @test size(interp_conv) == (180, 180, 50)
    end

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

    coords = Fields.coordinate_field(horiz_space)

    interp_x = Remapping.interpolate(horiz_remapper, coords.x)
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts]
    end

    interp_y = Remapping.interpolate(horiz_remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts]
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
    dest = ArrayType(zeros(21, 21))
    Remapping.interpolate!(dest, horiz_remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end


    # Two fields
    dest = ArrayType(zeros(21, 21, 2))
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 3))
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
        @test interp_x ≈ dest[:, :, 3]
    end

    # Convenience interpolate horizontal (just checking the shape, correctness
    # of the interpolation is checked elsewhere)
    interp_conv = Remapping.interpolate(coords.x)
    if ClimaComms.iamroot(context)
        @test size(interp_conv) == (180, 180)
    end
end

@testset "3D box - bilinear horizontal" begin
    # Same setup as "3D box" but with horizontal_method = BilinearRemapping().
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
    horztopology = Topologies.Topology2D(context, horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    coords = Fields.coordinate_field(hv_center_space)

    xpts = range(-500.0, 500.0, length = 21)
    ypts = range(-500.0, 500.0, length = 21)
    zpts = range(0.0, 1000.0, length = 21)
    hcoords = [Geometry.XYPoint(x, y) for x in xpts, y in ypts]
    zcoords = [Geometry.ZPoint(z) for z in zpts]

    remapper = Remapping.Remapper(
        hv_center_space,
        hcoords,
        zcoords;
        buffer_length = 2,
        horizontal_method = Remapping.BilinearRemapping(),
    )

    interp_x = Remapping.interpolate(remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts, z in zpts]
    end

    interp_y = Remapping.interpolate(remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, y in ypts, z in zpts]
    if ClimaComms.iamroot(context)
        @test Array(interp_z[:, :, 2:(end - 1)]) ≈ expected_z[:, :, 2:(end - 1)]
        @test Array(interp_z[:, :, 1]) ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
        @test Array(interp_z[:, :, end]) ≈
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
    dest = ArrayType(zeros(21, 21, 21))
    Remapping.interpolate!(dest, remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 21, 2))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 21, 3))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
        @test interp_x ≈ dest[:, :, :, 3]
    end

    # Horizontal-only with bilinear
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(
        horiz_space,
        hcoords;
        buffer_length = 2,
        horizontal_method = Remapping.BilinearRemapping(),
    )

    coords_h = Fields.coordinate_field(horiz_space)

    interp_x = Remapping.interpolate(horiz_remapper, coords_h.x)
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts]
    end

    interp_y = Remapping.interpolate(horiz_remapper, coords_h.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts]
    end

    # Two fields
    interp_xy = Remapping.interpolate(horiz_remapper, [coords_h.x, coords_h.y])
    if ClimaComms.iamroot(context)
        @test interp_xy[:, :, 1] ≈ interp_x
        @test interp_xy[:, :, 2] ≈ interp_y
    end

    # Three fields
    interp_xyx =
        Remapping.interpolate(horiz_remapper, [coords_h.x, coords_h.y, coords_h.x])
    if ClimaComms.iamroot(context)
        @test interp_xyx[:, :, 1] ≈ interp_x
        @test interp_xyx[:, :, 2] ≈ interp_y
        @test interp_xyx[:, :, 3] ≈ interp_x
    end

    # Remapping in-place one field
    dest = ArrayType(zeros(21, 21))
    Remapping.interpolate!(dest, horiz_remapper, coords_h.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 2))
    Remapping.interpolate!(dest, horiz_remapper, [coords_h.x, coords_h.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 3))
    Remapping.interpolate!(dest, horiz_remapper, [coords_h.x, coords_h.y, coords_h.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
        @test interp_x ≈ dest[:, :, 3]
    end

    # 1D horizontal with BilinearRemapping (linear on 2-point cell)
    horzdomain_1d = Domains.IntervalDomain(
        Geometry.XPoint(-500.0) .. Geometry.XPoint(500.0),
        periodic = true,
    )
    horzmesh_1d = Meshes.IntervalMesh(horzdomain_1d, nelems = 10)
    horztopology_1d = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        horzmesh_1d,
    )
    horzspace_1d = Spaces.SpectralElementSpace1D(horztopology_1d, quad)
    xpts_1d = range(-500.0, 500.0, length = 11)
    hcoords_1d = [Geometry.XPoint(x) for x in xpts_1d]
    remapper_1d = Remapping.Remapper(
        horzspace_1d,
        hcoords_1d;
        buffer_length = 1,
        horizontal_method = Remapping.BilinearRemapping(),
    )
    coords_1d = Fields.coordinate_field(horzspace_1d)
    interp_x_1d = Remapping.interpolate(remapper_1d, coords_1d.x)
    if ClimaComms.iamroot(context)
        @test Array(interp_x_1d) ≈ [x for x in xpts_1d]
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
        context,
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
    interp_x2 = Remapping.interpolate(coords.x, hcoords, zcoords)
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts, z in zpts]
        @test Array(interp_x2) ≈ [x for x in xpts, y in ypts, z in zpts]
    end

    interp_y = Remapping.interpolate(remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts, z in zpts]
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in xpts, y in ypts, z in zpts]
    if ClimaComms.iamroot(context)
        @test Array(interp_z[:, :, 2:(end - 1)]) ≈ expected_z[:, :, 2:(end - 1)]
        @test Array(interp_z[:, :, 1]) ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in xpts, y in ypts]
        @test Array(interp_z[:, :, end]) ≈
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
    dest = ArrayType(zeros(21, 21, 21))
    Remapping.interpolate!(dest, remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 21, 2))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 21, 3))
    Remapping.interpolate!(dest, remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, :, 1]
        @test interp_y ≈ dest[:, :, :, 2]
        @test interp_x ≈ dest[:, :, :, 3]
    end


    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

    coords = Fields.coordinate_field(horiz_space)

    interp_x = Remapping.interpolate(horiz_remapper, coords.x)
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test Array(interp_x) ≈ [x for x in xpts, y in ypts]
    end

    interp_y = Remapping.interpolate(horiz_remapper, coords.y)
    if ClimaComms.iamroot(context)
        @test Array(interp_y) ≈ [y for x in xpts, y in ypts]
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
    dest = ArrayType(zeros(21, 21))
    Remapping.interpolate!(dest, horiz_remapper, coords.x)
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest
    end


    # Two fields
    dest = ArrayType(zeros(21, 21, 2))
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
    end

    # Three fields (more than buffer length)
    dest = ArrayType(zeros(21, 21, 3))
    Remapping.interpolate!(dest, horiz_remapper, [coords.x, coords.y, coords.x])
    if ClimaComms.iamroot(context)
        @test interp_x ≈ dest[:, :, 1]
        @test interp_y ≈ dest[:, :, 2]
        @test interp_x ≈ dest[:, :, 3]
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
    interp_sin_long2 =
        Remapping.interpolate(sind.(coords.long), hcoords, zcoords)
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test Array(interp_sin_long) ≈
              [sind(x) for x in longpts, y in latpts, z in zpts] rtol = 0.01
        @test Array(interp_sin_long2) ≈
              [sind(x) for x in longpts, y in latpts, z in zpts] rtol = 0.01
    end

    interp_sin_lat = Remapping.interpolate(remapper, sind.(coords.lat))
    if ClimaComms.iamroot(context)
        @test Array(interp_sin_lat) ≈
              [sind(y) for x in longpts, y in latpts, z in zpts] rtol = 0.01
    end

    interp_z = Remapping.interpolate(remapper, coords.z)
    expected_z = [z for x in longpts, y in latpts, z in zpts]
    if ClimaComms.iamroot(context)
        @test Array(interp_z[:, :, 2:(end - 1)]) ≈ expected_z[:, :, 2:(end - 1)]
        @test Array(interp_z[:, :, 1]) ≈
              [1000.0 * (0 / 30 + 1 / 30) / 2 for x in longpts, y in latpts]
        @test Array(interp_z[:, :, end]) ≈
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
    dest = ArrayType(zeros(21, 21, 21))
    Remapping.interpolate!(dest, remapper, sind.(coords.long))
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 21, 2))
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
    dest = ArrayType(zeros(21, 21, 21, 3))
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

    # Horizontal space
    horiz_space = Spaces.horizontal_space(hv_center_space)
    horiz_remapper = Remapping.Remapper(horiz_space, hcoords, buffer_length = 2)

    coords = Fields.coordinate_field(horiz_space)

    interp_sin_long = Remapping.interpolate(horiz_remapper, sind.(coords.long))
    # Only root has the final result
    if ClimaComms.iamroot(context)
        @test Array(interp_sin_long) ≈ [sind(x) for x in longpts, y in latpts] rtol = 0.01
    end

    interp_sin_lat = Remapping.interpolate(horiz_remapper, sind.(coords.lat))
    if ClimaComms.iamroot(context)
        @test Array(interp_sin_lat) ≈ [sind(y) for x in longpts, y in latpts] rtol = 0.01
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
    dest = ArrayType(zeros(21, 21))
    Remapping.interpolate!(dest, horiz_remapper, sind.(coords.long))
    if ClimaComms.iamroot(context)
        @test interp_sin_long ≈ dest
    end

    # Two fields
    dest = ArrayType(zeros(21, 21, 2))
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
    dest = ArrayType(zeros(21, 21, 3))
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

if !with_mpi
    @testset "Purely vertical space" begin
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
        cspace = Spaces.CenterFiniteDifferenceSpace(verttopo)
        fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

        zpts = range(0.0, 1000.0, 21)
        zcoords = [Geometry.ZPoint(z) for z in zpts]

        # Center space
        cremapper = Remapping.Remapper(
            cspace;
            target_zcoords = zcoords,
            buffer_length = 2,
        )
        ccoords = Fields.coordinate_field(cspace)
        cinterp_z = Remapping.interpolate(cremapper, ccoords.z)
        cexpected_z = ArrayType(zpts)
        ClimaComms.allowscalar(device) do
            cexpected_z[1] = 1000.0 * (0 / 30 + 1 / 30) / 2
            cexpected_z[end] = 1000.0 * (29 / 30 + 30 / 30) / 2
        end
        @test cinterp_z ≈ cexpected_z

        # Face space
        fremapper = Remapping.Remapper(
            fspace;
            target_zcoords = zcoords,
            buffer_length = 2,
        )
        fcoords = Fields.coordinate_field(fspace)
        finterp_z = Remapping.interpolate(fremapper, fcoords.z)
        fexpected_z = ArrayType(zpts)
        finterp_z ≈ fexpected_z

        # Remapping two fields
        cinterp = Remapping.interpolate(cremapper, [ccoords.z, ccoords.z])
        @test cexpected_z ≈ cinterp[:, 1]
        @test cexpected_z ≈ cinterp[:, 2]

        finterp = Remapping.interpolate(fremapper, [fcoords.z, fcoords.z])
        @test fexpected_z ≈ finterp[:, 1]
        @test fexpected_z ≈ finterp[:, 2]

        # Remapping three fields (more than the buffer length)
        cinterp =
            Remapping.interpolate(cremapper, [ccoords.z, ccoords.z, ccoords.z])
        @test cexpected_z ≈ cinterp[:, 1]
        @test cexpected_z ≈ cinterp[:, 2]
        @test cexpected_z ≈ cinterp[:, 3]

        finterp =
            Remapping.interpolate(fremapper, [fcoords.z, fcoords.z, fcoords.z])
        @test fexpected_z ≈ finterp[:, 1]
        @test fexpected_z ≈ finterp[:, 2]
        @test fexpected_z ≈ finterp[:, 3]

        # Remapping in-place one field
        dest = ArrayType(zeros(21))
        Remapping.interpolate!(dest, cremapper, ccoords.z)
        @test cexpected_z ≈ dest

        Remapping.interpolate!(dest, fremapper, fcoords.z)
        @test fexpected_z ≈ dest

        # Three fields (more than buffer length)
        dest = ArrayType(zeros(21, 3))
        Remapping.interpolate!(
            dest,
            cremapper,
            [ccoords.z, ccoords.z, ccoords.z],
        )
        @test cexpected_z ≈ dest[:, 1]
        @test cexpected_z ≈ dest[:, 2]
        @test cexpected_z ≈ dest[:, 3]

        Remapping.interpolate!(
            dest,
            fremapper,
            [fcoords.z, fcoords.z, fcoords.z],
        )
        @test fexpected_z ≈ dest[:, 1]
        @test fexpected_z ≈ dest[:, 2]
        @test fexpected_z ≈ dest[:, 3]

        # Convenience interpolate (just checking the shape, correctness of the
        # interpolation is checked elsewhere)
        if ClimaComms.iamroot(context)
            @test size(Remapping.interpolate(fcoords.z)) == (30,)
        end
        if ClimaComms.iamroot(context)
            @test size(Remapping.interpolate(ccoords.z)) == (30,)
        end
    end
end

@testset "Default coordinates" begin

    # 3D Sphere space
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

    @test all(
        Remapping.default_target_zcoords(hv_center_space, zresolution = 50) .≈
        Geometry.ZPoint.(range(0.0, 1000; length = 50)),
    )

    @test all(
        Remapping.default_target_zcoords(hv_center_space) .≈
        Geometry.ZPoint.(
            Array(
                Fields.field2array(Fields.coordinate_field(hv_center_space).z),
            )[
                :,
                1,
            ]
        ),
    )

    @test Remapping.default_target_hcoords(hv_center_space) == [
        Geometry.LatLongPoint(x, y) for x in range(-90.0, 90.0, length = 180),
        y in range(-180.0, 180.0, length = 180)
    ]

    # Purely horizontal 2D space sphere
    @test Remapping.default_target_hcoords(horzspace) == [
        Geometry.LatLongPoint(x, y) for x in range(-90.0, 90.0, length = 180),
        y in range(-180.0, 180.0, length = 180)
    ]

    # Purely vertical spaces

    @test all(
        Remapping.default_target_zcoords(vert_center_space) .≈
        Geometry.ZPoint.(
            Array(
                Fields.field2array(
                    Fields.coordinate_field(vert_center_space).z,
                ),
            )[
                :,
                1,
            ]
        ),
    )
    @test all(
        Remapping.default_target_zcoords(
            Spaces.FaceFiniteDifferenceSpace(vert_center_space),
            zresolution = 50,
        ) .≈ Geometry.ZPoint.(range(0.0, 1000; length = 50)),
    )

    # 3D box space

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
        Geometry.YPoint(-300.0) .. Geometry.YPoint(300.0),
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

    @test all(
        Remapping.default_target_hcoords(hv_center_space) .≈ [
            Geometry.XYPoint(x, y) for x in range(-500.0, 500.0, length = 180),
            y in range(-300.0, 300.0, length = 180)
        ],
    )

    # Purely horizontal 2D space box
    @test all(
        Remapping.default_target_hcoords(horzspace) .≈ [
            Geometry.XYPoint(x, y) for x in range(-500.0, 500.0, length = 180),
            y in range(-300.0, 300.0, length = 180)
        ],
    )

    # 2D slice space

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

    @test all(
        Remapping.default_target_hcoords(hv_center_space) .≈
        [Geometry.XPoint(x) for x in range(-500.0, 500.0, length = 180)],
    )

    @test all(
        Remapping.default_target_zcoords(hv_center_space; zresolution = 50) .≈
        Geometry.ZPoint.(range(0.0, 1000; length = 50)),
    )

    # Purely horizontal 1D space
    @test all(
        Remapping.default_target_hcoords(horzspace) .≈
        [Geometry.XPoint(x) for x in range(-500.0, 500.0, length = 180)],
    )
end

@testset "Mask" begin
    @test_throws ErrorException Remapping.Remapper(
        CommonSpaces.ExtrudedCubedSphereSpace(;
            z_elem = 10,
            z_min = 0,
            z_max = 1,
            radius = 10,
            h_elem = 10,
            n_quad_points = 4,
            staggering = CommonSpaces.CellCenter(),
            enable_mask = true,
        ),
    )
end
