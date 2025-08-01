using Test
import ClimaCore
using ClimaCore:
    Geometry,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    DataLayouts,
    Hypsography,
    InputOutput,
    Grids

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())
pid, nprocs = ClimaComms.init(comms_ctx)
filename = ClimaComms.bcast(comms_ctx, tempname(pwd()))
if ClimaComms.iamroot(comms_ctx)
    @info "Comms context" comms_ctx nprocs filename
end

@testset "HDF5 restart test for 3d hybrid deep cubed sphere with topography and deep" begin
    for deep in (false, true)
        FT = Float32
        R = FT(6.371229e6)

        npoly = 4
        z_max = FT(30e3)
        z_elem = 10
        h_elem = 4
        device = ClimaComms.device(comms_ctx)
        # horizontal space
        domain = Domains.SphereDomain(R)
        horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
        topology = Topologies.Topology2D(
            comms_ctx,
            horizontal_mesh,
            Topologies.spacefillingcurve(horizontal_mesh),
        )
        quad = Quadratures.GLL{npoly + 1}()
        h_space = Spaces.SpectralElementSpace2D(topology, quad)
        # vertical space
        z_domain = Domains.IntervalDomain(
            Geometry.ZPoint(zero(z_max)),
            Geometry.ZPoint(z_max);
            boundary_names = (:bottom, :top),
        )


        z_surface =
            Geometry.ZPoint.(
                z_max / 8 .* (
                    cosd.(Fields.coordinate_field(h_space).lat) .+
                    cosd.(Fields.coordinate_field(h_space).long) .+ 1
                ),
            )

        z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
        z_topology = Topologies.IntervalTopology(device, z_mesh)
        z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
        # Extruded 3D space
        h_grid = Spaces.grid(h_space)
        z_grid = Spaces.grid(z_space)
        hypsography = Hypsography.LinearAdaption(z_surface)
        grid = Grids.ExtrudedFiniteDifferenceGrid(
            h_grid,
            z_grid,
            hypsography;
            deep,
        )

        center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
        face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)

        ᶜlocal_geometry = Fields.local_geometry_field(center_space)
        ᶠlocal_geometry = Fields.local_geometry_field(face_space)

        Y = Fields.FieldVector(; c = ᶜlocal_geometry, f = ᶠlocal_geometry)

        # write field vector to hdf5 file
        InputOutput.HDF5Writer(filename, comms_ctx) do writer
            InputOutput.write!(writer, Y, "Y")
        end

        InputOutput.HDF5Reader(filename, comms_ctx) do reader
            restart_Y = InputOutput.read_field(reader, "Y") # read fieldvector from hdf5 file
            @test restart_Y == Y # test if restart is exact
        end
    end
end


@testset "HDF5 restart test for a Named Tuple of Levels of a 3D hybrid cubed sphere for deep" begin
    # This I/O is used for the computation of the topographic drag
    FT = Float32

    for space in (
        TU.CenterExtrudedFiniteDifferenceSpace(FT, context = comms_ctx),
        TU.FaceExtrudedFiniteDifferenceSpace(FT, context = comms_ctx),
    )
        TU.levelable(space) || continue
        field = fill((; x = FT(1)), space)

        level_of_field = Fields.Field(
            Spaces.level(Fields.field_values(field.x), 1),
            Spaces.level(space, TU.fc_index(1, space)),
        )

        fake_drag = fill(
            (;
                t11 = FT(0.0),
                t12 = FT(0.0),
                t21 = FT(0.0),
                t22 = FT(0.0),
                hmin = FT(0.0),
                hmax = FT(0.0),
            ),
            axes(level_of_field),
        )

        # write field vector to hdf5 file
        InputOutput.HDF5Writer(filename, comms_ctx) do writer
            InputOutput.write!(writer, fake_drag, "fake_drag")
        end

        InputOutput.HDF5Reader(filename, comms_ctx) do reader
            restart_fake_drag = InputOutput.read_field(reader, "fake_drag") # read fieldvector from hdf5 file

            # The underlying space is of a different instance, so we cannot use == to check for equivalence.
            # Instead, we make sure that the values and types are the same.
            @test typeof(restart_fake_drag) == typeof(fake_drag)
            @test maximum(
                abs.(parent(fake_drag.t21) .- parent(restart_fake_drag.t21)),
            ) == 0.0f0
        end
    end
end
