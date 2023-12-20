import ClimaCore
import ClimaCoreTempestRemap as CCTR
using ClimaComms
using ClimaCore: Geometry, Meshes, Domains, Topologies, Spaces, Fields, Quadratures
using Test
# use these packages for manual inspection of solutions
# using Plots
# using ClimaCorePlots

@testset "distributed online remapping" begin
    # general setup
    comms_ctx = ClimaComms.MPICommsContext()
    pid, nprocs = ClimaComms.init(comms_ctx)
    OUTPUT_DIR = ClimaComms.bcast(
        comms_ctx,
        mkpath(get(ENV, "CI_OUTPUT_DIR", tempname())),
    )

    comms_ctx_singleton = ClimaComms.SingletonCommsContext()

    # domain
    radius = 1.0 # unit sphere
    domain = Domains.SphereDomain(radius)

    # construct source spaces
    ne_i = 20
    mesh_i = Meshes.EquiangularCubedSphere(domain, ne_i)
    topology_i_distr = Topologies.Topology2D(comms_ctx, mesh_i)

    nq_i = 3 # polynomial order for SE discretization
    quad_i = Quadratures.GLL{nq_i}()
    space_i_distr = Spaces.SpectralElementSpace2D(topology_i_distr, quad_i)

    topology_i_singleton = Topologies.Topology2D(comms_ctx_singleton, mesh_i)
    space_i_singleton =
        Spaces.SpectralElementSpace2D(topology_i_singleton, quad_i)

    # construct target spaces
    ne_o = 5
    mesh_o = Meshes.EquiangularCubedSphere(domain, ne_o)
    topology_o_distr = Topologies.Topology2D(comms_ctx, mesh_o)

    nq_o = 4
    quad_o = Quadratures.GLL{nq_o}()
    space_o_distr = Spaces.SpectralElementSpace2D(topology_o_distr, quad_o)

    topology_o_singleton = Topologies.Topology2D(comms_ctx_singleton, mesh_o)
    space_o_singleton =
        Spaces.SpectralElementSpace2D(topology_o_singleton, quad_o)

    # generate test data in the Field format (on non-distributed topo. for non-halo exchange version)
    field_i_singleton = sind.(Fields.coordinate_field(space_i_singleton).long)

    # global exchange (no buffer fill/send here) - convert to superhalo in next implementation
    root_pid = 0
    ClimaComms.gather(comms_ctx, parent(field_i_singleton))
    field_i_singleton = ClimaComms.bcast(comms_ctx, field_i_singleton)

    R_distr = CCTR.generate_map(
        space_o_singleton,
        space_i_singleton,
        target_space_distr = space_o_distr,
    )

    if ClimaComms.iamroot(comms_ctx)
        # remap without MPI (for testing comparison) and plot solution
        R_singleton = CCTR.generate_map(space_o_singleton, space_i_singleton)
        field_o_singleton = Fields.zeros(space_o_singleton)
        CCTR.remap!(field_o_singleton, R_singleton, field_i_singleton)
    end

    # setup and apply the remap using distributed approach
    field_o_distr = Fields.zeros(space_o_distr)

    # apply the remapping to field_i_singleton and store the result in field_o_distr
    CCTR.remap!(field_o_distr, R_distr, field_i_singleton)

    # compute analytical solution for comparison
    field_ref = sind.(Fields.coordinate_field(space_o_distr).long)

    # write distributed fields to hdf5 files
    field_ref_file = OUTPUT_DIR * "/field_ref.hdf5"
    ClimaComms.barrier(comms_ctx)
    writer = ClimaCore.InputOutput.HDF5Writer(field_ref_file, comms_ctx)
    ClimaCore.InputOutput.write!(writer, field_ref, "field_ref")
    ClimaComms.barrier(comms_ctx)
    close(writer)

    field_o_distr_file = OUTPUT_DIR * "/field_o_distr.hdf5"
    ClimaComms.barrier(comms_ctx)
    writer = ClimaCore.InputOutput.HDF5Writer(field_o_distr_file, comms_ctx)
    ClimaCore.InputOutput.write!(writer, field_o_distr, "field_o_distr")
    ClimaComms.barrier(comms_ctx)
    close(writer)

    ClimaComms.barrier(comms_ctx)

    # plot input data and remapped data for comparison
    if ClimaComms.iamroot(comms_ctx)
        # read distributed fields from hdf5 files
        reader = ClimaCore.InputOutput.HDF5Reader(
            field_ref_file,
            comms_ctx_singleton,
        )
        restart_field_ref =
            ClimaCore.InputOutput.read_field(reader, "field_ref")
        close(reader)

        reader = ClimaCore.InputOutput.HDF5Reader(
            field_o_distr_file,
            comms_ctx_singleton,
        )
        restart_field_o_distr =
            ClimaCore.InputOutput.read_field(reader, "field_o_distr")
        close(reader)

        # plot source data, serial, analytical, and distributed solutions - for manual inspection
        # field_i_fig = plot(field_i_singleton, title = "source data")
        # field_ref_fig =
        #     plot(restart_field_ref, title = "target data (analytical)")
        # field_o_fig = plot(field_o_singleton, title = "target data (non-MPI)")
        # savefig(field_o_fig, OUTPUT_DIR * "/target_data_serial")
        # field_o_distr_fig =
        #     plot(restart_field_o_distr, title = "target data (MPI)")
        # savefig(field_i_fig, OUTPUT_DIR * "/source_data")
        # savefig(field_ref_fig, OUTPUT_DIR * "/target_data_soln")
        # savefig(field_o_distr_fig, OUTPUT_DIR * "/target_data_mpi")

        # compare distributed and serial solutions
        @test parent(restart_field_o_distr) ≈ parent(field_o_singleton) atol =
            1e-20
    end
end
