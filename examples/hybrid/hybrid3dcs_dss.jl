ENV["TEST_NAME"] = "sphere/baroclinic_wave_rhoe"
using LinearAlgebra
using Colors
using JLD2

include("../nvtx.jl")

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

using Logging
using ClimaComms
resolution = get(ARGS, 1, "low")
npoly = parse(Int, get(ARGS, 2, 4))
FT = get(ARGS, 3, Float64) == "Float64" ? Float64 : Float32

include("../common_spaces.jl")
include("../hybrid/sphere/baroclinic_wave_utilities.jl")


center_initial_condition(local_geometry) =
    center_initial_condition(local_geometry, Val(:ρe))

function dss_comms!(topology, Y, ghost_buffer)
    Spaces.fill_send_buffer!(topology, Fields.field_values(Y), ghost_buffer)
    ClimaComms.start(ghost_buffer.graph_context)
    ClimaComms.finish(ghost_buffer.graph_context)
    return nothing
end

function weighted_dss_full!(Y, ghost_buffer)
    Spaces.weighted_dss_start!(Y, ghost_buffer)
    Spaces.weighted_dss_internal!(Y, ghost_buffer)
    Spaces.weighted_dss_ghost!(Y, ghost_buffer)
    return nothing
end


function hybrid3dcubedsphere_dss_profiler(
    ::Type{FT},
    resolution,
    npoly,
) where {FT}
    comms_ctx = ClimaComms.MPICommsContext()
    pid, nprocs = ClimaComms.init(comms_ctx)
    iamroot = ClimaComms.iamroot(comms_ctx)
    # log output only from root process
    logger_stream = iamroot ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end

    if iamroot
        println("running distributed DSS using $nprocs processes")
    end
    GC.gc(false)
    R = FT(6.371229e6)
    z_max = FT(30e3)
    z_elem, h_elem = resolution == "low" ? (10, 4) : (45, 24)
    z_stretch = Meshes.Uniform()
    z_stretch_string = "uniform"
    horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = h_elem)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    h_topology = Topologies.Topology2D(
        comms_ctx,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    h_space = Spaces.SpectralElementSpace2D(h_topology, quad)

    center_space, face_space =
        make_hybrid_spaces(h_space, z_max, z_elem; z_stretch)
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    Y = Fields.FieldVector(
        c = center_initial_condition(ᶜlocal_geometry),
        f = face_initial_condition(ᶠlocal_geometry),
    )
    ghost_buffer = (
        c = Spaces.create_ghost_buffer(Y.c),
        f = Spaces.create_ghost_buffer(Y.f),
    )
    dss_buffer_f = Spaces.create_dss_buffer(Y.f)
    dss_buffer_c = Spaces.create_dss_buffer(Y.c)
    nsamples = 10000
    nsamplesprofiling = 100

    # precompile relevant functions
    space = axes(Y.c)
    horizontal_topology = space.horizontal_space.topology
    Spaces.weighted_dss_internal!(Y.c, ghost_buffer.c)
    weighted_dss_full!(Y.c, ghost_buffer.c)
    Spaces.fill_send_buffer!(
        horizontal_topology,
        Fields.field_values(Y.c),
        ghost_buffer.c,
    )
    dss_comms!(horizontal_topology, Y.c, ghost_buffer.c)
    Spaces.weighted_dss!(Y.c, dss_buffer_c)
    ClimaComms.barrier(comms_ctx)

    # timing
    walltime_dss_full = @elapsed begin # timing weighted dss
        for i in 1:nsamples
            weighted_dss_full!(Y.c, ghost_buffer.c)
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss_full /= FT(nsamples)

    walltime_dss2_full = @elapsed begin # timing weighted dss2
        for i in 1:nsamples
            Spaces.weighted_dss!(Y.c, dss_buffer_c)
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss2_full /= FT(nsamples)

    ClimaComms.barrier(comms_ctx)
    walltime_dss_internal = @elapsed begin # timing internal dss
        for i in 1:nsamples
            Spaces.weighted_dss_internal!(Y.c, ghost_buffer.c)
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss_internal /= FT(nsamples)

    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms = @elapsed begin # timing dss_comms
        for i in 1:nsamples
            dss_comms!(horizontal_topology, Y.c, ghost_buffer.c)
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms /= FT(nsamples)

    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms_fsb = @elapsed begin # timing dss_fill_send_buffer
        for i in 1:nsamples
            Spaces.fill_send_buffer!(
                horizontal_topology,
                Fields.field_values(Y.c),
                ghost_buffer.c,
            )
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms_fsb /= FT(nsamples)

    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms_other = @elapsed begin # timing dss_fill_send_buffer
        for i in 1:nsamples
            ClimaComms.start(ghost_buffer.c.graph_context)
            ClimaComms.finish(ghost_buffer.c.graph_context)
        end
    end
    ClimaComms.barrier(comms_ctx)
    walltime_dss_comms_other /= FT(nsamples)

    # profiling
    ClimaComms.barrier(comms_ctx)

    for i in 1:nsamplesprofiling # profiling weighted dss
        @nvtx "dss-loop" color = colorant"green" begin
            @nvtx "start" color = colorant"brown" begin
                Spaces.weighted_dss_start!(Y.c, ghost_buffer.c)
            end
            @nvtx "internal" color = colorant"blue" begin
                Spaces.weighted_dss_internal!(Y.c, ghost_buffer.c)
            end
            @nvtx "ghost" color = colorant"yellow" begin
                Spaces.weighted_dss_ghost!(Y.c, ghost_buffer.c)
            end
        end
    end
    ClimaComms.barrier(comms_ctx)

    for i in 1:nsamplesprofiling # profiling dss_comms
        @nvtx "dss-comms-loop" color = colorant"green" begin
            dss_comms!(horizontal_topology, Y.c, ghost_buffer.c)
        end
    end
    ClimaComms.barrier(comms_ctx)

    if iamroot
        println("# of samples = $nsamples")
        println("walltime_dss_full per sample = $walltime_dss_full (sec)")
        println("walltime_dss2_full per sample = $walltime_dss2_full (sec)")
        println(
            "walltime_dss_internal per sample = $walltime_dss_internal (sec)",
        )
        println("walltime_dss_comms per sample = $walltime_dss_comms (sec)")
        println(
            "walltime_dss_comms_fsb per sample = $walltime_dss_comms_fsb (sec)",
        )
        println(
            "walltime_dss_comms_other per sample = $walltime_dss_comms_other (sec)",
        )
        output_dir = joinpath(Base.@__DIR__, "dss_output_$(resolution)_res")
        mkpath(output_dir)
        dss_scaling_file = joinpath(
            output_dir,
            "dss_scaling_data_$(nprocs)_processes_npoly_$npoly.jld2",
        )
        println("writing to dss_scaling file: $dss_scaling_file")
        JLD2.jldsave(
            dss_scaling_file;
            nprocs,
            nsamples,
            walltime_dss_full,
            walltime_dss2_full,
            walltime_dss_internal,
            walltime_dss_comms,
            walltime_dss_comms_fsb,
            walltime_dss_comms_other,
        )
    end

    return nothing
end

hybrid3dcubedsphere_dss_profiler(FT, resolution, npoly)
