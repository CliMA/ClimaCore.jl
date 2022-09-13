ENV["CLIMACORE_DISTRIBUTED"] = "MPI"
usempi = true
npoly = parse(Int, get(ARGS, 1, 4))
FT = get(ARGS, 2, Float64) == "Float64" ? Float64 : Float32
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
if usempi
    using ClimaComms
    using ClimaCommsMPI
end

set_initial_condition(space) =
    map(Fields.local_geometry_field(space)) do local_geometry
        coord = local_geometry.coordinates
        h = 1.0
        u = Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(1.0, 1.0),
            local_geometry,
        )
        return (h = h, u = u)
    end

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

function shallow_water_dss_profiler(usempi::Bool, ::Type{FT}, npoly) where {FT}
    context = ClimaCommsMPI.MPICommsContext()
    pid, nprocs = ClimaComms.init(context)
    iamroot = ClimaComms.iamroot(context)
    # log output only from root process
    logger_stream = iamroot ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end

    if iamroot
        println("running distributed DSS using $nprocs processes")
    end
    # Set up discretization
    ne = 9 # the rossby_haurwitz test case's initial state has a singularity at the pole. We avoid it by using odd number of elements
    Nq = npoly + 1
    R = FT(6.37122e6) # radius of earth
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Spaces.Quadratures.GLL{Nq}()
    grid_topology = Topologies.DistributedTopology2D(
        context,
        mesh,
        Topologies.spacefillingcurve(mesh),
    )
    global_grid_topology = Topologies.Topology2D(mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    global_space = Spaces.SpectralElementSpace2D(global_grid_topology, quad)
    Y = set_initial_condition(space)
    ghost_buffer = usempi ? Spaces.create_ghost_buffer(Y) : nothing
    nsamples = 10000

    # precompile relevant functions
    Spaces.weighted_dss_internal!(Y, ghost_buffer)
    weighted_dss_full!(Y, ghost_buffer)
    Spaces.fill_send_buffer!(
        grid_topology,
        Fields.field_values(Y),
        ghost_buffer,
    )
    dss_comms!(grid_topology, Y, ghost_buffer)
    ClimaComms.barrier(context)

    # timing
    walltime_dss_full = @elapsed begin # timing weighted dss
        for i in 1:nsamples
            weighted_dss_full!(Y, ghost_buffer)
        end
    end
    ClimaComms.barrier(context)
    walltime_dss_full /= FT(nsamples)


    ClimaComms.barrier(context)
    walltime_dss_internal = @elapsed begin # timing internal dss
        for i in 1:nsamples
            Spaces.weighted_dss_internal!(Y, ghost_buffer)
        end
    end
    ClimaComms.barrier(context)
    walltime_dss_internal /= FT(nsamples)

    ClimaComms.barrier(context)
    walltime_dss_comms = @elapsed begin # timing dss_comms
        for i in 1:nsamples
            dss_comms!(grid_topology, Y, ghost_buffer)
        end
    end
    ClimaComms.barrier(context)
    walltime_dss_comms /= FT(nsamples)

    ClimaComms.barrier(context)
    walltime_dss_comms_fsb = @elapsed begin # timing dss_fill_send_buffer
        for i in 1:nsamples
            Spaces.fill_send_buffer!(
                grid_topology,
                Fields.field_values(Y),
                ghost_buffer,
            )
        end
    end
    ClimaComms.barrier(context)
    walltime_dss_comms_fsb /= FT(nsamples)

    ClimaComms.barrier(context)
    walltime_dss_comms_other = @elapsed begin # timing dss_fill_send_buffer
        for i in 1:nsamples
            ClimaComms.start(ghost_buffer.graph_context)
            ClimaComms.finish(ghost_buffer.graph_context)
        end
    end
    ClimaComms.barrier(context)
    walltime_dss_comms_other /= FT(nsamples)


    # profiling
    ClimaComms.barrier(context)
    for i in 1:nsamples # profiling weighted dss
        @nvtx "dss-loop" color = colorant"green" begin
            @nvtx "start" color = colorant"brown" begin
                Spaces.weighted_dss_start!(Y, ghost_buffer)
            end
            @nvtx "internal" color = colorant"blue" begin
                Spaces.weighted_dss_internal!(Y, ghost_buffer)
            end
            @nvtx "ghost" color = colorant"yellow" begin
                Spaces.weighted_dss_ghost!(Y, ghost_buffer)
            end
        end
    end
    ClimaComms.barrier(context)

    for i in 1:nsamples # profiling dss_comms
        @nvtx "dss-comms-loop" color = colorant"green" begin
            dss_comms!(grid_topology, Y, ghost_buffer)
        end
    end
    ClimaComms.barrier(context)


    if iamroot
        println("# of samples = $nsamples")
        println("walltime_dss_full per sample = $walltime_dss_full (sec)")
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
        output_dir = joinpath(Base.@__DIR__, "output")
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
            walltime_dss_internal,
            walltime_dss_comms,
            walltime_dss_comms_fsb,
            walltime_dss_comms_other,
        )
    end
    return nothing
end

shallow_water_dss_profiler(usempi, FT, npoly)
