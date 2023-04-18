#####
##### Utilities for the ClimaCore spectral element CuArray-backed operator benchmark
#####

using LinearAlgebra: ×
import PrettyTables
import LinearAlgebra as LA
import OrderedCollections
import ClimaCore.Operators as Operators
import ClimaCore.Domains as Domains
import ClimaCore.Meshes as Meshes
import ClimaCore.Spaces as Spaces
import ClimaCore.Topologies as Topologies
import ClimaCore.Geometry as Geometry
import ClimaCore as CC
import ClimaComms
import ClimaCommsMPI

using CUDA
using ArgParse
using StatsBase
using Test
using BenchmarkTools

function benchmark_kernel_array!(
    args,
    kernel_fun!,
    device::ClimaComms.CPU;
    silent = true,
)
    (; ϕ_arr, ψ_arr) = args
    kernel_fun!(args) # compile first
    trial = BenchmarkTools.@benchmark $kernel_fun!($args)
    if !silent
        show(stdout, MIME("text/plain"), trial)
        println()
    end
    return trial
end

function benchmark_kernel_array!(
    args,
    kernel_fun!,
    device::ClimaComms.CUDA;
    silent = true,
)
    # Taken from: https://cuda.juliagpu.org/stable/tutorials/introduction/
    (; ϕ_arr, ψ_arr) = args
    N = length(ϕ_arr)
    fill!(ϕ_arr, 1)
    fill!(ψ_arr, 2)
    kernel = @cuda launch = false kernel_fun!(args)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    kernel(args; threads, blocks)
    @test all(Array(ϕ_arr) .== Array(ψ_arr)) # compile and confirm correctness

    # Perform benchmark
    trial = BenchmarkTools.@benchmark CUDA.@sync $kernel(
        $args,
        threads = $threads,
        blocks = $blocks,
    )
    if !silent
        show(stdout, MIME("text/plain"), trial)
        println()
    end
    return trial
end

function benchmark_kernel!(args, kernel_fun!, ::ClimaComms.CPU; silent)
    kernel_fun!(args) # compile first
    trial = BenchmarkTools.@benchmark $kernel_fun!($args)
    if !silent
        show(stdout, MIME("text/plain"), trial)
        println()
    end
    return trial
end

function benchmark_kernel!(args, kernel_fun!, ::ClimaComms.CUDA; silent)
    kernel_fun!(args) # compile first
    trial = BenchmarkTools.@benchmark CUDA.@sync $kernel_fun!($args)
    if !silent
        show(stdout, MIME("text/plain"), trial)
        println()
    end
    return trial
end

function initial_velocity(space)
    uλ, uϕ = zeros(space), zeros(space)
    return @. Geometry.Covariant12Vector(Geometry.UVVector(uλ, uϕ))
end

function ismpi()
    # detect common environment variables used by MPI launchers
    #   PMI_RANK appears to be used by MPICH and srun
    #   OMPI_COMM_WORLD_RANK appears to be used by OpenMPI
    return haskey(ENV, "PMI_RANK") || haskey(ENV, "OMPI_COMM_WORLD_RANK")
end

function create_space(
    context;
    float_type = Float64,
    panel_size = 9,
    poly_nodes = 4,
)
    earth_radius = float_type(6.37122e6)
    domain = Domains.SphereDomain(earth_radius)
    mesh = Meshes.EquiangularCubedSphere(domain, panel_size)
    quad = Spaces.Quadratures.GLL{poly_nodes}()
    topology = Topologies.Topology2D(context, mesh)
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

function setup_kernel_args(ARGS::Vector{String} = ARGS)
    s = ArgParseSettings(prog = "spectralelement operator benchmarks")
    @add_arg_table! s begin
        "--device"
        help = "Computation device (CPU, CUDA)"
        arg_type = String
        default = CUDA.functional() ? "CUDA" : "CPU"
        "--comms"
        help = "Communication type (Singleton, MPI)"
        arg_type = String
        default = ismpi() ? "MPI" : "Singleton"
        "--float-type"
        help = "Floating point type (Float32, Float64)"
        eval_arg = true
        default = Float64
        "--panel-size"
        help = "Number of elements across each panel"
        arg_type = Int
        default = 8
        "--poly-nodes"
        help = "Number of nodes in each dimension to use in the polynomial approximation. Polynomial degree = poly-nodes - 1."
        arg_type = Int
        default = 4
    end
    args = parse_args(ARGS, s)

    device =
        args["device"] == "CUDA" ? ClimaComms.CUDA() :
        args["device"] == "CPU" ? ClimaComms.CPU() :
        error("Unknown device: $(args["device"])")

    context =
        args["comms"] == "MPI" ? ClimaCommsMPI.MPICommsContext(device) :
        args["comms"] == "Singleton" ?
        ClimaComms.SingletonCommsContext(device) :
        error("Unknown comms: $(args["comms"])")

    ClimaComms.init(context)

    if context isa ClimaCommsMPI.MPICommsContext && device isa ClimaComms.CUDA
        # assign GPUs based on local rank
        local_comm = MPI.Comm_split_type(
            context.mpicomm,
            MPI.COMM_TYPE_SHARED,
            MPI.Comm_rank(context.mpicomm),
        )
        CUDA.device!(MPI.Comm_rank(local_comm) % length(CUDA.devices()))
    end
    float_type = args["float-type"]
    panel_size = args["panel-size"]
    poly_nodes = args["poly-nodes"]
    space = create_space(context; float_type, panel_size, poly_nodes)
    if ClimaComms.iamroot(context)
        nprocs = ClimaComms.nprocs(context)
        @info "Setting up benchmark" device context float_type panel_size poly_nodes
    end

    # Fields
    FT = float_type
    ϕ = zeros(space)
    ψ = zeros(space)
    u = initial_velocity(space)
    du = initial_velocity(space)
    ϕ_buffer = Spaces.create_dss_buffer(ϕ)
    u_buffer = Spaces.create_dss_buffer(u)
    f = @. Geometry.Contravariant3Vector(Geometry.WVector(ϕ))

    s = size(parent(ϕ))
    array_kernel_args = if device isa ClimaComms.CPU
        (; ϕ_arr = fill(FT(1), s), ψ_arr = fill(FT(2), s))
    else
        device isa ClimaComms.CUDA
        (; ϕ_arr = CUDA.fill(FT(1), s), ψ_arr = CUDA.fill(FT(2), s))
    end

    kernel_args = (; ϕ, ψ, u, du, f)
    buffers = (; u_buffer, ϕ_buffer) # cannot reside in CuArray kernels

    arr_args = (; array_kernel_args..., kernel_args..., device)
    return (; arr_args..., buffers, arr_args, float_type)
end

get_summary(trial, trial_arr) = (;
    # Using some BenchmarkTools internals :/
    t_mean_float = StatsBase.mean(trial.times),
    t_mean = BenchmarkTools.prettytime(StatsBase.mean(trial.times)),
    t_mean_arr = BenchmarkTools.prettytime(StatsBase.mean(trial_arr.times)),
    n_samples = length(trial),
)

function tabulate_summary(summary)
    summary_keys = collect(keys(summary))
    t_mean = map(k -> summary[k].t_mean, summary_keys)
    t_mean_arr = map(k -> summary[k].t_mean_arr, summary_keys)
    n_samples = map(k -> summary[k].n_samples, summary_keys)

    # TODO: add a speedup column

    table_data =
        hcat(string.(collect(keys(summary))), t_mean, t_mean_arr, n_samples)

    header = (
        ["Operator", "Mean time", "Mean time", "N-samples"],
        ["", "ClimaCore", "Array", ""],
    )

    PrettyTables.pretty_table(
        table_data;
        header,
        crop = :none,
        alignment = vcat(:l, repeat([:r], length(header[1]) - 1)),
    )
end

function test_against_best_times(bm, best_times)
    buffer = 1.3
    pass(k) = bm[k].t_mean_float < best_times[k] * buffer
    intersect_keys = intersect(collect(keys(bm)), collect(keys(best_times)))
    if !all(k -> pass(k), intersect_keys)
        for k in keys(bm)
            pass(k) ||
                @error "$k failed: $(bm[k].t_mean_float) < $(best_times[k] * buffer)."
        end
        error("Spectral element CUDA operator benchmarks failed")
    else
        @info "Spectral element CUDA operator benchmarks passed 🎉"
    end
    setdiff_keys = setdiff(keys(bm), keys(best_times))
    if !(length(intersect_keys) == length(best_times) == length(bm))
        @show collect(keys(bm))
        @show collect(keys(best_times))
        error("Benchmark and best times keys must match.")
    end
end
