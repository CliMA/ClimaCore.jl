# Test-specific definitions (may be overwritten in each test case file)
# TODO: Allow some of these to be enironment variables or command line arguments
upwinding_mode = :none
horizontal_mesh = nothing # must be object of type AbstractMesh
npoly = 0
z_max = 0
z_elem = 0
t_end = 0
dt = 0
dt_save_to_sol = 0 # 0 means don't save to sol
dt_save_to_disk = 0 # 0 means don't save to disk
ode_algorithm = nothing # must be object of type OrdinaryDiffEqAlgorithm
jacobian_flags = (;) # only required by implicit ODE algorithms
max_newton_iters = 10 # only required by ODE algorithms that use Newton's method
show_progress_bar = false
additional_callbacks = () # e.g., printing diagnostic information
additional_solver_kwargs = (;) # e.g., abstol and reltol
test_implicit_solver = false # makes solver extremely slow when set to `true`
additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)
additional_tendency!(Yₜ, Y, p, t) = nothing
center_initial_condition(local_geometry) = (;)
face_initial_condition(local_geometry) = (;)
postprocessing(sol, output_dir) = nothing

################################################################################

import ClimaTimeSteppers as CTS
using ClimaComms
ClimaComms.@import_required_backends
import SciMLBase
const comms_ctx = ClimaComms.context()
is_distributed = comms_ctx isa ClimaComms.MPICommsContext
using ClimaCore: DataLayouts

using Logging

if is_distributed
    const pid, nprocs = ClimaComms.init(comms_ctx)
    logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    @info "Setting up distributed run on $nprocs \
        processor$(nprocs == 1 ? "" : "s") on a $(comms_ctx.device) device"
else
    using TerminalLoggers: TerminalLogger
    prev_logger = global_logger(TerminalLogger())
end
atexit() do
    global_logger(prev_logger)
end

using SciMLBase
using DiffEqCallbacks
using JLD2

const FT = get(ENV, "FLOAT_TYPE", "Float32") == "Float32" ? Float32 : Float64

include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")

if get(ENV, "Z_STRETCH", "false") == "true"
    z_stretch_scale = FT(7e3)
    z_stretch = Meshes.ExponentialStretching(z_stretch_scale)
    z_stretch_string = "stretched"
else
    z_stretch = Meshes.Uniform()
    z_stretch_string = "uniform"
end

if haskey(ENV, "TEST_NAME")
    test_dir, test_file_name = split(ENV["TEST_NAME"], '/')
else
    error("ENV[\"TEST_NAME\"] required (e.g., \"sphere/baroclinic_wave_rhoe\")")
end
include(joinpath(test_dir, "$test_file_name.jl"))

if z_stretch_string == "stretched"
    test_file_name = "$(z_stretch_string)_$(test_file_name)"
end

if haskey(ENV, "RESTART_FILE")
    restart_file_name = ENV["RESTART_FILE"]
    if is_distributed
        restart_file_name =
            split(restart_file_name, ".jld2")[1] * "_pid$pid.jld2"
    end
    restart_data = jldopen(restart_file_name)
    t_start = restart_data["t"]
    Y = restart_data["Y"]
    close(restart_data)
    ᶜlocal_geometry = Fields.local_geometry_field(Y.c)
    ᶠlocal_geometry = Fields.local_geometry_field(Y.f)
else
    t_start = FT(0)
    horizontal_layout_types = Dict()
    horizontal_layout_types["IJFH"] = DataLayouts.IJFH
    horizontal_layout_types["IJHF"] = DataLayouts.IJHF
    horizontal_layout_type =
        horizontal_layout_types[get(ENV, "horizontal_layout_type", "IJFH")]
    h_space = make_horizontal_space(
        horizontal_mesh,
        npoly,
        comms_ctx,
        horizontal_layout_type,
    )
    center_space, face_space =
        make_hybrid_spaces(h_space, z_max, z_elem; z_stretch)
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    Y = Fields.FieldVector(
        c = center_initial_condition(ᶜlocal_geometry),
        f = face_initial_condition(ᶠlocal_geometry),
    )
end
p = get_cache(ᶜlocal_geometry, ᶠlocal_geometry, Y, dt, upwinding_mode)

include("ode_config.jl")

ode_algo =
    ode_configuration(FT; ode_name = string(ode_algorithm), max_newton_iters)

if haskey(ENV, "OUTPUT_DIR")
    output_dir = ENV["OUTPUT_DIR"]
else
    output_dir =
        joinpath(@__DIR__, test_dir, "output", test_file_name, string(FT))
end
mkpath(output_dir)

function make_save_to_disk_func(output_dir, test_file_name, is_distributed)
    function save_to_disk_func(integrator)
        day = floor(Int, integrator.t / (60 * 60 * 24))
        @info "Saving prognostic variables to JLD2 file on day $day"
        suffix = is_distributed ? "_pid$pid.jld2" : ".jld2"
        output_file = joinpath(output_dir, "$(test_file_name)_day$day$suffix")
        jldsave(output_file; t = integrator.t, Y = integrator.u)
        return nothing
    end
    return save_to_disk_func
end

save_to_disk_func =
    make_save_to_disk_func(output_dir, test_file_name, is_distributed)

dss_callback = FunctionCallingCallback(func_start = true) do Y, t, integrator
    p = integrator.p
    Spaces.weighted_dss!(Y.c, p.ghost_buffer.c)
    Spaces.weighted_dss!(Y.f, p.ghost_buffer.f)
end
if dt_save_to_disk == 0
    save_to_disk_callback = nothing
else
    save_to_disk_callback = PeriodicCallback(
        save_to_disk_func,
        dt_save_to_disk;
        initial_affect = true,
    )
end
callback = SciMLBase.CallbackSet(
    dss_callback,
    save_to_disk_callback,
    additional_callbacks...,
)

problem = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(;
        T_imp! = ODEFunction(
            implicit_tendency!;
            jac_kwargs(ode_algo, Y, jacobian_flags)...,
        ),
        T_exp! = remaining_tendency!,
    ),
    Y,
    (t_start, t_end),
    p,
)
integrator = SciMLBase.init(
    problem,
    ode_algo;
    saveat = dt_save_to_sol == 0 ? [] : collect(t_start:dt_save_to_sol:t_end),
    callback = callback,
    dt = dt,
    adaptive = false,
    progress = show_progress_bar,
    progress_steps = 20,
    additional_solver_kwargs...,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

@info "Running `$test_dir/$test_file_name` test case"
@info "on a vertical $z_stretch_string grid"

walltime = @elapsed sol = SciMLBase.solve!(integrator)
any(isnan, sol.u[end]) && error("NaNs found in result.")

if is_distributed # replace sol.u on the root processor with the global sol.u
    global_Y_c_1 =
        DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[1].c))
    global_Y_f_1 =
        DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[1].f))
    if ClimaComms.iamroot(comms_ctx)
        global_h_space = make_horizontal_space(
            horizontal_mesh,
            npoly,
            ClimaComms.SingletonCommsContext(),
        )
        global_center_space, global_face_space =
            make_hybrid_spaces(global_h_space, z_max, z_elem; z_stretch)
        global_Y_c_type =
            Fields.Field{typeof(global_Y_c_1), typeof(global_center_space)}
        global_Y_f_type =
            Fields.Field{typeof(global_Y_f_1), typeof(global_face_space)}
        global_Y_type = Fields.FieldVector{
            FT,
            NamedTuple{(:c, :f), Tuple{global_Y_c_type, global_Y_f_type}},
        }
        global_sol_u = similar(sol.u, global_Y_type)
    end
    for i in 1:length(sol.u)
        global_Y_c =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].c))
        global_Y_f =
            DataLayouts.gather(comms_ctx, Fields.field_values(sol.u[i].f))
        if ClimaComms.iamroot(comms_ctx)
            global_sol_u[i] = Fields.FieldVector(
                c = Fields.Field(global_Y_c, global_center_space),
                f = Fields.Field(global_Y_f, global_face_space),
            )
        end
    end
    if ClimaComms.iamroot(comms_ctx)
        sol = SciMLBase.sensitivity_solution(sol, global_sol_u, sol.t)
        output_file =
            joinpath(output_dir, "scaling_data_$(nprocs)_processes.jld2")
        println("writing performance data to $output_file")
        jldsave(output_file; nprocs, walltime)
    end
end
if !is_distributed || ClimaComms.iamroot(comms_ctx)
    println("Walltime = $walltime seconds")
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    # https://github.com/CliMA/ClimaCore.jl/issues/2058
    if !(Fields.field_values(sol.u[1].c) isa DataLayouts.VIJHF)
        postprocessing(sol, output_dir)
    end
end
