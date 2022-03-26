usempi = get(ENV, "CLIMACORE_DISTRIBUTED", "") == "MPI"

using Logging
if usempi
    using ClimaComms
    using ClimaCommsMPI
    const Context = ClimaCommsMPI.MPICommsContext
    const pid, nprocs = ClimaComms.init(Context)
    if pid == 1
        println("parallel run with $nprocs processes")
    end
    logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull

    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end
else
    using Logging: global_logger
    using TerminalLoggers: TerminalLogger
    global_logger(TerminalLogger())
end

import ClimaCore: enable_threading
enable_threading() = false

using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2

default_test_name = "sphere/baroclinic_wave_rhoe"

test_implicit_solver = false # makes solver extremely slow when set to `true`

# Definitions that are specific to each test:
space = nothing
t_end = 0
dt = 0
dt_save_to_sol = 0 # 0 means don't save to sol
dt_save_to_disk = 0 # 0 means don't save to disk
ode_algorithm = OrdinaryDiffEq.SSPRK33
jacobian_flags = (;) # only required by implicit ODE algorithms
max_newton_iters = 10 # only required by ODE algorithms that use Newton's method
show_progress_bar = false
additional_callbacks = () # e.g., printing diagnostic information
additional_solver_kwargs = (;) # e.g., abstol and reltol
center_initial_condition(local_geometry) = (;)
face_initial_condition(local_geometry) = (;)
postprocessing(sol, p, output_dir) = nothing
################################################################################

const FT = get(ENV, "FLOAT_TYPE", "Float64") == "Float32" ? Float32 : Float64

if haskey(ENV, "TEST_NAME")
    test_name = ENV["TEST_NAME"]
else
    test_name = default_test_name
end
test_dir, test_file_name = split(test_name, '/')
include(joinpath(test_dir, "$test_file_name.jl"))

if haskey(ENV, "RESTART_FILE")
    restart_data = jldopen(ENV["RESTART_FILE"])
    t_start = restart_data["t"]
    Y = restart_data["Y"]
    close(restart_data)
    ᶜlocal_geometry = Fields.local_geometry_field(Y.c)
    ᶠlocal_geometry = Fields.local_geometry_field(Y.f)
else
    t_start = FT(0)
    ᶜlocal_geometry, ᶠlocal_geometry =
        Fields.local_geometry_field(hv_center_space),
        Fields.local_geometry_field(hv_face_space)

    Y = Fields.FieldVector(
        c = map(center_initial_condition, ᶜlocal_geometry),
        f = map(face_initial_condition, ᶠlocal_geometry),
    )
end
p = get_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt)

if ode_algorithm <: Union{
    OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
    OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
}
    use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
    W = SchurComplementW(Y, use_transform, jacobian_flags, test_implicit_solver)
    jac_kwargs =
        use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
        (; jac_prototype = W, Wfact = Wfact!)

    alg_kwargs = (; linsolve = linsolve!)
    if ode_algorithm <: Union{
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }
        alg_kwargs =
            (; alg_kwargs..., nlsolve = NLNewton(; max_iter = max_newton_iters))
    end
else
    jac_kwargs = alg_kwargs = (;)
end

if haskey(ENV, "OUTPUT_DIR")
    output_dir = ENV["OUTPUT_DIR"]
else
    output_dir = joinpath(@__DIR__, test_dir, "output", test_file_name)
end
mkpath(output_dir)

function make_save_to_disk(output_dir, test_file_name)
    function save_to_disk(integrator)
        day = floor(Int, integrator.t / (60 * 60 * 24))
        @info "Saving prognostic variables to JLD2 file on day $day"
        output_file = joinpath(output_dir, "$(test_file_name)_day$day.jld2")
        #output_file = joinpath(output_dir, "$(test_file_name)_day$day."*string(integrator.t - day*3600*24)*".jld2")
        jldsave(output_file; t = integrator.t, Y = integrator.u)
        return nothing
    end
    return save_to_disk
end
if dt_save_to_disk == 0
    saving_callback = nothing
else
    saving_callback = PeriodicCallback(
        make_save_to_disk(output_dir, test_file_name),
        dt_save_to_disk;
        initial_affect = true,
    )
end
callback = CallbackSet(saving_callback, additional_callbacks...)
problem = SplitODEProblem(
    ODEFunction(
        implicit_tendency!;
        jac_kwargs...,
        tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= FT(0)),
    ),
    remaining_tendency!,
    Y,
    (t_start, t_end),
    p,
)
integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm(; alg_kwargs...);
    saveat = dt_save_to_sol == 0 ? [] : dt_save_to_sol,
    callback = callback,
    dt = dt,
    adaptive = false,
    progress = show_progress_bar,
    progress_steps = 1,
    additional_solver_kwargs...,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

@info "Running `$test_name`"
sol = @timev OrdinaryDiffEq.solve!(integrator)

#@show typeof(sol.u)
#@show typeof(sol.u[1])
#@show typeof(sol.u[1].c)
#@show typeof(sol.u[1].f)
ENV["GKSwstype"] = "nul" # avoid displaying plots
postprocessing(sol, p, output_dir, usempi)
