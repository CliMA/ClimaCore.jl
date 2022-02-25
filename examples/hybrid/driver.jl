if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

import ClimaCore: enable_threading
enable_threading() = true

using OrdinaryDiffEq
using DiffEqCallbacks
using JLD2

const FT = Float64
default_test_name = "sphere/balanced_flow_rhotheta"
test_implicit_solver = false # makes solver extremely slow when set to `true`

# Definitions that are specific to each test:
space = nothing
tend = nothing
dt = nothing
dt_save_to_sol = FT(0) # 0 means don't save at all
dt_save_to_disk = FT(0) # 0 means don't save at all
ode_algorithm = OrdinaryDiffEq.SSPRK33
jacobian_flags = (;) # only required by implicit ODE algorithms
max_newton_iters = 10 # only required by ODE algorithms that use Newton's method
show_progress_bar = false
additional_callbacks = () # e.g., printing diagnostic information
additional_solver_kwargs = (;) # e.g., abstol and reltol
center_initial_condition(local_geometry) = nothing
face_initial_condition(local_geometry) = nothing
postprocessing(sol, p, output_dir) = nothing
# Values set to `nothing` must be overwritten in every test file; other values
# may optionally be overwritten.

################################################################################

if haskey(ENV, "TEST_NAME")
    test_name = ENV["TEST_NAME"]
else
    test_name = default_test_name
end
test_dir, test_file_name = split(test_name, '/')

include("staggered_nonhydrostatic_model.jl")
include(joinpath(test_dir, "$test_file_name.jl"))

if haskey(ENV, "RESTART_FILE")
    jldopen(ENV["RESTART_FILE"]) do restart_data
        tstart = restart_data["t"]
        Y = restart_data["Y"]
    end
    ᶜlocal_geometry = Fields.local_geometry_field(Y.c)
    ᶠlocal_geometry = Fields.local_geometry_field(Y.f)
else
    tstart = FT(0)
    ᶜlocal_geometry, ᶠlocal_geometry = local_geometry_fields(space)
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
    output_dir = joinpath(@__DIR__, "output", test_dir, test_file_name)
end
mkpath(output_dir)
function save_to_disk(integrator)
    t = integrator.t
    day = floor(Int, t / (60 * 60 * 24))
    @info "Saving prognostic variables to JLD2 file on day $day"
    jldsave(joinpath(output_dir, "day$day.jld2"); t, Y = integrator.u)
    return nothing
end
if dt_save_to_disk == 0
    saving_callback = nothing
else
    saving_callback =
        PeriodicCallback(save_to_disk, dt_save_to_disk; initial_affect = true)
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
    (tstart, tend),
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

ENV["GKSwstype"] = "nul" # avoid displaying plots
postprocessing(sol, p, output_dir)
