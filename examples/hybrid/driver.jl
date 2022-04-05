# Test-specific definitions; these can be redefined in every test file
setups = [] # should be a collection of `HybridDriverSetup`s (see below)
postprocessing(sols, output_dir) = nothing # processes the results of `setups`
test_implicit_solver = false # makes solver extremely slow when set to `true`

################################################################################

if !haskey(ENV, "FLOAT_TYPE") || ENV["FLOAT_TYPE"] == "Float32"
    const FT = Float32
elseif ENV["FLOAT_TYPE"] == "Float64"
    const FT = Float64
else
    error("ENV[\"FLOAT_TYPE\"] can only be set to \"Float32\" or \"Float64\"")
end

################################################################################

using ClimaCore: Meshes
using OrdinaryDiffEq

default_additional_cache(ᶜlocal_geometry, ᶠlocal_geometry, dt) = (;)
default_additional_tendency!(Yₜ, Y, p, t) = nothing
default_center_initial_condition(local_geometry) = (;)
default_face_initial_condition(local_geometry) = (;)

# TODO: Add parameter specification to HybridDriverSetup, and get default
# parameter values using ClimaParameters.
Base.@kwdef struct HybridDriverSetup{AC, AT, CIC, FIC, HM, OA, JF, ACB, ASK}
    # Model specification
    additional_cache::AC = default_additional_cache
    additional_tendency!::AT = default_additional_tendency!

    # Initial condition specification
    center_initial_condition::CIC = default_center_initial_condition
    face_initial_condition::FIC = default_face_initial_condition
    horizontal_mesh::HM = nothing # no good default value
    npoly::Int = 1
    z_max::FT = FT(0)
    z_elem::Int = 1
    restart_file_name::String = "" # empty string means don't use restart file

    # Timestepping specification
    t_end::FT = FT(0)
    dt::FT = FT(0)
    dt_save_to_sol::FT = FT(0) # 0 means don't save to sol
    dt_save_to_disk::FT = FT(0) # 0 means don't save to disk
    ode_algorithm::OA = SSPRK33 # good general-purpose explicit ODE algorithm
    jacobian_flags::JF = (;) # only for implicit algs
    max_newton_iters::Int = 10 # default in OrdinaryDiffEq; only for Newton algs
    additional_callbacks::ACB = () # e.g., for printing diagnostic information
    additional_solver_kwargs::ASK = (;) # e.g., for setting abstol and reltol
end

################################################################################

is_distributed = haskey(ENV, "CLIMACORE_DISTRIBUTED")

using Logging
if is_distributed
    using ClimaComms
    if ENV["CLIMACORE_DISTRIBUTED"] == "MPI"
        using ClimaCommsMPI
        const Context = ClimaCommsMPI.MPICommsContext
    else
        error("ENV[\"CLIMACORE_DISTRIBUTED\"] can only be set to \"MPI\"")
    end
    const pid, nprocs = ClimaComms.init(Context)
    logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull
    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    @info "Using ClimaCore in distributed mode with $nprocs \
        processor$(nprocs == 1 ? "" : "s")"
else
    using TerminalLoggers: TerminalLogger
    prev_logger = global_logger(TerminalLogger())
end
atexit() do
    global_logger(prev_logger)
end

import ClimaCore: enable_threading
if haskey(ENV, "CLIMACORE_MULTITHREADED")
    enable_threading() = true
    @info "Using ClimaCore in multi-threaded mode with $(Threads.nthreads()) \
        thread$(Threads.nthreads() == 1 ? "" : "s")"
else
    enable_threading() = false
end

using DiffEqCallbacks
using JLD2

include("../implicit_solver_debugging_tools.jl")
include("../ordinary_diff_eq_bug_fixes.jl")
include("../common_spaces.jl")

if haskey(ENV, "TEST_NAME")
    test_dir, test_file_name = split(ENV["TEST_NAME"], '/')
else
    error("ENV[\"TEST_NAME\"] required (e.g., \"sphere/baroclinic_wave_rhoe\")")
end
include(joinpath(test_dir, "$test_file_name.jl"))
@info "Running `$test_dir/$test_file_name`"

output_dir = get(
    ENV,
    "OUTPUT_DIR",
    joinpath(@__DIR__, test_dir, "output", test_file_name, string(FT)),
)
mkpath(output_dir)

function make_dss_func(comms_ctx)
    _dss!(x::Fields.Field) = Spaces.weighted_dss!(x, comms_ctx)
    _dss!(::Any) = nothing
    dss_func(Y, t, integrator) = foreach(_dss!, Fields._values(Y))
    return dss_func
end
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

set_zero_tgrad!(∂Y∂t, Y, p, t) = ∂Y∂t .= FT(0)

# Using @goto and @label instead of a for loop removes the need for `global`
# annotations on all variables useful for debugging. If any setup causes the
# simulation to crash, all the variables defined below for that
# TODO: Eliminate redundant operations (e.g., if multiple setups use the same
# initial conditions, don't recompute the initial conditions every time).
# TODO: Allow multiple setups to be run in parallel.
indices = 1:length(setups)
sols = similar(setups, SciMLBase.AbstractODESolution)
begin
    iterator = iterate(indices)
    @label driver_loop
    if !isnothing(iterator)
        index, state = iterator
        (;
            additional_cache,
            additional_tendency!,
            center_initial_condition,
            face_initial_condition,
            horizontal_mesh,
            npoly,
            z_max,
            z_elem,
            restart_file_name,
            t_end,
            dt,
            dt_save_to_sol,
            dt_save_to_disk,
            ode_algorithm,
            jacobian_flags,
            max_newton_iters,
            additional_callbacks,
            additional_solver_kwargs,
        ) = setups[index]

        restart_file_name = get(ENV, "RESTART_FILE", restart_file_name)
        if restart_file_name == ""
            t_start = FT(0)
            if is_distributed
                # TODO: When is_distributed is true, automatically compute the
                # maximum number base type objects required to store an element
                # from Y.c or Y.f (or, to be accurate, an element from any Field
                # on which gather() or weighted_dss!() will get called). One
                # option is to make a non-distributed space, extract the
                # local_geometry type, and find the sizes of the output types of
                # center_initial_condition() and face_initial_condition() for
                # that local_geometry type. This is rather inefficient, though,
                # so for now we will just hardcode the value of 4.
                max_field_element_size = 4 # ρ = 1 FT, 𝔼 = 1 FT, uₕ = 2 FTs
                h_space, comms_ctx = make_distributed_horizontal_space(
                    horizontal_mesh,
                    npoly,
                    Context,
                    z_elem + 1,
                    max_field_element_size,
                )
            else
                h_space = make_horizontal_space(horizontal_mesh, npoly)
                comms_ctx = nothing
            end
            center_space, face_space =
                make_hybrid_spaces(h_space, z_max, z_elem)
            ᶜlocal_geometry = Fields.local_geometry_field(center_space)
            ᶠlocal_geometry = Fields.local_geometry_field(face_space)
            Y = Fields.FieldVector(
                c = center_initial_condition.(ᶜlocal_geometry),
                f = face_initial_condition.(ᶠlocal_geometry),
            )
        else
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
            if is_distributed
                comms_ctx = Spaces.setup_comms(
                    Context,
                    axes(Y.c).horizontal_space.topology, # Y.f would also work
                    Spaces.Quadratures.GLL{npoly + 1}(),
                    Spaces.nlevels(axes(Y.f)),
                    max(sizeof(eltype(Y.c)), sizeof(eltype(Y.f))) ÷
                    sizeof(eltype(Y)),
                )
            else
                comms_ctx = nothing
            end
        end
        p = get_cache(
            ᶜlocal_geometry,
            ᶠlocal_geometry,
            additional_cache,
            additional_tendency!,
            comms_ctx,
            dt,
        )

        if ode_algorithm <: Union{
            OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
            OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
        }
            use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
            W = SchurComplementW(
                Y,
                use_transform,
                jacobian_flags,
                test_implicit_solver,
            )
            jac_kwargs =
                use_transform ? (; jac_prototype = W, Wfact_t = Wfact!) :
                (; jac_prototype = W, Wfact = Wfact!)

            alg_kwargs = (; linsolve = linsolve!)
            if ode_algorithm <: Union{
                OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
                OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
            }
                alg_kwargs = (;
                    alg_kwargs...,
                    nlsolve = NLNewton(; max_iter = max_newton_iters),
                )
            end
        else
            jac_kwargs = alg_kwargs = (;)
        end

        dss_callback =
            FunctionCallingCallback(make_dss_func(comms_ctx); func_start = true)
        if dt_save_to_disk == 0
            save_to_disk_callback = nothing
        else
            save_to_disk_callback = PeriodicCallback(
                make_save_to_disk_func(
                    output_dir,
                    test_file_name,
                    is_distributed,
                ),
                dt_save_to_disk;
                initial_affect = true,
            )
        end
        callback = CallbackSet(
            dss_callback,
            save_to_disk_callback,
            additional_callbacks...,
        )

        ode_function = SplitFunction(
            ODEFunction(
                implicit_tendency!;
                jac_kwargs...,
                tgrad = set_zero_tgrad!,
            ),
            remaining_tendency!;
        )
        integrator = init(
            SplitODEProblem(ode_function, Y, (t_start, t_end), p),
            ode_algorithm(; alg_kwargs...);
            saveat = dt_save_to_sol == 0 ? [] : dt_save_to_sol,
            callback = callback,
            dt = dt,
            adaptive = false,
            progress = isinteractive(), # show progress bar when running in REPL
            progress_steps = 1,
            additional_solver_kwargs...,
        )

        if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
            throw(:exit_profile)
        end

        sols[index] = @timev solve!(integrator)

        iterator = iterate(indices, state)
        @goto driver_loop
    end
end

if is_distributed
    # Replace sol.u on the root processor with the sol.u.
    function global_sol(sol, setup)
        (; horizontal_mesh, npoly, z_max, z_elem) = setup
        if ClimaComms.iamroot(Context)
            global_h_space = make_horizontal_space(horizontal_mesh, npoly)
            global_center_space, global_face_space =
                make_hybrid_spaces(global_h_space, z_max, z_elem)
            global_Y_c_type = Fields.Field{
                typeof(Fields.field_values(Y.c)),
                typeof(global_center_space),
            }
            global_Y_f_type = Fields.Field{
                typeof(Fields.field_values(Y.f)),
                typeof(global_face_space),
            }
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
            if ClimaComms.iamroot(Context)
                global_sol_u[i] = Fields.FieldVector(
                    c = Fields.Field(global_Y_c, global_center_space),
                    f = Fields.Field(global_Y_f, global_face_space),
                )
            end
        end
        if ClimaComms.iamroot(Context)
            return DiffEqBase.sensitivity_solution(sol, global_sol_u, sol.t)
        else
            return nothing
        end
    end
    sols = map(global_sol, sols, setups)
end
if !is_distributed || (is_distributed && ClimaComms.iamroot(Context))
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    postprocessing(sols, output_dir)
end
