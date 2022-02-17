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

const FT = Float64

default_test_name = "balanced_flow_rhotheta"
test_implicit_solver = false # makes solver extremely slow when set to `true`

################################################################################

if haskey(ENV, "TEST_NAME")
    test_name = ENV["TEST_NAME"]
else
    test_name = default_test_name
end

include("utilities.jl")
include("$test_name.jl")

@unpack zmax,
velem,
helem,
npoly,
tmax,
dt,
ode_algorithm,
jacobian_flags, # relevant for implicit/IMEX ODE algorithms
max_newton_iters, # relevant for ODE algorithms that use Newton's method
save_every_n_steps,
additional_solver_kwargs = driver_values(FT)

center_local_geometry, face_local_geometry =
    local_geometry_fields(FT, zmax, velem, helem, npoly)
Yc = map(initial_condition, center_local_geometry)
uₕ = map(initial_condition_velocity, center_local_geometry)
w = map(_ -> Geometry.Covariant3Vector(FT(0.0)), face_local_geometry)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)
p = merge(implicit_cache_values(Y, dt), remaining_cache_values(Y, dt))

use_transform = !(ode_algorithm in (Rosenbrock23, Rosenbrock32))
W_kwarg = use_transform ? (; Wfact_t = Wfact!) : (; Wfact = Wfact!)
W = SchurComplementW(Y, use_transform, jacobian_flags, test_implicit_solver)
problem = SplitODEProblem(
    ODEFunction(
        implicit_tendency!;
        W_kwarg...,
        jac_prototype = W,
        tgrad = (dY, Y, p, t) -> (dY .= zero(eltype(dY))),
    ),
    remaining_tendency!,
    Y,
    (FT(0.0), tmax),
    p,
)

alg_kwargs = (;)
if ode_algorithm <: Union{
    OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm,
    OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
}
    alg_kwargs = (; alg_kwargs..., linsolve = linsolve!)
    if ode_algorithm <: Union{
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAlgorithm,
        OrdinaryDiffEq.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }
        alg_kwargs =
            (; alg_kwargs, nlsolve = NLNewton(; max_iter = max_newton_iters))
    end
end

mkpath(joinpath(@__DIR__, "checkpoints"))
saver = SaveJLD2(
    joinpath(@__DIR__, "checkpoints", test_name),
    dt * save_every_n_steps,
    0.0,
)

integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm(; alg_kwargs...);
    dt = dt,
    saveat = dt * save_every_n_steps,
    adaptive = false,
    progress = true,
    progress_steps = 1,
    callback = DiscreteCallback(saver, integrator -> nothing),
    additional_solver_kwargs...,
)

if haskey(ENV, "RESTART_FILE")
    Yr, t = load(ENV["RESTART_FILE"], "u", "t")
    set_t!(integrator, t)
    saver.next_t = t
    Y = Fields.FieldVector(
        Yc = Fields.Field(Fields.field_values(Yr.Yc), axes(Y.Yc)),
        uₕ = Fields.Field(Fields.field_values(Yr.uₕ), axes(Y.uₕ)),
        w = Fields.Field(Fields.field_values(Yr.w), axes(Y.w)),
    )
    set_u!(integrator, Y)
end

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

ENV["GKSwstype"] = "nul"
path = joinpath(@__DIR__, "output", test_name)
mkpath(path)

postprocessing(sol, path)
