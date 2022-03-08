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

default_test_name = "baroclinic_wave_rhoe"
test_implicit_solver = false # makes solver extremely slow when set to `true`

################################################################################

if haskey(ENV, "TEST_NAME")
    test_name = ENV["TEST_NAME"]
else
    test_name = default_test_name
end

ENV["GKSwstype"] = "nul"
path = joinpath(@__DIR__, "output", test_name)
mkpath(path)

if haskey(ENV, "OUTPUT_DIR")
    output_dir = ENV["OUTPUT_DIR"]
    mkpath(output_dir)
else
    output_dir = path
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

if haskey(ENV, "RESTART_INFILE")
    data_restart = jldopen(ENV["RESTART_INFILE"])
    Yc = data_restart["uend"].Yc
    uₕ = data_restart["uend"].uₕ
    w = data_restart["uend"].w
    const tend = data_restart["tend"]
else
    Yc = map(initial_condition, center_local_geometry)
    uₕ = map(initial_condition_velocity, center_local_geometry)
    w = map(_ -> Geometry.Covariant3Vector(FT(0.0)), face_local_geometry)
    const tend = FT(0)
end

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
integrator = OrdinaryDiffEq.init(
    problem,
    ode_algorithm(; alg_kwargs...);
    dt = dt,
    saveat = dt * save_every_n_steps,
    adaptive = false,
    progress = false,
    additional_solver_kwargs...,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol = @timev OrdinaryDiffEq.solve!(integrator)

postprocessing(sol, path)
