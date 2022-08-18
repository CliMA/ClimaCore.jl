#=
u_{n + 1} = u_n + dt * f(u_{n + 1}, t_{n + 1})
f(u_{n + 1}, t_{n + 1}) ≈ f(u_n, t_n) + J(u_n, t_n) * (u_{n + 1} - u_n) # assume no t-dependence
u_{n + 1} = u_n + dt * f(u_n, t_n) + dt * J(u_n, t_n) * (u_{n + 1} - u_n)
(I / dt - J(u_n, t_n)) (u_{n + 1} - u_n) = f(u_n, t_n)
u_{n + 1} = (I / dt - J(u_n, t_n)) \ f(u_n, t_n) + u_n
=#

OrdinaryDiffEq.nlsolve_f(f, alg::ClimaTimeSteppers.DistributedODEAlgorithm) =
    f isa OrdinaryDiffEq.SplitFunction ? f.f1 : f
OrdinaryDiffEq.ODEFunction{iip}(
    f::ClimaTimeSteppers.ForwardEulerODEFunction
) where {iip} = f
Base.@kwdef struct IMEXEulerAlgorithm{L, N} <:
        ClimaTimeSteppers.DistributedODEAlgorithm
    linsolve::L
    nlsolve::N
    extrapolant::Symbol = :linear
end
struct IMEXEulerCache{N} <: OrdinaryDiffEq.OrdinaryDiffEqMutableCache
    nlsolver::N
end
function ClimaTimeSteppers.cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXEulerAlgorithm;
    kwargs...
)
    u = uprev = rate_prototype = prob.u0
    p = prob.p
    t = prob.tspan[1]
    dt = eltype(t)(0) # unsure about this
    f = prob.f
    uEltypeNoUnits = uBottomEltypeNoUnits = tTypeNoUnits = FT
    γ = 1//1 # unsure about this
    c = 1 # unsure about this
    nlsolver = OrdinaryDiffEq.build_nlsolver(
        alg,
        u,
        uprev,
        p,
        t,
        dt,
        f,
        rate_prototype,
        uEltypeNoUnits,
        uBottomEltypeNoUnits,
        tTypeNoUnits,
        γ,
        c,
        Val(true),
    )
    return IMEXEulerCache(nlsolver)
end
function ClimaTimeSteppers.step_u!(integrator, cache::IMEXEulerCache)
    (; t, dt, u, prob, p, alg) = integrator
    (; f1, f2) = prob.f
    (; nlsolver) = cache
    (; tmp, z) = nlsolver

    if f2 isa ClimaTimeSteppers.ForwardEulerODEFunction
        f2(tmp, u, p, t, dt)
    elseif f2 isa OrdinaryDiffEq.ODEFunction
        f2(tmp, u, p, t)
        @. tmp = u + dt * tmp
    else
        error("Unexpected type of f2: $(typeof(f2))")
    end

    nlsolver.γ = 1 # do we need this?
    OrdinaryDiffEq.markfirststage!(nlsolver) # do we need this?

    # initial guess
    if alg.extrapolant == :linear
        @assert f1 isa OrdinaryDiffEq.ODEFunction
        f1(z, u, p, t)
        @. z *= dt
    else # :constant
        @. z = zero(eltype(u))
    end

    z = OrdinaryDiffEq.nlsolve!(nlsolver, integrator, cache, false)
    OrdinaryDiffEq.nlsolvefail(nlsolver) && return
    @. u = tmp + z
end