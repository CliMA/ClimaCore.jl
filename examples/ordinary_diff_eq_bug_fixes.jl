import OrdinaryDiffEq: alg_cache, nlsolve!
import OrdinaryDiffEq.SciMLBase: SplitFunction

using OrdinaryDiffEq:
    Rosenbrock32,
    build_J_W,
    constvalue,
    Rosenbrock32Tableau,
    TimeGradientWrapper,
    UJacobianWrapper,
    build_grad_config,
    build_jac_config,
    Rosenbrock32Cache,
    AbstractNLSolver,
    isnewton,
    DIRK,
    update_W!,
    initialize!,
    Divergence,
    get_new_W!,
    initial_η,
    compute_step!,
    apply_step!,
    Convergence,
    isJcurrent,
    postamble!,
    TryAgain
using OrdinaryDiffEq.SciMLBase: RECOMPILE_BY_DEFAULT

#=
Issue: Rosenbrock32 requires Wfact, rather than Wfact_t. However, the function
       that constructs a cache for Rosenbrock32 is missing a Val(false), so it
       attempts to construct a jacobian from Wfact_t.
=#
function alg_cache(
    alg::Rosenbrock32,
    u,
    rate_prototype,
    ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits},
    ::Type{tTypeNoUnits},
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    k₁ = zero(rate_prototype)
    k₂ = zero(rate_prototype)
    k₃ = zero(rate_prototype)
    du1 = zero(rate_prototype)
    du2 = zero(rate_prototype)
    # f₀ = zero(u) fsalfirst
    f₁ = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    fsallast = zero(rate_prototype)
    dT = zero(rate_prototype)
    J, W = build_J_W(alg, u, uprev, p, t, dt, f, uEltypeNoUnits, Val(true))
    tmp = zero(rate_prototype)
    atmp = similar(u, uEltypeNoUnits)
    tab = Rosenbrock32Tableau(constvalue(uBottomEltypeNoUnits))

    tf = TimeGradientWrapper(f, uprev, p)
    uf = UJacobianWrapper(f, t, p)
    linsolve_tmp = zero(rate_prototype)
    linsolve = alg.linsolve(Val{:init}, uf, u)
    grad_config = build_grad_config(alg, f, tf, du1, t)
    jac_config =
        build_jac_config(alg, f, uf, du1, uprev, u, tmp, du2, Val(false))
    Rosenbrock32Cache(
        u,
        uprev,
        k₁,
        k₂,
        k₃,
        du1,
        du2,
        f₁,
        fsalfirst,
        fsallast,
        dT,
        J,
        W,
        tmp,
        atmp,
        tab,
        tf,
        uf,
        linsolve_tmp,
        linsolve,
        jac_config,
        grad_config,
    )
end

#=
Issue: calc_W! attempts to access f.Wfact when has_Wfact(f) and f.Wfact_t when
       has_Wfact_t(f). However, when f is a SplitFunction, Wfact and Wfact_t are
       stored in f.f1, and the corresponding fields in f are set to nothing.
       In addition, build_J_W uses f.jac_prototype to construct J and W when
       f.jac_prototype !== nothing. However, when f is a SplitFunction,
       jac_prototype is stored in f.f1, and f.jac_prototype is set to nothing.
=#
function SplitFunction{iip}(f1, f2; kwargs...) where {iip}
    f1 = ODEFunction(f1)
    f2 = ODEFunction{iip}(f2)
    SplitFunction{iip, RECOMPILE_BY_DEFAULT}(
        f1,
        f2;
        kwargs...,
        Wfact = f1.Wfact,
        Wfact_t = f1.Wfact_t,
        jac_prototype = f1.jac_prototype,
    )
end

#=
Issue: nlsolve! has a goto statement that restarts the solver when the
       conditions for DiffEqBase.SlowConvergence (denoted by TryAgain) are
       satisfied. These conditions include !isJcurrent, which is always true
       when a custom Wfact or Wfact_t is used because of how calc_W! is written.
       So, when a custom Wfact or Wfact_t is used, nlsolve! sometimes enters an
       infinite loop (though there are also situations where it converges after
       several iterations of the loop). This is probably not the intended
       behavior, so this patch limits the number of loop iterations to 100,
       after which divergence is declared.
=#
function nlsolve!(
    nlsolver::AbstractNLSolver,
    integrator,
    cache = nothing,
    repeat_step = false,
)
    num_redos = 0
    @label REDO
    if isnewton(nlsolver)
        cache === nothing && throw(
            ArgumentError(
                "cache is not passed to `nlsolve!` when using NLNewton",
            ),
        )
        if nlsolver.method === DIRK
            γW = nlsolver.γ * integrator.dt
        else
            γW = nlsolver.γ * integrator.dt / nlsolver.α
        end
        update_W!(nlsolver, integrator, cache, γW, repeat_step)
    end

    @unpack maxiters, κ, fast_convergence_cutoff = nlsolver

    initialize!(nlsolver, integrator)
    nlsolver.status = Divergence
    η = get_new_W!(nlsolver) ? initial_η(nlsolver, integrator) : nlsolver.ηold

    local ndz
    for iter in 1:maxiters
        nlsolver.iter = iter

        # compute next step and calculate norm of residuals
        iter > 1 && (ndzprev = ndz)
        ndz = compute_step!(nlsolver, integrator)
        if !isfinite(ndz)
            nlsolver.status = Divergence
            break
        end

        # check divergence (not in initial step)
        if iter > 1
            θ = ndz / ndzprev

            # divergence
            if θ > 2
                nlsolver.status = Divergence
                break
            end
        end

        apply_step!(nlsolver, integrator)

        # check for convergence
        iter > 1 && (η = θ / (1 - θ))
        if (iter == 1 && ndz < 1e-5) ||
           (iter > 1 && (η >= zero(η) && η * ndz < κ))
            nlsolver.status = Convergence
            break
        end
    end

    if isnewton(nlsolver) &&
       nlsolver.status == Divergence &&
       !isJcurrent(nlsolver, integrator)
        num_redos += 1
        if num_redos < 100
            nlsolver.status = TryAgain
            @goto REDO
        end
    end

    nlsolver.ηold = η
    postamble!(nlsolver, integrator)
end
