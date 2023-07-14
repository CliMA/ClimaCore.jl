import DiffEqBase
import OrdinaryDiffEq as ODE
import ClimaTimeSteppers as CTS

is_explicit_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.ERKAlgorithmName

is_imex_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.IMEXARKAlgorithmName

is_implicit_type(::typeof(ODE.IMEXEuler)) = true
is_implicit_type(alg_or_tableau) =
    alg_or_tableau <: Union{
        ODE.OrdinaryDiffEqImplicitAlgorithm,
        ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm,
    } || is_imex_CTS_algo_type(alg_or_tableau)

is_ordinary_diffeq_newton(::typeof(ODE.IMEXEuler)) = true
is_ordinary_diffeq_newton(alg_or_tableau) =
    alg_or_tableau <: Union{
        ODE.OrdinaryDiffEqNewtonAlgorithm,
        ODE.OrdinaryDiffEqNewtonAdaptiveAlgorithm,
    }

is_imex_CTS_algo(::CTS.IMEXAlgorithm) = true
is_imex_CTS_algo(::DiffEqBase.AbstractODEAlgorithm) = false

is_implicit(::ODE.OrdinaryDiffEqImplicitAlgorithm) = true
is_implicit(::ODE.OrdinaryDiffEqAdaptiveImplicitAlgorithm) = true
is_implicit(ode_algo) = is_imex_CTS_algo(ode_algo)

is_rosenbrock(::ODE.Rosenbrock23) = true
is_rosenbrock(::ODE.Rosenbrock32) = true
is_rosenbrock(::DiffEqBase.AbstractODEAlgorithm) = false
use_transform(ode_algo) =
    !(is_imex_CTS_algo(ode_algo) || is_rosenbrock(ode_algo))

function jac_kwargs(ode_algo, Y, jacobi_flags)
    if is_implicit(ode_algo)
        W = SchurComplementW(Y, use_transform(ode_algo), jacobi_flags)
        if use_transform(ode_algo)
            return (; jac_prototype = W, Wfact_t = Wfact!)
        else
            return (; jac_prototype = W, Wfact = Wfact!)
        end
    else
        return NamedTuple()
    end
end

function ode_configuration(
    ::Type{FT};
    ode_name::Union{String, Nothing} = nothing,
    max_newton_iters = nothing,
) where {FT}
    if occursin(".", ode_name)
        ode_name = split(ode_name, ".")[end]
    end
    ode_sym = Symbol(ode_name)
    alg_or_tableau = if hasproperty(ODE, ode_sym)
        @warn "apply_limiter flag is ignored for OrdinaryDiffEq algorithms"
        getproperty(ODE, ode_sym)
    else
        getproperty(CTS, ode_sym)
    end
    @info "Using ODE config: `$alg_or_tableau`"

    if is_explicit_CTS_algo_type(alg_or_tableau)
        return CTS.ExplicitAlgorithm(alg_or_tableau())
    elseif !is_implicit_type(alg_or_tableau)
        return alg_or_tableau()
    elseif is_ordinary_diffeq_newton(alg_or_tableau)
        if max_newton_iters == 1
            error("OridinaryDiffEq requires at least 2 Newton iterations")
        end
        # κ like a relative tolerance; its default value in ODE is 0.01
        nlsolve = ODE.NLNewton(;
            κ = max_newton_iters == 2 ? Inf : 0.01,
            max_iter = max_newton_iters,
        )
        return alg_or_tableau(; linsolve = linsolve!, nlsolve)
    elseif is_imex_CTS_algo_type(alg_or_tableau)
        newtons_method = CTS.NewtonsMethod(; max_iters = max_newton_iters)
        return CTS.IMEXAlgorithm(alg_or_tableau(), newtons_method)
    else
        return alg_or_tableau(; linsolve = linsolve!)
    end
end
