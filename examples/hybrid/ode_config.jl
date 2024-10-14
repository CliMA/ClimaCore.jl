import SciMLBase
import ClimaTimeSteppers as CTS

is_explicit_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.ERKAlgorithmName

is_imex_CTS_algo_type(alg_or_tableau) =
    alg_or_tableau <: CTS.IMEXARKAlgorithmName

is_imex_CTS_algo(::CTS.IMEXAlgorithm) = true
is_imex_CTS_algo(::SciMLBase.AbstractODEAlgorithm) = false

use_transform(ode_algo) = !is_imex_CTS_algo(ode_algo)

function jac_kwargs(ode_algo, Y, jacobi_flags)
    if is_imex_CTS_algo(ode_algo)
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
    alg_or_tableau = getproperty(CTS, ode_sym)
    @info "Using ODE config: `$alg_or_tableau`"

    if is_explicit_CTS_algo_type(alg_or_tableau)
        return CTS.ExplicitAlgorithm(alg_or_tableau())
    elseif !is_imex_CTS_algo_type(alg_or_tableau)
        return alg_or_tableau()
    elseif is_imex_CTS_algo_type(alg_or_tableau)
        newtons_method = CTS.NewtonsMethod(; max_iters = max_newton_iters)
        return CTS.IMEXAlgorithm(alg_or_tableau(), newtons_method)
    else
        error("Uncaught case")
    end
end
