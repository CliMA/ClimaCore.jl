using SciMLBase: RECOMPILE_BY_DEFAULT

import SciMLBase: SplitFunction

#=
Issue: calc_W! attempts to access f.Wfact when has_Wfact(f) and f.Wfact_t when
       has_Wfact_t(f). However, when f is a SplitFunction, Wfact and Wfact_t are
       stored in f.f1, and the corresponding fields in f are set to nothing.
       In addition, build_J_W uses f.jac_prototype to construct J and W when
       f.jac_prototype !== nothing. However, when f is a SplitFunction,
       jac_prototype is stored in f.f1, and f.jac_prototype is set to nothing.
       Similarly, calc_tderivative! uses f.tgrad when has_tgrad(f).
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
        tgrad = f1.tgrad,
    )
end
