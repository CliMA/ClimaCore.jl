import OrdinaryDiffEq: alg_cache
import OrdinaryDiffEq.SciMLBase: SplitFunction

using OrdinaryDiffEq: Rosenbrock32, build_J_W, constvalue, Rosenbrock32Tableau,
    TimeGradientWrapper, UJacobianWrapper, build_grad_config, build_jac_config,
    Rosenbrock32Cache
using OrdinaryDiffEq.SciMLBase: RECOMPILE_BY_DEFAULT

#=
Issue: Rosenbrock32 requires Wfact_t, rather than Wfact. However, the function
       that constructs a cache for Rosenbrock32 is missing a Val(false), which
       causes it to attempt to construct a jacobian from Wfact.
=#
function alg_cache(alg::Rosenbrock32,u,rate_prototype,::Type{uEltypeNoUnits},::Type{uBottomEltypeNoUnits},::Type{tTypeNoUnits},uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true}) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
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
  J,W = build_J_W(alg,u,uprev,p,t,dt,f,uEltypeNoUnits,Val(true))
  tmp = zero(rate_prototype)
  atmp = similar(u, uEltypeNoUnits)
  tab = Rosenbrock32Tableau(constvalue(uBottomEltypeNoUnits))

  tf = TimeGradientWrapper(f,uprev,p)
  uf = UJacobianWrapper(f,t,p)
  linsolve_tmp = zero(rate_prototype)
  linsolve = alg.linsolve(Val{:init},uf,u)
  grad_config = build_grad_config(alg,f,tf,du1,t)
  jac_config = build_jac_config(alg,f,uf,du1,uprev,u,tmp,du2,Val(false))
  Rosenbrock32Cache(u,uprev,k₁,k₂,k₃,du1,du2,f₁,fsalfirst,fsallast,dT,J,W,tmp,atmp,tab,tf,uf,linsolve_tmp,linsolve,jac_config,grad_config)
end

#=
Issue: calc_W! attempts to access f.Wfact when has_Wfact(f) and f.Wfact_t when
       has_Wfact_t(f). However, when f is a SplitFunction, Wfact and Wfact_t are
       stored in f.f1, and the corresponding fields in f are set to nothing.

       In addition, build_J_W uses f.jac_prototype to construct J and W when
       f.jac_prototype !== nothing. However, when f is a SplitFunction,
       jac_prototype is stored in f.f1, and f.jac_prototype === nothing.
=#
function SplitFunction{iip}(f1,f2; kwargs...) where iip
  f1 = ODEFunction(f1)
  f2 = ODEFunction{iip}(f2)
  SplitFunction{iip,RECOMPILE_BY_DEFAULT}(f1,f2; kwargs...,
                                          Wfact = f1.Wfact,
                                          Wfact_t = f1.Wfact_t,
                                          jac_prototype = f1.jac_prototype,
  )
end