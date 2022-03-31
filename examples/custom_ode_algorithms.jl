using OrdinaryDiffEq:
    OrdinaryDiffEqRosenbrockAlgorithm,
    DEFAULT_LINSOLVE,
    _unwrap_val,
    _transformtab,
    RosenbrockFixedTableau,
    _masktab,
    gen_tableau_struct,
    gen_tableau,
    gen_cache_struct,
    OrdinaryDiffEqConstantCache,
    RosenbrockMutableCache,
    gen_algcache,
    gen_initialize,
    gen_constant_perform_step,
    gen_perform_step,
    @muladd,
    @..,
    build_J_W,
    constvalue,
    TimeGradientWrapper,
    UJacobianWrapper,
    build_grad_config,
    build_jac_config,
    calc_rosenbrock_differentiation!

import OrdinaryDiffEq: alg_cache, initialize!, perform_step!

"""
https://github.com/CliMA/CGDycore.jl/blob/main/src/IntegrationMethods/RosenbrockMethod.jl
"""
function oswaldsBagOfTableaus(algname)
    if algname == :SSPKnoth
        Alpha = [0 0 0; 1 0 0; 1/4 1/4 0]
        Gamma = [1 0 0; 0 1 0; -3/4 -3/4 1]
        B = [1/6, 1/6, 2/3]
    elseif algname == :RK3_H
        Alpha = [0 0 0; 1/3 0 0; 0 1/2 0]
        γ = (3 + sqrt(3)) / 6
        Gamma = [γ 0 0; (1 - 12γ^2)/(-9 + 36γ) γ 0; (-1/4 + 2γ) (1/4 - 3γ) γ]
        B = [0, 0, 1]
    elseif algname == :RODAS
        Alpha = [0 0 0 0; 0 0 0 0; 1 0 0 0; 3/4 -1/4 1/2 0]
        Gamma = [1/2 0 0 0; 1 1/2 0 0; -1/4 -1/4 1/2 0; 1/12 1/12 -2/3 1/2]
        B = [5/6, -1/6, -1/6, 1/2]
    elseif algname == :TSROSWSANDU3
        Alpha = [
            0 0 0
            0.43586652150845899941601945119356 0 0
            0.43586652150845899941601945119356 0 0
        ]
        Gamma = [
            0.43586652150845899941601945119356 0 0
            -0.19294655696029095575009695436041 0.43586652150845899941601945119356 0
            0 1.74927148125794685173529749738960 0.43586652150845899941601945119356
        ]
        B = [
            -0.75457412385404315829818998646589
            1.94100407061964420292840123379419
            -0.18642994676560104463021124732829
        ]
    elseif algname == :TROSWLASSP3P4S2C
        Alpha = [0 0 0 0; 1/2 0 0 0; 1/2 1/2 0 0; 1/6 1/6 1/6 0]
        Gamma = [1/2 0 0 0; 0 3/4 0 0; -2/3 -23/9 2/9 0; 1/18 65/108 -2/27 0]
        B = [1/6, 1/6, 1/6, 1/2]
    elseif algname == :ROSAMF
        Alpha = [0 0; 2/3 0]
        γ = (2 + sqrt(3)) / 6
        Gamma = [γ 0; (-4/3 * γ) γ]
        B = [1/4, 3/4]
    elseif algname == :ROS2
        Alpha = [0 0; 2/3 0]
        γ = (1/2 + sqrt(3)) / 6
        Gamma = [γ 0; (-4/3 * γ) γ]
        B = [1/4, 3/4]
    end
    Bhat = B # Bhat does not matter, since the algorithm is not adaptive
    a, C, b, btilde, d, c = _transformtab(Alpha, Gamma, B, Bhat) # ignore btilde
    gamma = Gamma[1, 1]
    return RosenbrockFixedTableau(a, C, b, gamma, d, c)
end

# Importing @cache from OrdinaryDiffEq doesn't work because of namespace issues.
macro cache(expr)
    expr
end

macro make_ode_algorithm(algnameexpr)
    algname = algnameexpr.value
    tabname = Symbol(algname, :Tableau)
    tabstructname = Symbol(algname, :TableauStruct)
    cachename = Symbol(algname, :Cache)
    constcachename = Symbol(algname, :ConstantCache)
    tab = oswaldsBagOfTableaus(algname)
    tabmask = _masktab(tab)
    n_normalstep = length(tab.b) - 1
    tabstructexpr = gen_tableau_struct(tabmask, tabstructname)
    tabexpr = gen_tableau(tab, tabstructexpr, tabname)
    constcacheexpr, cacheexpr =
        gen_cache_struct(tabmask, cachename, constcachename)
    algcacheexpr = gen_algcache(cacheexpr, constcachename, algname, tabname)
    initializeexpr = gen_initialize(cachename, constcachename)
    constperformstepexpr =
        gen_constant_perform_step(tabmask, constcachename, n_normalstep)
    performstepexpr = gen_perform_step(tabmask, cachename, n_normalstep)
    expr = quote
        struct $algname{CS, AD, F, FDT, ST} <:
                OrdinaryDiffEqRosenbrockAlgorithm{CS, AD, FDT, ST}
            linsolve::F
        end
        $algname(;
            chunk_size = Val{0}(),
            autodiff = true,
            standardtag = Val{true}(),
            diff_type = Val{:central},
            linsolve = DEFAULT_LINSOLVE,
        ) = $algname{
            _unwrap_val(chunk_size),
            _unwrap_val(autodiff),
            typeof(linsolve),
            diff_type,
            _unwrap_val(standardtag),
        }(linsolve)
        $tabstructexpr
        $tabexpr
        $constcacheexpr
        $cacheexpr
        $algcacheexpr
        $initializeexpr
        $constperformstepexpr
        $performstepexpr
    end
    return esc(expr)
end

@make_ode_algorithm(:SSPKnoth)
@make_ode_algorithm(:RK3_H)
@make_ode_algorithm(:RODAS)
@make_ode_algorithm(:TSROSWSANDU3)
# @make_ode_algorithm(:TROSWLASSP3P4S2C) # LAPACKException(4)???
@make_ode_algorithm(:ROSAMF)
@make_ode_algorithm(:ROS2)

# TODO: ROS3Pw and ROSRK3 (can't use _transformtab), isWmethod, alg_order
