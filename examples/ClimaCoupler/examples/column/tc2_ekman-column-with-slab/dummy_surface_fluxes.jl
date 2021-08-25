export calculate_sfc_fluxes_energy
export DryBulkFormulaWithRadiation, DryBulkFormula, LinearRelaxation
#surface_fluxes
"""
dummy for SurfaceFluxes.jl
"""
abstract type SurfaceFluxType end

struct LinearRelaxation <: SurfaceFluxType end
struct DryBulkFormula <: SurfaceFluxType end
struct DryBulkFormulaWithRadiation <: SurfaceFluxType end

calculate_sfc_fluxes_energy(forumation::LinearRelaxation, p, T_sfc, T1) = p.λ .* (T_sfc .- T1)

calculate_sfc_fluxes_energy(forumation::DryBulkFormula, p, T_sfc, T1, u_1, v_1, ρ_1 ) = p.Ch * p.C_p * ρ_1 * sqrt(u_1^2 + v_1^2) * (T_sfc .- T1)

function calculate_sfc_fluxes_energy(forumation::DryBulkFormulaWithRadiation, parameters, T_sfc, T1, u_1, v_1, ρ_1, t ) 

    p = parameters

    R_SW = (1-p.α) * p.τ * p.F_sol * (1 .+ sin(t * 2π / p.τ_d) )
    R_LW = p.ϵ * (p.σ * T_sfc .^ 4 - p.σ * T1 .^ 4)
    SH   = p.Ch * p.C_p * ρ_1 * sqrt(u_1^2 + v_1^2) * (T_sfc - T1) 
    #p_sfc = p.pₒ
    #LH   = p.lambda * p.g_w * (q_sat(T_sfc, p_sfc) - q_a) 

    F_tot = - (R_SW - R_LW - SH )#- LH 
end