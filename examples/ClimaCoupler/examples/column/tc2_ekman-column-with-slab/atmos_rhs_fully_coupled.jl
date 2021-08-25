# Atmos RHS

#=
Ekman column:
    ∂_t ρ =  ∇ (μ ∇ ρ - w ρ) 
    ∂_t ρθ =  ∇ (μ ∇ ρθ - w ρθ)
    ∂_t u =  ∇ (μ ∇ u - w u) + (v - v_g)
    ∂_t v =  ∇ (μ ∇ v - w v) - (u - u_g) 
    ∂_t w =  ∇ (μ ∇ w - w w) - g - c_p θ ∂_z Π  

where 
    Π = (p/p_0)^{R/c_p}

top BCs are insulating and impenetrable:
    ∂_t T = 0           
    u = u_g             
    v = v_g             
    w = 0.0            
    ∂_t ρ = 0     

and bottom BCs use bulk formulae for surface fluxes of heat and momentum:
    ∂_t ρθ = F₃ = -Ch ρ ||u|| (T_sfc - ρθ / ρ)       
    ∂_t u  = F₁ = -Cd u ||u||                        
    ∂_t v  = F₂ = -Cd v ||u||                        
    w = 0.0                                     
    ∂_t ρ = 0                                   

We also use this model to accumulate fluxes it calculates
    ∂_t F_accum = -(F₁, F₂, F₃)
=#
function ∑tendencies_atm!(dY, Y, (parameters, T_sfc), t)

    UnPack.@unpack Cd, f, ν, ug, vg, C_p, MSLP, R_d, R_m, C_v, grav = parameters

    (Yc, Yf, _ ) = Y.x
    (dYc, dYf, dF_sfc) = dY.x
    UnPack.@unpack ρ, u, v, ρθ = Yc
    UnPack.@unpack w = Yf
    dρ = dYc.ρ
    du = dYc.u
    dv = dYc.v
    dρθ = dYc.ρθ
    dw = dYf.w

    # auxiliary calculations of surface variables (can be replaced by functions which call exatrapolation or MO)
    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate()) # XXX is this correct? (testing: all elems aside from boundaries change)
    u_1 = parent(If(u))[1]
    v_1 = parent(If(v))[1]
    ρ_1 = parent(If(ρ))[1]
    θ_1 = parent(If(ρθ))[1] / parent(If(ρ))[1]
    u_wind = sqrt(u_1^2 + v_1^2)

    # surface flux calculations 
    #calculate_flux(T_sfc, T1) = parameters.λ .* (T_sfc .- T1)
    #F_sfc = - calculate_flux( T_sfc[1], parent(ρθ)[1] / parent(ρ)[1] ) 
    F_sfc = - calculate_sfc_fluxes_energy(DryBulkFormulaWithRadiation(), parameters, T_sfc[1], θ_1 , u_1, v_1, ρ_1, t ) # W / m2

    dY.x[3] .= - F_sfc[1]

    # density (centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))

    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    @. dρ = gradf2c( -w * If(ρ) ) # Eq. 4.11

    # potential temperature (centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(F_sfc), top = Operators.SetValue(0.0)) 

    @. dρθ = gradf2c( -w * If(ρθ) + ν * gradc2f(ρθ/ρ) ) 
    # u velocity (centers)
    gradc2f = Operators.GradientC2F(top = Operators.SetValue(ug)) 
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(Cd * u_wind * u_1))
    
    A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    @. du = gradf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u) 

    # v velocity (centers)
    gradc2f = Operators.GradientC2F(top = Operators.SetValue(vg)) 
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(Cd * u_wind * v_1)) 

    A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    @. dv = gradf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v) 

    # w velocity (faces)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))

    B = Operators.SetBoundaryOperator(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    Π(ρθ) = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v) ## This should be R_d/ C_p (but needs stabilising...reported to original developer)
    @. dw = B( -(If(ρθ / ρ) * gradc2f(Π(ρθ))) - grav + gradc2f(ν * gradf2c(w)) - w * If(gradf2c(w))) 
    return dY

end