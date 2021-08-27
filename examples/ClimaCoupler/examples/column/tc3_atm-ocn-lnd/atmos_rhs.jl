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

    UnPack.@unpack Ch, Cd, f, ν, ug, vg, C_p, MSLP, R_d, R_m, C_v, grav = parameters

    #@show(t, Y.x[3]) 
    (Yc, Yf, F_sfc) = Y.x
    (dYc, dYf, dF_sfc) = dY.x

    UnPack.@unpack ρ, u, v, ρθ = Yc
    UnPack.@unpack w = Yf
    dρ = dYc.ρ
    du = dYc.u
    dv = dYc.v
    dρθ = dYc.ρθ
    dw = dYf.w

    # Auxiliary calculations
    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    ρ_1 = parent(ρ)[1]
    ρθ_1 = parent(ρθ)[1]
    u_wind = sqrt(u_1^2 + v_1^2)

    # surface flux calculations 
    surface_flux_ρθ = - calculate_sfc_fluxes_energy(DryBulkFormulaWithRadiation(), parameters, T_sfc[1], parent(ρθ)[1] / parent(ρ)[1] , u_1, v_1, ρ_1, t ) ./ C_p
    surface_flux_u =  - Cd * u_1 * sqrt(u_1^2 + v_1^2)
    surface_flux_v =  - Cd * v_1 * sqrt(u_1^2 + v_1^2)

    # accumulate in the required right units
    @inbounds begin
        dY.x[3][1] = - ρ_1 * surface_flux_u  # 
        dY.x[3][2] = - ρ_1 * surface_flux_v  # 
        dY.x[3][3] = - C_p * surface_flux_ρθ # W / m^2
    end

    # @inbounds begin
    #     dY.x[3][1] = - 10.0 
    #     dY.x[3][2] = - 1.0  
    #     dY.x[3][3] = - 1.0 
    # end

    # Density tendency (located at cell centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))

    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    @. dρ = gradf2c( -w * If(ρ) ) # Eq. 4.11

    # Potential temperature tendency (located at cell centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(surface_flux_ρθ), top = Operators.SetValue(0.0)) # Eq. 4.20, 4.21

    @. dρθ = gradf2c( -w * If(ρθ) + ν * gradc2f(ρθ/ρ) ) # Eq. 4.12

    # u velocity tendency (located at cell centers)
    gradc2f = Operators.GradientC2F(top = Operators.SetValue(ug)) # Eq. 4.18
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(Cd * u_wind * u_1)) # Eq. 4.16
    
    A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    @. du = gradf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

    # v velocity (centers)
    gradc2f = Operators.GradientC2F(top = Operators.SetValue(vg)) # Eq. 4.18
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(Cd * u_wind * v_1)) # Eq. 4.16

    A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    @. dv = gradf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v) # Eq. 4.9

    # w velocity (faces)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))

    B = Operators.SetBoundaryOperator(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    println(R_d ./ C_p)
    println(ρθ)
    println(t)
    Π(ρθ) = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v)
    
    @. dw = B( -(If(ρθ / ρ) * gradc2f(Π(ρθ))) - grav + gradc2f(ν * gradf2c(w)) - w * If(gradf2c(w))) # Eq. 4.10 # this makes everything unstable... use new ClimaAtmos rhs!
    
    return dY
end