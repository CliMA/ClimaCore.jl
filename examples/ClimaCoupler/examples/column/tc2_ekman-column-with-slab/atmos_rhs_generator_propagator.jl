# Atmos RHS

function ∑tendencies_atm!(dY, Y, (parameters, T_sfc), t)
    # Heat diffusion:
    # ∂_t ρ =  ∇ (μ ∇ ρ - w ρ) 
    # ∂_t ρθ =  ∇ (μ ∇ ρθ - w ρθ) 
    # ∂_t w =  ∇ (μ ∇ w - w w) 

    # where
    # ∂_t T = n \cdot F   at z = zmin_atm
    # ∂_t T = 0           at z = zmax_atm
    # We also use this model to accumulate fluxes
    # ∂_t ϕ_bottom = n \cdot F

    # μ = FT(0.0001) # diffusion coefficient

    # T = u.x[1]

    # F_sfc = - calculate_flux( T_sfc[1], parent(T)[1] )
    
    # # set BCs
    # bcs_bottom = Operators.SetValue(F_sfc) # struct w bottom BCs
    # bcs_top = Operators.SetValue(FT(280.0))

    # gradc2f = Operators.GradientC2F(top = bcs_top) # gradient struct w BCs
    # gradf2c = Operators.GradientF2C(bottom = bcs_bottom)
    # @. du.x[1] = gradf2c( μ * gradc2f(T))
    # du.x[2] .= - F_sfc[1]

    UnPack.@unpack Cd, f, ν, ug, vg, C_p, MSLP, R_d, R_m, C_v, grav = parameters

    (Yc, Yf, F_sfc) = Y.x
    (dYc, dYf, dF_sfc) = dY.x
    UnPack.@unpack ρ, u, v, ρθ = Yc
    UnPack.@unpack w = Yf
    dρ = dYc.ρ
    du = dYc.u
    dv = dYc.v
    dρθ = dYc.ρθ
    dw = dYf.w

    # auxiliary calculations
    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    u_wind = sqrt(u_1^2 + v_1^2)

    # density (centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))

    If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
    @. dρ = gradf2c( -w * If(ρ) ) # Eq. 4.11

    # potential temperature (centers)
    gradc2f = Operators.GradientC2F()
    gradf2c = Operators.GradientF2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0)) # Eq. 4.20, 4.21

    @. dρθ = gradf2c( -w * If(ρθ) + ν * gradc2f(ρθ/ρ) ) # Eq. 4.12

    # u velocity (centers)
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
    Π(ρθ) = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v)
    #dw = ρθ .* 0.0
    @. dw = B( -(If(ρθ / ρ) * gradc2f(Π(ρθ))) - grav + gradc2f(ν * gradf2c(w)) - w * If(gradf2c(w))) # Eq. 4.10

    return dY





    
end