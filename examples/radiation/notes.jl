Given:
    The longwave optical depth τ = τ_lw(P), where τ_lw(0) = 0 and τ_lw(P_sfc) = τ_sfc.
    The shortwave optical depth τ_sw(τ), where τ_sw(0) = 0.
    The longwave source function S(τ) = σ T(τ)^4. The shortwave source function is 0.
    Some constant D = 3/2? Need a better explanation for where this value comes from.
Want to find:
    The fluxes F_up_lw(τ), F_dn_lw(τ), F_up_sw(τ), F_dn_sw(τ).
    The equilibrium temperature T_eq(τ).
From Schwarzschild's equation for radiative transfer:
    dF_up_lw(τ)/dτ = D (F_up_lw(τ) - S(τ))
    dF_dn_lw(τ)/dτ = -D (F_dn_lw(τ) - S(τ))
    dF_up_sw(τ)/dτ_sw(τ) = D F_up_sw(τ)
    dF_dn_sw(τ)/dτ_sw(τ) = -D F_dn_sw(τ)
Rearranging:
    dF_up_lw(τ)/dτ - D F_up_lw(τ) = -D S(τ)
    dF_dn_lw(τ)/dτ + D F_dn_lw(τ) = D S(τ)
    dF_up_sw(τ)/dτ - D dτ_sw(τ)/dτ F_up_sw(τ) = 0
    dF_dn_sw(τ)/dτ + D dτ_sw(τ)/dτ F_dn_sw(τ) = 0
General solutions:
    F_up_lw(τ) = exp(D τ) (-D ∫_0^τ S(τ′) exp(-D τ′) dτ′ + C_up_lw)
    F_dn_lw(τ) = exp(-D τ) (D ∫_0^τ S(τ′) exp(D τ′) dτ′ + C_dn_lw)
    F_up_sw(τ) = C_up_sw exp(D τ_sw(τ))
    F_dn_sw(τ) = C_dn_sw exp(-D τ_sw(τ))
Boundary conditions for three of the fluxes:
    No downward longwave flux at top ==>
    F_dn_lw(0) = 0 ==>
    C_dn_lw = 0
    Constant downward shortwave flux at top ==>
    F_dn_sw(0) = F_TOA ==>
    C_dn_sw = F_TOA
    Constant shortwave albedo at surface ==>
    F_up_sw(τ_sfc) = α F_dn_sw(τ_sfc) ==>
    C_up_sw exp(D τ_sw(τ_sfc)) = α F_TOA exp(-D τ_sw(τ_sfc)) ==>
    C_up_sw = α F_TOA exp(-2 D τ_sw(τ_sfc))
Substituting into general solutions:
    F_dn_lw(τ) = D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′
    F_up_sw(τ) = α F_TOA exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ)))
    F_dn_sw(τ) = F_TOA exp(-D τ_sw(τ))
Fixed surface temperature:
    Equilibrium condition:
        F_net(τ) = F_e ==>
        F_up_lw(τ) - F_dn_lw(τ) + F_up_sw(τ) - F_dn_sw(τ) = F_e ==>
        exp(D τ) (-D ∫_0^τ S(τ′) exp(-D τ′) dτ′ + C_up_lw) - D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ + α F_TOA exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) - F_TOA exp(-D τ_sw(τ)) = F_e ==>
        C_up_lw = exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (exp(-D τ_sw(τ)) - α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ)))) + F_e) ==>
    Temporarily assume τ_sw(τ) = 0; i.e., no shortwave absorption:
        C_up_lw = exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (1 - α) + F_e)
    C_up_lw is a constant:
        dC_up_lw/dτ = 0 ==>
        -D exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (1 - α) + F_e) + D exp(-D τ) (2 S(τ) + ∫_0^τ S(τ′) (D exp(-D (τ′ - τ)) - D exp(-D (τ - τ′))) dτ′) = 0 ==>
        D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = S(τ) - (F_TOA (1 - α) + F_e)/2
    Differentiate both sides with respect to τ:
        D S(τ) - D^2 ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = dS(τ)/dτ ==>
        D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = S(τ) - dS(τ)/dτ / D
    Substitute in previous equation (C_up_lw is a constant):
        S(τ) - (F_TOA (1 - α) + F_e)/2 = S(τ) - dS(τ)/dτ / D ==>
        dS(τ)/dτ = D (F_TOA (1 - α) + F_e)/2
    General solution:
        S(τ) = D (F_TOA (1 - α) + F_e)/2 τ + C_S


    Boundary condition for remaining flux:
        Constant upward longwave flux at surface ==>
        F_up_lw(τ_sfc) = S_g ==>
        exp(D τ_sfc) (-D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + C_up_lw) = S_g ==>
        C_up_lw = D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D τ_sfc)
    Substituting into general solution:
        F_up_lw(τ) = -D ∫_0^τ S(τ′) exp(D (τ - τ′)) dτ′ + D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D (τ_sfc - τ))
    Equilibrium condition:
        F_net(τ) = F_e ==>
        F_up_lw(τ) - F_dn_lw(τ) + F_up_sw(τ) - F_dn_sw(τ) = F_e ==>
        -D ∫_0^τ S(τ′) exp(D (τ - τ′)) dτ′ + D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D (τ_sfc - τ)) - D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ + α F_TOA exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) - F_TOA exp(-D τ_sw(τ)) = F_e ==>
        D ∫_0^τ S(τ′) (exp(D (τ - τ′)) + exp(-D (τ - τ′))) dτ′ = D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D (τ_sfc - τ)) - F_e - F_TOA (exp(-D τ_sw(τ)) - α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))))
    Temporarily assume τ_sw(τ) = 0; i.e., no shortwave absorption:
        D ∫_0^τ S(τ′) (exp(D (τ - τ′)) + exp(-D (τ - τ′))) dτ′ = D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D (τ_sfc - τ)) - F_e - F_TOA (1 - α)



    Boundary condition for remaining flux:
        Constant upward longwave flux at surface ==>
        F_up_lw(τ_sfc) = S_g ==>
        exp(D τ_sfc) (-D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + C_up_lw) = S_g ==>
        C_up_lw = D ∫_0^τ_sfc S(τ′) exp(-D τ′) dτ′ + S_g exp(-D τ_sfc)
    Substituting into general solution:
        F_up_lw(τ) = D ∫_τ^τ_sfc S(τ′) exp(-D (τ′ - τ)) dτ′ + S_g exp(-D (τ_sfc - τ))
    From equilibrium condition:
        dF_net(τ)/dτ = 0 ==>
        dF_up_lw(τ)/dτ - dF_dn_lw(τ)/dτ + dF_up_sw(τ)/dτ - dF_dn_sw(τ)/dτ = 0 ==>
        D (F_up_lw(τ) - S(τ)) + D (F_dn_lw(τ) - S(τ)) + D dτ_sw(τ)/dτ F_up_sw(τ) + D dτ_sw(τ)/dτ F_dn_sw(τ) = 0 ==>
        F_up_lw(τ) + F_dn_lw(τ) + dτ_sw(τ)/dτ (F_up_sw(τ) + F_dn_sw(τ)) - 2 S(τ) = 0 ==>
        D ∫_τ^τ_sfc S(τ′) exp(-D (τ′ - τ)) dτ′ + S_g exp(-D (τ_sfc - τ)) + D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ + dτ_sw(τ)/dτ F_TOA (α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) + exp(-D τ_sw(τ))) - 2 S(τ) = 0 ==>
        S(τ) = D/2 (∫_τ^τ_sfc S(τ′) exp(-D (τ′ - τ)) dτ′ + ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′) + S_g/2 exp(-D (τ_sfc - τ)) + dτ_sw(τ)/dτ F_TOA/2 (α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) + exp(-D τ_sw(τ)))
    Definition:
        F(τ) = S_g/2 exp(-D (τ_sfc - τ)) + dτ_sw(τ)/dτ F_TOA/2 (α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) + exp(-D τ_sw(τ)))
    Substituting into equilibrium condition:
        S(τ) = D/2 (∫_τ^τ_sfc S(τ′) exp(-D (τ′ - τ)) dτ′ + ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′) + F(τ)
Variable surface temperature:
    Equilibrium condition:
        F_net(τ) = 0 ==>
        F_up_lw(τ) - F_dn_lw(τ) + F_up_sw(τ) - F_dn_sw(τ) = 0 ==>
        exp(D τ) (-D ∫_0^τ S(τ′) exp(-D τ′) dτ′ + C_up_lw) - D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ + α F_TOA exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))) - F_TOA exp(-D τ_sw(τ)) = 0 ==>
        C_up_lw = exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (exp(-D τ_sw(τ)) - α exp(-D (2 τ_sw(τ_sfc) - τ_sw(τ))))) ==>
    Temporarily assume τ_sw(τ) = 0; i.e., no shortwave absorption:
        C_up_lw = exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (1 - α))
    C_up_lw is a constant:
        dC_up_lw/dτ = 0 ==>
        -D exp(-D τ) (D ∫_0^τ S(τ′) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (1 - α)) + D exp(-D τ) (2 S(τ) + ∫_0^τ S(τ′) (D exp(-D (τ′ - τ)) - D exp(-D (τ - τ′))) dτ′) = 0 ==>
        D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = S(τ) - F_TOA/2 (1 - α)
    Differentiate both sides with respect to τ:
        D S(τ) - D^2 ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = dS(τ)/dτ ==>
        D ∫_0^τ S(τ′) exp(-D (τ - τ′)) dτ′ = S(τ) - dS(τ)/dτ / D
    Substitute in previous equation (C_up_lw is a constant):
        S(τ) - F_TOA/2 (1 - α) = S(τ) - dS(τ)/dτ / D ==>
        dS(τ)/dτ = D F_TOA/2 (1 - α)
    General solution:
        S(τ) = D F_TOA/2 (1 - α) τ + C_S
    Substitute into previous equation (C_up_lw is a constant):
        D ∫_0^τ (D F_TOA/2 (1 - α) τ′ + C_S) exp(-D (τ - τ′)) dτ′ = D F_TOA/2 (1 - α) τ + C_S - F_TOA/2 (1 - α) ==>
        D^2 F_TOA/2 (1 - α) ∫_0^τ τ′ exp(-D (τ - τ′)) dτ′ + C_S D ∫_0^τ exp(-D (τ - τ′)) dτ′ = F_TOA/2 (1 - α) (D τ - 1) + C_S ==>
        F_TOA/2 (1 - α) (D τ + exp(-D τ) - 1) + C_S (1 - exp(-D τ)) = F_TOA/2 (1 - α) (D τ - 1) + C_S ==>
        C_S = F_TOA/2 (1 - α)
    Substitute into general solution:
        S(τ) = F_TOA/2 (1 - α) (D τ + 1)
    Substitute into formula for C_up_lw:
        C_up_lw = exp(-D τ) (D F_TOA/2 (1 - α) ∫_0^τ (D τ′ + 1) (exp(-D (τ′ - τ)) + exp(-D (τ - τ′))) dτ′ + F_TOA (1 - α)) ==>
        C_up_lw = exp(-D τ) (F_TOA (1 - α) (exp(D τ) - 1) + F_TOA (1 - α)) ==>
        C_up_lw = F_TOA (1 - α)
    Substitute formula for S(τ) and C_up_lw into formula for F_up_lw(τ):
        F_up_lw(τ) = exp(D τ) (-D F_TOA/2 (1 - α) ∫_0^τ (D τ′ + 1) exp(-D τ′) dτ′ + F_TOA (1 - α)) ==>
        F_up_lw(τ) = exp(D τ) (F_TOA (1 - α) (exp(-D τ) (D/2 τ + 1) - 1) + F_TOA (1 - α)) ==>
        F_up_lw(τ) = F_TOA/2 (1 - α) (D τ + 2)
    Find upward longwave flux from the ground:
        S_g = F_up_lw(τ_sfc) = F_TOA/2 (1 - α) (D τ_sfc + 2)

    for
