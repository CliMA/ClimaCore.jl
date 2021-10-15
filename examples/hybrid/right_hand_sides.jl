const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

# P = ρ * R_d * T = ρ * R_d * θ * (P / MSLP)^(R_d / C_p) ==>
# (P / MSLP)^(1 - R_d / C_p) = R_d * ρθ / MSLP ==>
# P = MSLP * (R_d * ρθ / MSLP)^γ
pressure_ρθ(ρθ) = MSLP * (R_d * ρθ / MSLP)^γ
# P = ρ * R_d * T = ρ * R_d * (ρe_int / ρ / C_v) = (γ - 1) * ρe_int
pressure_ρe_int(ρe_int) = (γ - 1) * ρe_int

# spectral horizontal operators
const hdiv = Operators.Divergence()

# vertical FD operators with BC's
const vdivc2f = Operators.DivergenceC2F(
    bottom = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
    top = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
)
const uvdivf2c = Operators.DivergenceF2C(
    bottom = Operators.SetValue(
        Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
    ),
    top = Operators.SetValue(
        Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
    ),
)
const If_bc = Operators.InterpolateC2F(
    bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
)
const If = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const Ic = Operators.InterpolateF2C()
const ∂ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
)
const ∂f = Operators.GradientC2F()
const B = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
)

function rhs!(dY, Y, p, t)
    @unpack uₕ, uₕf, P, Φ, ∂Φf = p

    # momentum and velocity variables
    @. uₕ = Y.Yc.ρuₕ / Y.Yc.ρ
    @. uₕf = If_bc(uₕ)
    if :ρw in propertynames(Y)
        ρw = Y.ρw
        w = p.w
        @. w = ρw / If(Y.Yc.ρ)
    elseif :w in propertynames(Y)
        w = Y.w
        ρw = p.ρw
        @. ρw = w * If(Y.Yc.ρ)
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y))"))
    end

    # ∂ρ/∂t = -∇⋅ρu
    @. dY.Yc.ρ = -hdiv(Y.Yc.ρuₕ)
    @. dY.Yc.ρ -= ∂(ρw)

    # ∂ρθ/∂t = -∇⋅ρuθ
    # ∂ρe/∂t = -∇⋅(ρue + uP)
    if :ρθ in propertynames(Y.Yc)
        @. P = pressure_ρθ(Y.Yc.ρθ)
        @. dY.Yc.ρθ = -hdiv(uₕ * Y.Yc.ρθ)
        @. dY.Yc.ρθ -= ∂(w * If(Y.Yc.ρθ))
    elseif :ρe_tot in propertynames(Y.Yc)
        @. P = pressure_ρe_int(
            Y.Yc.ρe_tot -
            Y.Yc.ρ * (
                Φ +
                0.5 * norm(
                    Geometry.transform(Geometry.Cartesian13Axis(), uₕ) +
                    Geometry.transform(Geometry.Cartesian13Axis(), Ic(w))
                )^2
            )
        )
        @. dY.Yc.ρe_int = -hdiv(uₕ * (Y.Yc.ρe_tot + P))
        @. dY.Yc.ρe_int -= ∂(w * If(Y.Yc.ρe_tot + P))
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y.Yc))"))
    end

    # ∂ρu/∂t = -∇P - ρ∇Φ - ∇⋅(ρu ⊗ u)
    # ∂u/∂t = -(∇P)/ρ - ∇Φ - u⋅∇u
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
            @SMatrix [1.]
        ),
    )
    @. dY.Yc.ρuₕ = -hdiv(P * Ih + Y.Yc.ρuₕ ⊗ uₕ)
    @. dY.Yc.ρuₕ -= uvdivf2c(ρw ⊗ uₕf)
    if :ρw in propertynames(Y)
        @. dY.ρw = B(
            Geometry.transform(
                Geometry.Cartesian3Axis(),
                -∂f(P) - If(Y.Yc.ρ) * ∂Φf
            ) -
            vvdivc2f(Ic(ρw ⊗ w)),
        )
        @. dY.ρw -= hdiv(uₕf ⊗ ρw)
    elseif :w in propertynames(Y)
        @. dY.w = B(
            Geometry.transform(
                Geometry.Cartesian3Axis(),
                -∂f(P) / If(Y.Yc.ρ) - ∂Φf
            ) -
            If(Ic(w) * ∂(w)),
        )
        @. dY.w -= Geometry.transform(Geometry.Cartesian13Axis(), uₕf) * hdiv(w)
    end

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.ρw)
    return dY
end

function rhs_implicit!(dY, Y, p, t)
    @unpack P, Φ, ∂Φf = p

    # momentum and velocity variables
    if :ρw in propertynames(Y)
        ρw = Y.ρw
        w = p.w
        @. w = ρw / If(Y.Yc.ρ)
    elseif :w in propertynames(Y)
        w = Y.w
        ρw = p.ρw
        @. ρw = w * If(Y.Yc.ρ)
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y))"))
    end

    # ∂ρ/∂t ≈ -∂ρw/∂z
    @. dY.Yc.ρ = -∂(ρw)

    # ∂ρθ/∂t ≈ -∂ρwθ/∂z
    # ∂ρe/∂t ≈ -∂(ρwe + wP)/∂z
    if :ρθ in propertynames(Y.Yc)
        @. P = pressure_ρθ(Y.Yc.ρθ)
        @. dY.Yc.ρθ -= ∂(w * If(Y.Yc.ρθ))
    elseif :ρe_tot in propertynames(Y.Yc)
        @. P = pressure_ρe_int(
            Y.Yc.ρe_tot -
            Y.Yc.ρ * (
                Φ +
                0.5 * norm(
                    Geometry.transform(
                        Geometry.Cartesian13Axis(),
                        Y.Yc.ρuₕ / Y.Yc.ρ
                    ) +
                    Geometry.transform(Geometry.Cartesian13Axis(), Ic(w))
                )^2
            )
        )
        @. dY.Yc.ρe_int -= ∂(w * If(Y.Yc.ρe_tot + P))
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y.Yc))"))
    end

    # ∂ρu/∂t ≈ -∇P - ρ∇Φ
    # ∂u/∂t ≈ -(∇P)/ρ - ∇Φ
    @. dY.Yc.ρuₕ *= 0.
    if :ρw in propertynames(Y)
        @. dY.ρw = B(
            Geometry.transform(
                Geometry.Cartesian3Axis(),
                -∂f(P) - If(Y.Yc.ρ) * ∂Φf
            ),
        )
    elseif :w in propertynames(Y)
        @. dY.w = B(
            Geometry.transform(
                Geometry.Cartesian3Axis(),
                -∂f(P) / If(Y.Yc.ρ) - ∂Φf
            ),
        )
    end

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.ρw)
    return dY
end

# Sets dY to the value of rhs! - rhs_implicit!.
function rhs_remainder!(dY, Y, p, t)
    @unpack uₕ, uₕf, P, Φ, ∂Φf = p

    # momentum and velocity variables
    @. uₕ = Y.Yc.ρuₕ / Y.Yc.ρ
    @. uₕf = If_bc(uₕ)
    if :ρw in propertynames(Y)
        ρw = Y.ρw
        w = p.w
        @. w = ρw / If(Y.Yc.ρ)
    elseif :w in propertynames(Y)
        w = Y.w
        ρw = p.ρw
        @. ρw = w * If(Y.Yc.ρ)
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y))"))
    end

    # ∂ρ/∂t Remainder = -∇⋅ρuₕ
    @. dY.Yc.ρ = -hdiv(Y.Yc.ρuₕ)

    # ∂ρθ/∂t Remainder = -∇⋅ρuₕθ
    # ∂ρe/∂t Remainder = -∇⋅(ρuₕe + uₕP)
    if :ρθ in propertynames(Y.Yc)
        @. P = pressure_ρθ(Y.Yc.ρθ)
        @. dY.Yc.ρθ = -hdiv(uₕ * Y.Yc.ρθ)
    elseif :ρe_tot in propertynames(Y.Yc)
        @. P = pressure_ρe_int(
            Y.Yc.ρe_tot -
            Y.Yc.ρ * (
                Φ +
                0.5 * norm(
                    Geometry.transform(Geometry.Cartesian13Axis(), uₕ) +
                    Geometry.transform(Geometry.Cartesian13Axis(), Ic(w))
                )^2
            )
        )
        @. dY.Yc.ρe_int = -hdiv(uₕ * (Y.Yc.ρe_tot + P))
    else
        throw(ArgumentError("No rhs available for vars $(propertynames(Y.Yc))"))
    end

    # ∂ρu/∂t Remainder = -∇⋅(ρu ⊗ u)
    # ∂u/∂t Remainder = -u⋅∇u
    Ih = Ref(
        Geometry.Axis2Tensor(
            (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
            @SMatrix [1.]
        ),
    )
    @. dY.Yc.ρuₕ = -hdiv(P * Ih + Y.Yc.ρuₕ ⊗ uₕ)
    @. dY.Yc.ρuₕ -= uvdivf2c(ρw ⊗ uₕf)
    if :ρw in propertynames(Y)
        @. dY.ρw = B(-vvdivc2f(Ic(ρw ⊗ w)))
        @. dY.ρw -= hdiv(uₕf ⊗ ρw)
    elseif :w in propertynames(Y)
        @. dY.w = B(-If(Ic(w) * ∂(w)))
        @. dY.w -= Geometry.transform(Geometry.Cartesian13Axis(), uₕf) * hdiv(w)
    end

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.ρw)
    return dY
end