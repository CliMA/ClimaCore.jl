# DG vector-invariant explicit tendency (`MOMENTUM_FORM=vector_invariant`):
# mass and energy in split flux form, and `∂ₜuₕ = -(f + ω³) × uₕ - ω¹² × w -
# ∇ₕ(K + Φ - Φ_r) - c_pd/2 [θ′ ∇ₕΠ + ∇ₕ(θ′ Π) - Π ∇ₕθ′]`, the split Exner
# pressure gradient. Mass and energy use the two-point form of
# `split_divₕ(ρu, h_tot)` with Rusanov interface penalties, and the velocity
# jumps are penalized at a ρ-weighted rate.

function dg_cache(
    ::Grids.DG,
    ::VectorInvariantForm,
    ᶜlocal_geometry,
    ᶠlocal_geometry,
    ᶜf,
    Y,
)
    UV = Geometry.UVVector{FT}
    UVW = Geometry.UVWVector{FT}
    ᶜdYt_uₕ = similar(
        ᶜlocal_geometry,
        NamedTuple{(:∇KΦ, :∇Π, :∇θΠ, :∇θ, :ω³), Tuple{UV, UV, UV, UV, FT}},
    )
    ᶜscalars = similar(
        ᶜlocal_geometry,
        NamedTuple{(:KΦ, :Π, :θ, :θΠ), NTuple{4, FT}},
    )
    return (;
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuv = similar(ᶜlocal_geometry, UV),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        ᶜvi_state = similar(
            ᶜlocal_geometry,
            NamedTuple{(:ρ, :ρe, :e, :p, :λ, :uv), Tuple{FT, FT, FT, FT, FT, UV}},
        ),
        ᶜvi_residual = similar(
            ᶜlocal_geometry,
            NamedTuple{(:ρ, :ρe), Tuple{FT, FT}},
        ),
        ᶜdYt_uₕ,
        ᶜscalars,
        ᶜvi_completion =
        Operators.tendency_completion(ᶜdYt_uₕ; numflux = dg_vi_numflux),
        ᶜpenalty = similar(ᶜlocal_geometry, NamedTuple{(:u, :v), NTuple{2, FT}}),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
    )
end

# Two-point flux for `(ρ, ρe)`: `F_ρ = {ρũ}`, the average of the nodal
# contravariant mass fluxes, and `F_ρe = {ρũ}{h_tot}`.
function dg_vi_scalars_flux(nvec_a, nvec_b, y_a, y_b)
    Fρ = (y_a.ρ * (y_a.uv' * nvec_a) + y_b.ρ * (y_b.uv' * nvec_b)) / 2
    h̄ = (y_a.e + y_a.p / y_a.ρ + y_b.e + y_b.p / y_b.ρ) / 2
    return (ρ = Fρ, ρe = Fρ * h̄)
end

# Interface flux for `dg_vi_scalars_flux`, with Rusanov penalties on `[[ρ]]`
# and `[[ρe]]`.
function dg_vi_scalars_numflux(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = dg_vi_scalars_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
    )
end

# Velocity-jump penalty at rate `max(λ)/2 {ρ}/ρ⁻`, so that the face kinetic
# energy tendency is `-max(λ)/2 {ρ} [[u]]² ≤ 0`; a lifting, not a flux.
rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻), (q⁺, ρ⁺, λ⁺)) =
    max(λ⁻, λ⁺) / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * (q⁺ - q⁻)

# Central fluxes for the weak gradients and `ω³`.
function dg_vi_numflux(normal, (s⁻, u⁻, v⁻), (s⁺, u⁺, v⁺))
    nu = normal.components.data.:1
    nv = normal.components.data.:2
    central(a, b) = -((a + b) / 2) * normal
    return (;
        ∇KΦ = central(s⁻.KΦ, s⁺.KΦ),
        ∇Π = central(s⁻.Π, s⁺.Π),
        ∇θΠ = central(s⁻.θΠ, s⁺.θΠ),
        ∇θ = central(s⁻.θ, s⁺.θ),
        ω³ = -(nu * (v⁻ + v⁺) - nv * (u⁻ + u⁺)) / 2,
    )
end

function default_remaining_tendency!(
    ::Grids.DG,
    ::VectorInvariantForm,
    Yₜ,
    Y,
    p,
    t,
)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu³, ᶜf) = p
    (; ᶜu, ᶜuv, ᶜλ, ᶜvi_state, ᶜvi_residual) = p
    (; ᶜdYt_uₕ, ᶜscalars, ᶜvi_completion, ᶜpenalty) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜu = Geometry.UVWVector(ᶜuvw)
    @. ᶜuv = Geometry.UVVector(ᶜuₕ)
    @. ᶜK = norm_sqr(ᶜuvw) / 2
    @. ᶜp = pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # Mass and energy, then their vertical transport.
    @. ᶜvi_state.ρ = ᶜρ
    @. ᶜvi_state.ρe = ᶜρe
    @. ᶜvi_state.e = ᶜρe / ᶜρ
    @. ᶜvi_state.p = ᶜp
    @. ᶜvi_state.λ = ᶜλ
    @. ᶜvi_state.uv = Geometry.project(Geometry.UVAxis(), ᶜu)
    fill!(parent(ᶜvi_residual), zero(FT))
    Operators.add_flux_differencing_divergence!(
        dg_vi_scalars_flux,
        ᶜvi_residual,
        ᶜvi_state,
    )
    Operators.add_numerical_flux_interior!(
        dg_vi_scalars_numflux,
        ᶜvi_residual,
        ᶜvi_state,
    )
    @. Yₜ.c.ρ += ᶜvi_residual.ρ / ᶜWJ
    @. Yₜ.c.ρe += ᶜvi_residual.ρe / ᶜWJ
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # Horizontal derivatives of the momentum equation.
    @. ᶜscalars.KΦ = ᶜK + ᶜΦ - phi_ref(ᶜp)
    @. ᶜscalars.Π = exner(ᶜp)
    @. ᶜscalars.θ = theta_prime(ᶜρ, ᶜp)
    @. ᶜscalars.θΠ = ᶜscalars.θ * ᶜscalars.Π
    @. ᶜdYt_uₕ.∇KΦ = Geometry.UVVector(wgradₕ(ᶜscalars.KΦ))
    @. ᶜdYt_uₕ.∇Π = Geometry.UVVector(wgradₕ(ᶜscalars.Π))
    @. ᶜdYt_uₕ.∇θΠ = Geometry.UVVector(wgradₕ(ᶜscalars.θΠ))
    @. ᶜdYt_uₕ.∇θ = Geometry.UVVector(wgradₕ(ᶜscalars.θ))
    @. ᶜdYt_uₕ.ω³ = vertical_component(Geometry.WVector(wcurlₕ(ᶜuₕ)))
    ᶜu_comp = ᶜuv.components.data.:1
    ᶜv_comp = ᶜuv.components.data.:2
    Operators.complete_tendency!(
        ᶜvi_completion,
        ᶜdYt_uₕ,
        ᶜscalars,
        ᶜu_comp,
        ᶜv_comp,
    )
    fill!(parent(ᶜpenalty), zero(FT))
    Operators.add_lifting_flux_interior!(
        rho_weighted_jump_penalty_lift,
        ᶜpenalty.u,
        ᶜu_comp,
        ᶜρ,
        ᶜλ,
    )
    Operators.add_lifting_flux_interior!(
        rho_weighted_jump_penalty_lift,
        ᶜpenalty.v,
        ᶜv_comp,
        ᶜρ,
        ᶜλ,
    )

    # `w` equation, which also leaves `ᶠω¹²` for the cross term.
    dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    @. ᶜω³ = CT3(Geometry.WVector(ᶜdYt_uₕ.ω³))
    @. ᶠu³ = CT3(ᶠw)
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹² × (ᶠinterp(ᶜρ * ᶜJ) * ᶠu³)) / (ᶜρ * ᶜJ) +
        (ᶜf + ᶜω³) × CT12(ᶜuₕ)
    # Split Exner pressure gradient.
    @. Yₜ.c.uₕ -= C12(
        ᶜdYt_uₕ.∇KΦ +
        cp_d * (
            ᶜscalars.θ * ᶜdYt_uₕ.∇Π + ᶜdYt_uₕ.∇θΠ - ᶜscalars.Π * ᶜdYt_uₕ.∇θ
        ) / 2,
    )
    @. Yₜ.c.uₕ += C12(Geometry.UVVector(ᶜpenalty.u, ᶜpenalty.v) / ᶜWJ)
    return Yₜ
end
