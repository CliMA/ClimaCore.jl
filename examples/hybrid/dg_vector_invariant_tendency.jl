# DG vector-invariant explicit tendency (`MOMENTUM_FORM=vector_invariant`):
# mass and energy in flux form, and `∂ₜuₕ = -(f + ω³) × uₕ - ω¹² × w -
# ∇ₕ(K + Φ - Φ_r) - c_pd/2 [θ′ ∇ₕΠ + ∇ₕ(θ′ Π) - Π ∇ₕθ′]`, the split Exner
# pressure gradient. Horizontal derivatives are weak-form
# operators completed by `Operators.complete_tendency!`.

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
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdYt_uₕ = similar(
        ᶜlocal_geometry,
        NamedTuple{
            (:∇KΦ, :∇Π, :∇θΠ, :∇θ, :ω³, :u, :v),
            Tuple{UV, UV, UV, UV, FT, FT, FT},
        },
    )
    ᶜscalars = similar(
        ᶜlocal_geometry,
        NamedTuple{(:KΦ, :Π, :θ, :θΠ), NTuple{4, FT}},
    )
    return (;
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuv = similar(ᶜlocal_geometry, UV),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = dg_thermo_numflux),
        ᶜdYt_uₕ,
        ᶜscalars,
        ᶜvi_completion =
        Operators.tendency_completion(ᶜdYt_uₕ; numflux = dg_vi_numflux),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
    )
end

# Central fluxes for the weak gradients and `ω³`, and a velocity-jump penalty
# at `λ = |u| + c` on orthonormal components.
function dg_vi_numflux(normal, (s⁻, u⁻, v⁻, λ⁻), (s⁺, u⁺, v⁺, λ⁺))
    nu = normal.components.data.:1
    nv = normal.components.data.:2
    λ = max(λ⁻, λ⁺)
    central(a, b) = -((a + b) / 2) * normal
    return (;
        ∇KΦ = central(s⁻.KΦ, s⁺.KΦ),
        ∇Π = central(s⁻.Π, s⁺.Π),
        ∇θΠ = central(s⁻.θΠ, s⁺.θΠ),
        ∇θ = central(s⁻.θ, s⁺.θ),
        ω³ = -(nu * (v⁻ + v⁺) - nv * (u⁻ + u⁺)) / 2,
        u = λ / 2 * (u⁻ - u⁺),
        v = λ / 2 * (v⁻ - v⁺),
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
    (; ᶜu, ᶜuv, ᶜλ, ᶜdYt, ᶜthermo_completion) = p
    (; ᶜdYt_uₕ, ᶜscalars, ᶜvi_completion) = p

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜu = Geometry.UVWVector(ᶜuvw)
    @. ᶜuv = Geometry.UVVector(ᶜuₕ)
    @. ᶜK = norm_sqr(ᶜuvw) / 2
    @. ᶜp = pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # Mass and energy, then their vertical transport.
    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe
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
    fill!(parent(ᶜdYt_uₕ.u), zero(FT))
    fill!(parent(ᶜdYt_uₕ.v), zero(FT))
    ᶜu_comp = ᶜuv.components.data.:1
    ᶜv_comp = ᶜuv.components.data.:2
    Operators.complete_tendency!(
        ᶜvi_completion,
        ᶜdYt_uₕ,
        ᶜscalars,
        ᶜu_comp,
        ᶜv_comp,
        ᶜλ,
    )

    # `w` equation, which also leaves `ᶠω¹²` for the cross term.
    dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    @. ᶜω³ = CT3(Geometry.WVector(ᶜdYt_uₕ.ω³))
    @. ᶠu³ = CT3(ᶠw)
    @. Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × CT12(ᶜuₕ)
    # Split Exner pressure gradient.
    @. Yₜ.c.uₕ -= C12(
        ᶜdYt_uₕ.∇KΦ +
        cp_d * (
            ᶜscalars.θ * ᶜdYt_uₕ.∇Π + ᶜdYt_uₕ.∇θΠ - ᶜscalars.Π * ᶜdYt_uₕ.∇θ
        ) / 2,
    )
    @. Yₜ.c.uₕ += C12(Geometry.UVVector(ᶜdYt_uₕ.u, ᶜdYt_uₕ.v))
    return Yₜ
end
