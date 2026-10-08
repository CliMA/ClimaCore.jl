# DG vector-invariant explicit tendency (`MOMENTUM_FORM=vector_invariant`):
# mass and energy in flux form, `∂ₜuₕ = -(f + ω³) × uₕ - ω¹² × w - ∇ₕp / ρ -
# ∇ₕ(K + Φ)`. Horizontal derivatives are weak-form operators completed by
# `Operators.complete_tendency!`; vertical terms and the implicit split are the
# CG form's.

function dg_cache(
    ::Grids.DG,
    ::VectorInvariantForm,
    ᶜlocal_geometry,
    ᶠlocal_geometry,
    ᶜf,
    Y,
)
    has_moisture(Y.c) &&
        error("the DG vector-invariant form does not transport water")
    UV = Geometry.UVVector{FT}
    UVW = Geometry.UVWVector{FT}
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdYt_uₕ = similar(
        ᶜlocal_geometry,
        NamedTuple{(:∇p, :∇K, :ω³, :u, :v), Tuple{UV, UV, FT, FT, FT}},
    )
    return (;
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuv = similar(ᶜlocal_geometry, UV),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = dg_thermo_numflux),
        ᶜdYt_uₕ,
        ᶜvi_completion =
        Operators.tendency_completion(ᶜdYt_uₕ; numflux = dg_vi_numflux),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
    )
end

# Central fluxes for the weak `∇ₕp`, `∇ₕK`, `ω³`, and a velocity-jump penalty
# at `λ = |u| + c`, on orthonormal components (single-valued at panel edges).
function dg_vi_numflux(normal, (p⁻, K⁻, u⁻, v⁻, λ⁻), (p⁺, K⁺, u⁺, v⁺, λ⁺))
    nu = normal.components.data.:1
    nv = normal.components.data.:2
    λ = max(λ⁻, λ⁺)
    return (;
        ∇p = -((p⁻ + p⁺) / 2) * normal,
        ∇K = -((K⁻ + K⁺) / 2) * normal,
        ω³ = -(nu * (v⁻ + v⁺) - nv * (u⁻ + u⁺)) / 2,
        u = λ / 2 * (u⁻ - u⁺),
        v = λ / 2 * (v⁻ - v⁺),
    )
end

function dg_vi_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu³, ᶜf) = p
    (; ᶜu, ᶜuv, ᶜλ, ᶜdYt, ᶜthermo_completion, ᶜdYt_uₕ, ᶜvi_completion) = p

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜu = Geometry.UVWVector(ᶜuvw)
    @. ᶜuv = Geometry.UVVector(ᶜuₕ)
    @. ᶜK = norm_sqr(ᶜuvw) / 2
    thermo_pressure!(ᶜp, Y.c, ᶜK, ᶜΦ, p.moisture)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # Mass and energy, then their vertical transport.
    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # Horizontal derivatives of the momentum equation.
    @. ᶜdYt_uₕ.∇p = Geometry.UVVector(wgradₕ(ᶜp))
    @. ᶜdYt_uₕ.∇K = Geometry.UVVector(wgradₕ(ᶜK))
    @. ᶜdYt_uₕ.ω³ = vertical_component(Geometry.WVector(wcurlₕ(ᶜuₕ)))
    fill!(parent(ᶜdYt_uₕ.u), zero(FT))
    fill!(parent(ᶜdYt_uₕ.v), zero(FT))
    ᶜu_comp = ᶜuv.components.data.:1
    ᶜv_comp = ᶜuv.components.data.:2
    Operators.complete_tendency!(
        ᶜvi_completion,
        ᶜdYt_uₕ,
        ᶜp,
        ᶜK,
        ᶜu_comp,
        ᶜv_comp,
        ᶜλ,
    )

    # `w` equation, which also leaves `ᶠω¹²` for the cross term.
    dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    @. ᶜω³ = CT3(Geometry.WVector(ᶜdYt_uₕ.ω³))
    @. ᶠu³ = CT3(ᶠw)
    @. Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × CT12(ᶜuₕ)
    # `Φ` is continuous: no face flux needed.
    @. Yₜ.c.uₕ -=
        C12(ᶜdYt_uₕ.∇p / ᶜρ + ᶜdYt_uₕ.∇K) + gradₕ(ᶜΦ)
    @. Yₜ.c.uₕ += C12(Geometry.UVVector(ᶜdYt_uₕ.u, ᶜdYt_uₕ.v))
    return Yₜ
end
