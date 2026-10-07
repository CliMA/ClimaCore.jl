# Vector-invariant horizontal tendency for the discontinuous-Galerkin (DG) form
# of the staggered nonhydrostatic model, selected by a state that carries
# velocity `uₕ` on a DG space (`MOMENTUM_FORM=vector_invariant`).
#
# Mass and energy are in flux form, so an interface numerical flux closes their
# budgets exactly as in `dg_tendency.jl`. The momentum equation is the
# vector-invariant one of the CG form,
#
#     ∂ₜuₕ = -(f + ω³) × uₕ - ω¹² × w - ∇ₕp / ρ - ∇ₕ(K + Φ),
#
# whose strong-form derivatives see only the inside of an element. Each is
# completed by a central face lift — the correction `(1/J) ∮ n̂ ⊗ (q* - q)` of
# the strong form toward a single-valued `q*` — and the velocity jumps are
# damped by a penalty at the fastest signal speed. Without the penalty nothing
# couples the velocity across a face dissipatively. The vertical terms, the `w`
# equation and the implicit split are shared with the flux form.

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
    return (;
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuv = similar(ᶜlocal_geometry, UV),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = dg_thermo_numflux),
        ᶜlift_p = similar(ᶜlocal_geometry, UV),
        ᶜlift_K = similar(ᶜlocal_geometry, UV),
        ᶜlift_ω³ = similar(ᶜlocal_geometry, FT),
        ᶜpenalty_u = similar(ᶜlocal_geometry, FT),
        ᶜpenalty_v = similar(ᶜlocal_geometry, FT),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
    )
end

# The mass-weighted face lift of `fn` over `args`, accumulated into `out`.
function dg_lift!(fn, out, args...)
    fill!(parent(out), zero(FT))
    Operators.add_lifting_flux_interior!(fn, out, args...)
    return out
end

function dg_vi_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜp, ᶜω³, ᶠω¹², ᶠu³, ᶜf) = p
    (; ᶜu, ᶜuv, ᶜλ, ᶜdYt, ᶜthermo_completion) = p
    (; ᶜlift_p, ᶜlift_K, ᶜlift_ω³, ᶜpenalty_u, ᶜpenalty_v) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜu = Geometry.UVWVector(ᶜuvw)
    @. ᶜuv = Geometry.UVVector(ᶜuₕ)
    @. ᶜK = norm_sqr(ᶜuvw) / 2
    thermo_pressure!(ᶜp, Y.c, ᶜK, ᶜΦ, p.moisture)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # Mass and energy: the weak-form flux divergence with a Rusanov interface
    # flux, then the vertical halves the CG form uses.
    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # The `w` equation, which also leaves `ᶠω¹²` for the cross term below. The
    # lifts are taken on the orthonormal components, which are single-valued
    # across panel edges where the covariant ones are not.
    dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    ᶜu_comp = ᶜuv.components.data.:1
    ᶜv_comp = ᶜuv.components.data.:2
    dg_lift!(Operators.central_curl3_lift, ᶜlift_ω³, ᶜu_comp, ᶜv_comp)
    @. ᶜω³ = curlₕ(ᶜuₕ) + CT3(Geometry.WVector(ᶜlift_ω³ / ᶜWJ))
    @. ᶠu³ = CT3(ᶠw)
    @. Yₜ.c.uₕ -= ᶜinterp(ᶠω¹² × ᶠu³) + (ᶜf + ᶜω³) × CT12(ᶜuₕ)

    # Pressure and kinetic-energy gradients; `Φ` is continuous and needs no
    # lift.
    dg_lift!(Operators.central_gradient_lift, ᶜlift_p, ᶜp)
    dg_lift!(Operators.central_gradient_lift, ᶜlift_K, ᶜK)
    @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    @. Yₜ.c.uₕ -= C12((ᶜlift_p / ᶜρ + ᶜlift_K) / ᶜWJ)

    # Velocity jump penalty at `λ = |u| + c`.
    dg_lift!(Operators.jump_penalty_lift, ᶜpenalty_u, ᶜu_comp, ᶜλ)
    dg_lift!(Operators.jump_penalty_lift, ᶜpenalty_v, ᶜv_comp, ᶜλ)
    @. Yₜ.c.uₕ += C12(Geometry.UVVector(ᶜpenalty_u, ᶜpenalty_v) / ᶜWJ)

    return Yₜ
end
