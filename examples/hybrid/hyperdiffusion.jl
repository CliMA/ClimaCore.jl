# Hyperdiffusion for the staggered nonhydrostatic model: a ∇⁴ operator applied
# to total energy and to horizontal momentum, built as two successive ∇² passes
# with a DSS in between. It is a numerical closure, present to remove grid-scale
# noise that the spectral element discretization does not damp on its own, not a
# physical mixing parameterization.
import LazyBroadcast: lazy

hyperdiffusion_cache(
    ᶜlocal_geometry;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
) = (;
    ᶜχ = similar(ᶜlocal_geometry, FT),
    ᶜχuₕ = similar(ᶜlocal_geometry, Geometry.Covariant12Vector{FT}),
    κ₄,
    divergence_damping_factor,
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume that ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor) = p

    # ∇⁴ is two ∇² passes with a DSS between them, so that the second pass
    # differentiates continuous intermediate fields. Both intermediates are
    # computed before a single weighted_dss! call, which batches the scalar
    # and vector exchanges into one communication phase. The Laplacian atoms
    # handle the horizontal-dimension distinction (no curl-curl term with one
    # horizontal dimension). On DG spaces weighted_dss! is a no-op and both
    # atoms include the face corrections themselves, so this tendency runs
    # unchanged on either discretization. The first pass writes through the
    # in-place forms, allocation-free on both; the second stays in the
    # returning form, which on CG is lazy and fuses into the tendency
    # broadcast (on DG it is a materialized field).
    ᶜh_tot = lazy.((Y.c.ρe .+ ᶜp) ./ ᶜρ)
    Operators.scalar_laplacian!(ᶜχ, ᶜh_tot)
    Operators.vector_laplacian!(ᶜχuₕ, ᶜuₕ)
    Spaces.weighted_dss!(ᶜχ => ghost_buffer.χ, ᶜχuₕ => ghost_buffer.χuₕ)

    Yₜ.c.ρe .-= κ₄ .* Operators.scalar_laplacian(ᶜχ; weight = ᶜρ)
    Yₜ.c.uₕ .-=
        κ₄ .* Operators.vector_laplacian(
            ᶜχuₕ;
            divergence_factor = divergence_damping_factor,
        )
end
