# Flux-form horizontal tendency for the discontinuous-Galerkin (DG) form of the
# staggered nonhydrostatic model.
#
# A DG element sees its neighbours only through a numerical flux, and a
# numerical flux is a flux of a conserved quantity, so the DG form carries
# horizontal momentum `ρuₕ` and writes its equation in flux form. The CG form
# carries velocity `uₕ` in the vector-invariant form, which needs no interface
# flux because DSS makes the field continuous.
#
# What the two forms share: the vertical finite differences, the implicit
# split, the Jacobian, the initial condition, and the vertical momentum
# equation, which stays vector-invariant here (a face lift supplies the
# element coupling its one horizontal derivative needs).
#
# On a curved space `∇·(ρu⊗u)` carries Christoffel terms that a spectral
# divergence omits; `Operators.cartesian_tensor_divergence!` removes them by
# rotating the momentum axis into the global Cartesian basis, where the
# connection vanishes, and rotates the result back. The vertical flux
# divergence drops the same terms and is rotated the same way.

import LinearAlgebra
import ClimaCore: Fields, Geometry, Grids, Operators, Spaces
import ClimaCore.Geometry: ⊗

##
## Physical fluxes
##

# One definition of each flux feeds both the element-local weak-form volume
# term and the interface numerical flux, which is what makes the two mutually
# consistent. `u` is the full 3D velocity in the local orthonormal basis: the
# horizontal divergence contracts its first two contravariant components, and
# the vertical one its third.
dg_thermo_flux(ρ, ρe, u, pres) = (; ρ = ρ * u, ρe = (ρe + pres) * u)

# Momentum flux `ρu⊗u + p𝟙`, transport axis first. Carrying the pressure in
# the flux keeps the horizontal pressure gradient conservative, which is what
# lets the interface flux close the momentum budget across an element face.
dg_momentum_flux(ρ, u, pres) = (ρ * u) ⊗ u + pres * LinearAlgebra.I

# The same without pressure, for the vertical flux divergence: the vertical
# pressure gradient belongs to the `w` equation, which the implicit solver
# owns. On a grid with topography that leaves out the part of the horizontal
# pressure gradient that the vertical coordinate carries, the same gap the CG
# form has in its `gradₕ(ᶜp)`.
dg_momentum_transport(ρ, u) = (ρ * u) ⊗ u

# Fastest signal speed, sound plus advection, setting the size of the
# interface penalty below.
dg_wavespeed(ρ, u, pres) = sqrt(γ * pres / ρ) + norm(u)

# Coriolis on the horizontal momentum, `-f k̂ × ρuₕ`, in the local (u, v)
# basis — the traditional approximation, as in the CG form.
dg_coriolis(f, ρuₕ) = Geometry.UVVector(
    f * ρuₕ.components.data.:2,
    -f * ρuₕ.components.data.:1,
)

##
## Numerical fluxes
##

# Rusanov (local Lax-Friedrichs): the two sides' fluxes averaged, plus a jump
# penalty scaled by the fastest signal speed. The penalty is what makes a DG
# transport scheme stable — a central flux adds no dissipation and the
# grid-scale energy the flow feeds it has nowhere to go.
function dg_thermo_numflux(normal, (y⁻, u⁻, p⁻, λ⁻), (y⁺, u⁺, p⁺, λ⁺))
    F⁻ = dg_thermo_flux(y⁻.ρ, y⁻.ρe, u⁻, p⁻)
    F⁺ = dg_thermo_flux(y⁺.ρ, y⁺.ρe, u⁺, p⁺)
    λ = max(λ⁻, λ⁺)
    return (;
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal + λ / 2 * (y⁻.ρ - y⁺.ρ),
        ρe = ((F⁻.ρe + F⁺.ρe) / 2)' * normal + λ / 2 * (y⁻.ρe - y⁺.ρe),
    )
end

# The momentum interface flux `cartesian_tensor_divergence!` calls. Its first
# face argument is the flux tensor with the momentum axis already rotated into
# the global Cartesian basis, so the penalty is taken on the Cartesian
# momentum `mc` passed alongside it: both sides of a face are then subtracted
# in one basis.
dg_momentum_numflux(normal, (T⁻, mc⁻, λ⁻), (T⁺, mc⁺, λ⁺)) =
    ((T⁻ + T⁺) / 2)' * normal + (max(λ⁻, λ⁺) / 2) * (mc⁻ - mc⁺)

# `curlₕ(ᶠw)` is the one horizontal derivative left in the vector-invariant
# vertical momentum equation, and its element-local value on a DG space misses
# the coupling across faces. The strong-form correction is
# `(1/J) ∮ n̂ × (w* - w)` with a central `w*`, and `n̂ × ê₃ = (n_v, -n_u)` for a
# horizontal `n̂`, which is the whole of it.
dg_w_curl_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

##
## Cache
##

dg_cache(ᶜlocal_geometry, ᶠlocal_geometry, ᶜf) = dg_cache(
    Spaces.discretization(axes(ᶜlocal_geometry)),
    ᶜlocal_geometry,
    ᶠlocal_geometry,
    ᶜf,
)

dg_cache(::Grids.CG, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf) = (;)

function dg_cache(::Grids.DG, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf)
    UVW = Geometry.UVWVector{FT}
    # The flux tensor's type, taken from the flux itself so the scratch and
    # the boundary value cannot drift from what the broadcast produces.
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdivT = similar(ᶜlocal_geometry, UVW)
    ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = dg_thermo_numflux)
    ᶜmomentum_completion =
        Operators.tendency_completion(ᶜdivT; numflux = dg_momentum_numflux)
    return (;
        ᶜfscalar = ᶜf,
        ᶜuₕ = similar(ᶜlocal_geometry, Geometry.UVVector{FT}),
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜmc = similar(ᶜlocal_geometry, UVW),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        ᶜT = similar(ᶜlocal_geometry, Tensor),
        ᶜTc = similar(ᶜlocal_geometry, Tensor),
        ᶜdivT,
        ᶜdYt,
        ᶠu = similar(ᶠlocal_geometry, UVW),
        ᶠTc = similar(ᶠlocal_geometry, Tensor),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, Geometry.UVVector{FT}),
        # No flux of momentum through the top or the bottom of the domain.
        ᶜdivᵥT = Operators.DivergenceF2C(
            top = Operators.SetValue(zero(Tensor)),
            bottom = Operators.SetValue(zero(Tensor)),
        ),
        ᶜthermo_completion,
        ᶜmomentum_completion,
    )
end

##
## Tendency
##

function dg_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp, ᶠω¹², ᶠu¹²) = p
    (; ᶜfscalar, ᶜuₕ, ᶜu, ᶜmc, ᶜλ, ᶜT, ᶜTc, ᶜdivT, ᶜdYt) = p
    (; ᶠu, ᶠTc, ᶠwvec, ᶠwlift, ᶜdivᵥT) = p
    (; ᶜthermo_completion, ᶜmomentum_completion) = p
    ᶜspace = axes(Y.c)
    ᶠspace = axes(Y.f)
    geometry = Spaces.global_geometry(ᶜspace)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)
    ᶠWJ = Fields.local_geometry_field(ᶠspace).WJ

    @. ᶜuₕ = Y.c.ρuₕ / ᶜρ
    @. ᶜu = Geometry.UVWVector(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
    @. ᶜK = norm_sqr(ᶜu) / 2
    @. ᶜp = pressure_ρe(ᶜρe, ᶜK, ᶜΦ, ᶜρ)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    # Mass and energy conservation. Horizontally, one weak-form volume
    # divergence of the physical flux completed by the interface flux;
    # vertically, the finite differences the CG form uses (the `w` half of the
    # vertical flux is implicit).
    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))

    # Momentum conservation. The horizontal flux divergence rotates the
    # momentum axis to Cartesian, completes the interfaces there, and rotates
    # back; the vertical one carries the same rotation, and is rotated back
    # separately, which agrees with rotating the sum because the rotation is
    # linear. Both are projected onto the horizontal axis: their `w`
    # components are the curvature terms of a 3D momentum equation, and the
    # state carries horizontal momentum only.
    @. ᶜT = dg_momentum_flux(ᶜρ, ᶜu, ᶜp)
    @. ᶜmc = Geometry.CartesianVector(ᶜρ * ᶜu, geometry, ᶜcoords)
    Operators.cartesian_tensor_divergence!(
        ᶜdivT,
        ᶜTc,
        ᶜT,
        ᶜmomentum_completion,
        ᶜmc,
        ᶜλ,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(Geometry.UVAxis(), ᶜdivT)

    @. ᶠu = Geometry.UVWVector(C123(ᶠinterp(ᶜuₕ)) + C123(ᶠw))
    @. ᶠTc = Geometry.CartesianTensor(
        dg_momentum_transport(ᶠinterp(ᶜρ), ᶠu),
        geometry,
        ᶠcoords,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(
        Geometry.UVAxis(),
        Geometry.LocalVector(ᶜdivᵥT(ᶠTc), geometry, ᶜcoords),
    )

    @. Yₜ.c.ρuₕ += dg_coriolis(ᶜfscalar, Y.c.ρuₕ)
    # `Φ` is continuous, so its horizontal gradient needs no face lift. It
    # vanishes on a grid without topography, where `z` is a function of the
    # vertical coordinate alone.
    @. Yₜ.c.ρuₕ -= ᶜρ * Geometry.project(Geometry.UVAxis(), gradₕ(ᶜΦ))

    # Vertical momentum: the vector-invariant equation of the CG form, with a
    # face lift completing the one horizontal derivative it takes. The lift
    # acts on the jump in the physical `w`, so the covariant component is
    # converted first; `.components.data.:1` of the result is a view, which is
    # what the face loop reads.
    @. ᶠwvec = Geometry.WVector(ᶠw)
    fill!(parent(ᶠwlift), zero(FT))
    Operators.add_lifting_flux_interior!(
        dg_w_curl_lift,
        ᶠwlift,
        ᶠwvec.components.data.:1,
    )
    @. ᶠω¹² = curlₕ(ᶠw)
    @. ᶠω¹² += CT12(ᶠwlift / ᶠWJ)
    @. ᶠω¹² += ᶠcurlᵥ(C12(ᶜuₕ))
    @. ᶠu¹² = CT12(ᶠinterp(ᶜuₕ))
    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²

    return Yₜ
end
