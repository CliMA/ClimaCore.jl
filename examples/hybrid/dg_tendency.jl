# Flux-form horizontal tendency for the DG form of the staggered nonhydrostatic
# model: prognostic momentum `ρuₕ`, coupled across element faces by an
# interface numerical flux. The vertical terms, the implicit split and the
# vector-invariant `w` equation are shared with the CG form.
#
# Momentum is carried in global Cartesian components, where the Christoffel
# terms of `∇·(ρu⊗u)` vanish. `DG_FLUX` selects the horizontal assembly (see
# `dg_flux_scheme`). Total water `ρq_tot`, when present, moves with the mass
# flux; its sources and its pressure are the case file's.

import LinearAlgebra
import ClimaCore: Fields, Geometry, Grids, Operators, Spaces
import ClimaCore.Geometry: ⊗

##
## Physical fluxes
##

# `u` is the full 3D velocity in the local orthonormal basis.
dg_thermo_flux(ρ, ρe, u, pres) = (; ρ = ρ * u, ρe = (ρe + pres) * u)

# `ρu⊗u + p𝟙`, transport axis first.
dg_momentum_flux(ρ, u, pres) = (ρ * u) ⊗ u + pres * LinearAlgebra.I

# Without pressure, for the vertical divergence (the vertical pressure
# gradient is in the implicit `w` equation).
dg_momentum_transport(ρ, u) = (ρ * u) ⊗ u

# Fastest signal speed, `c + |u|`.
dg_wavespeed(ρ, u, pres) = sqrt(γ * pres / ρ) + norm(u)

# `-f k̂ × ρuₕ` in the local (u, v) basis.
dg_coriolis(f, ρuₕ) = Geometry.UVVector(
    f * ρuₕ.components.data.:2,
    -f * ρuₕ.components.data.:1,
)

##
## Interface fluxes for the weak-form assembly
##

# Rusanov flux for mass and energy.
function dg_thermo_numflux(normal, (y⁻, u⁻, p⁻, λ⁻), (y⁺, u⁺, p⁺, λ⁺))
    F⁻ = dg_thermo_flux(y⁻.ρ, y⁻.ρe, u⁻, p⁻)
    F⁺ = dg_thermo_flux(y⁺.ρ, y⁺.ρe, u⁺, p⁺)
    λ = max(λ⁻, λ⁺)
    return (;
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal + λ / 2 * (y⁻.ρ - y⁺.ρ),
        ρe = ((F⁻.ρe + F⁺.ρe) / 2)' * normal + λ / 2 * (y⁻.ρe - y⁺.ρe),
    )
end

# Rusanov flux for momentum, on the Cartesian-rotated flux tensor that
# `cartesian_tensor_divergence!` passes.
dg_momentum_numflux(normal, (T⁻, ρ⁻, uc⁻, λ⁻), (T⁺, ρ⁺, uc⁺, λ⁺)) =
    ((T⁻ + T⁺) / 2)' * normal +
    (max(λ⁻, λ⁺) / 2) * (ρ⁻ * uc⁻ - ρ⁺ * uc⁺)

##
## The vertical momentum equation
##

# Central face lift completing `curlₕ(ᶠw)`: `n̂ × ê₃ = (n_v, -n_u)`.
dg_w_curl_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

##
## Two-point volume flux
##

"""
    dg_kennedy_gruber_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for `(ρ, ρe, ρu⃗)` with Cartesian momentum
(kinetic-energy and pressure-equilibrium preserving), for
`Operators.add_flux_differencing_divergence!`.
"""
function dg_kennedy_gruber_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    F = (;
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄ * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄ * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄ * Ē3n,
    )
    return dg_with_tracer(F, y_a, y_b)
end

# Water moves with the mass flux at the mean specific humidity.
@inline dg_with_tracer(F, y_a, y_b) =
    haskey(y_a, :q) ? (; F..., ρq_tot = F.ρ * (y_a.q + y_b.q) / 2) : F

##
## Interface fluxes for the flux-differencing assembly
##

"""
    dg_rusanov(normal, argvals⁻, argvals⁺)

Kennedy-Gruber central flux plus a jump penalty at `λ = c + |u|`.
"""
function dg_rusanov(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = dg_kennedy_gruber_flux(normal, normal, y⁻, y⁺)
    Δ = (;
        ρ = y⁺.ρ - y⁻.ρ,
        ρe = y⁺.ρe - y⁻.ρe,
        ρu1 = y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1,
        ρu2 = y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2,
        ρu3 = y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3,
    )
    Δ = haskey(y⁻, :q) ? (; Δ..., ρq_tot = y⁺.ρ * y⁺.q - y⁻.ρ * y⁻.q) : Δ
    return map((f, δ) -> f - λ / 2 * δ, F, Δ)
end

"""
    dg_roe(normal, argvals⁻, argvals⁺)

Kennedy-Gruber central flux plus Roe dissipation: acoustic waves damped at
`|ûₙ ± ĉ|`, entropy and shear waves at `max(|ûₙ|, ĉ/20)`. The floor keeps
density jumps in near-stagnant columns damped.
"""
function dg_roe(normal, (y⁻,), (y⁺,))
    F = dg_kennedy_gruber_flux(normal, normal, y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ)
    # face normal in Cartesian components
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    # jumps and wave amplitudes
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.p - y⁻.p
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δun = Δu1 * n1 + Δu2 * n2 + Δu3 * n3
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    # `B` absorbs the geopotential and vertical kinetic parts of `ρe`
    B = Ĥ - ĉ^2 / (γd - 1)
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3))
    D = (; ρ = Dρ, ρe = Dρe, ρu1 = Dρu1, ρu2 = Dρu2, ρu3 = Dρu3)
    # water rides every wave at its Roe average; its own jump is a contact
    D = if haskey(y⁻, :q)
        q̂ = a⁻ * y⁻.q + a⁺ * y⁺.q
        (; D..., ρq_tot = q̂ * Dρ + s₀ * ρ̂ * (y⁺.q - y⁻.q))
    else
        D
    end
    return map((f, d) -> f - d / 2, F, D)
end

##
## Flux schemes
##

const dg_flux_name = get(ENV, "DG_FLUX", "kg-roe")

"""
    dg_flux_scheme(name)

The horizontal assembly named by `DG_FLUX`, as `(; volume2pt, numflux)`:

  - `"rusanov"`: weak-form volume divergence with Rusanov interface fluxes
    (`volume2pt = nothing`).
  - `"kg-rusanov"`, `"kg-roe"`: Kennedy-Gruber flux differencing with a
    Rusanov or Roe interface flux.
"""
function dg_flux_scheme(name)
    name == "rusanov" &&
        return (; volume2pt = nothing, numflux = dg_thermo_numflux)
    numflux = if name == "kg-rusanov"
        dg_rusanov
    elseif name == "kg-roe"
        dg_roe
    else
        error("DG_FLUX must be one of \"rusanov\", \"kg-rusanov\", \
               \"kg-roe\"; got $(repr(name))")
    end
    return (; volume2pt = dg_kennedy_gruber_flux, numflux)
end

##
## Cache
##

dg_cache(ᶜlocal_geometry, ᶠlocal_geometry, ᶜf, Y) = dg_cache(
    Spaces.discretization(axes(ᶜlocal_geometry)),
    momentum_form(Y.c),
    ᶜlocal_geometry,
    ᶠlocal_geometry,
    ᶜf,
    Y,
)

dg_cache(::Grids.CG, _, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf, Y) = (;)

function dg_cache(::Grids.DG, ::FluxForm, ᶜlocal_geometry, ᶠlocal_geometry, ᶜf, Y)
    UV = Geometry.UVVector{FT}
    UVW = Geometry.UVWVector{FT}
    scheme = dg_flux_scheme(dg_flux_name)
    isnothing(scheme.volume2pt) && has_moisture(Y.c) &&
        error("DG_FLUX=rusanov does not transport water; use a \
               flux-differencing scheme")
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    return (;
        ᶜfscalar = ᶜf,
        ᶜuₕ = similar(ᶜlocal_geometry, UV),
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜuc = similar(ᶜlocal_geometry, UVW),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        dg_horizontal_cache(scheme.volume2pt, ᶜlocal_geometry, scheme, Y)...,
        ᶠu = similar(ᶠlocal_geometry, UVW),
        ᶠTc = similar(ᶠlocal_geometry, Tensor),
        ᶠwvec = similar(ᶠlocal_geometry, Geometry.WVector{FT}),
        ᶠwlift = similar(ᶠlocal_geometry, UV),
        # no momentum flux through the top or bottom
        ᶜdivᵥT = Operators.DivergenceF2C(
            top = Operators.SetValue(zero(Tensor)),
            bottom = Operators.SetValue(zero(Tensor)),
        ),
    )
end

# Weak-form assembly scratch.
function dg_horizontal_cache(::Nothing, ᶜlocal_geometry, scheme, Y)
    UVW = Geometry.UVWVector{FT}
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdivT = similar(ᶜlocal_geometry, UVW)
    return (;
        ᶜT = similar(ᶜlocal_geometry, Tensor),
        ᶜTc = similar(ᶜlocal_geometry, Tensor),
        ᶜdivT,
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = scheme.numflux),
        ᶜmomentum_completion =
        Operators.tendency_completion(ᶜdivT; numflux = dg_momentum_numflux),
        volume2pt = nothing,
    )
end

# Flux-differencing scratch: the node state the fluxes read (with `q` when the
# state carries water) and the mass-weighted residual.
function dg_horizontal_cache(volume2pt::V, ᶜlocal_geometry, scheme, Y) where {V}
    UV = Geometry.UVVector{FT}
    state_names = (:ρ, :ρe, :e, :p, :λ, :uv, :u1, :u2, :u3, :E1, :E2, :E3)
    state_types = (FT, FT, FT, FT, FT, UV, FT, FT, FT, UV, UV, UV)
    residual_names = (:ρ, :ρe, :ρu1, :ρu2, :ρu3)
    if has_moisture(Y.c)
        state_names = (state_names..., :q)
        state_types = (state_types..., FT)
        residual_names = (residual_names..., :ρq_tot)
    end
    ᶜfluxstate = similar(
        ᶜlocal_geometry,
        NamedTuple{state_names, Tuple{state_types...}},
    )
    # Tangential projections of the Cartesian unit vectors, filled once.
    space = axes(ᶜlocal_geometry)
    geometry = Spaces.global_geometry(space)
    coords = Fields.coordinate_field(space)
    for (Ec, ê) in (
        (ᶜfluxstate.E1, Geometry.Cartesian123Vector(FT(1), FT(0), FT(0))),
        (ᶜfluxstate.E2, Geometry.Cartesian123Vector(FT(0), FT(1), FT(0))),
        (ᶜfluxstate.E3, Geometry.Cartesian123Vector(FT(0), FT(0), FT(1))),
    )
        tangent(geom, coord) = Geometry.project(
            Geometry.UVAxis(),
            Geometry.LocalVector(ê, geom, coord),
        )
        Ec .= tangent.(Ref(geometry), coords)
    end
    numflux = scheme.numflux
    return (;
        ᶜfluxstate,
        ᶜresidual = similar(
            ᶜlocal_geometry,
            NamedTuple{
                residual_names,
                NTuple{length(residual_names), FT},
            },
        ),
        volume2pt,
        numflux,
    )
end

##
## Tendency
##

function dg_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜp) = p
    (; ᶜfscalar, ᶜuₕ, ᶜu, ᶜuc, ᶜλ) = p
    (; ᶠu, ᶠTc, ᶜdivᵥT) = p
    ᶜspace = axes(Y.c)
    ᶠspace = axes(Y.f)
    geometry = Spaces.global_geometry(ᶜspace)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)

    @. ᶜuₕ = Y.c.ρuₕ / ᶜρ
    @. ᶜu = Geometry.UVWVector(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
    @. ᶜuc = Geometry.CartesianVector(ᶜu, geometry, ᶜcoords)
    @. ᶜK = norm_sqr(ᶜu) / 2
    thermo_pressure!(ᶜp, Y.c, ᶜK, ᶜΦ, p.moisture)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    dg_horizontal_tendency!(p.volume2pt, Yₜ, Y, p, geometry, ᶜcoords)

    # Vertical transport by `uₕ` (the `w` part of mass and energy is implicit).
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))
    has_moisture(Y.c) && dg_vertical_water_tendency!(Yₜ, Y, ᶜuₕ)
    # Vertical momentum flux, rotated to Cartesian and back like the horizontal.
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
    # `Φ` is continuous, so its gradient needs no face lift.
    @. Yₜ.c.ρuₕ -= ᶜρ * Geometry.project(Geometry.UVAxis(), gradₕ(ᶜΦ))

    dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    return Yₜ
end

# Explicit vertical water transport: by `uₕ` as for mass, and by `w` as the
# implicit mass flux times an upwinded `q`.
function dg_vertical_water_tendency!(Yₜ, Y, ᶜuₕ)
    ᶜρ = Y.c.ρ
    ᶜρq = Y.c.ρq_tot
    @. Yₜ.c.ρq_tot -= ᶜdivᵥ(ᶠinterp(ᶜρq * ᶜuₕ))
    @. Yₜ.c.ρq_tot -=
        ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product3(Y.f.w, ᶜρq / ᶜρ))
    return Yₜ
end

# Vector-invariant `w` equation with a face lift completing `curlₕ(ᶠw)`.
# Shared by both DG forms; `ᶜuₕ` may be in either basis.
function dg_w_tendency!(Yₜ, Y, p, ᶜuₕ)
    ᶠw = Y.f.w
    (; ᶠω¹², ᶠu¹², ᶠwvec, ᶠwlift) = p
    ᶠWJ = Fields.local_geometry_field(axes(Y.f)).WJ
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

##
## The two horizontal assemblies
##

# Weak form: separate interface fluxes for mass/energy and for momentum.
function dg_horizontal_tendency!(::Nothing, Yₜ, Y, p, geometry, ᶜcoords)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    (; ᶜp, ᶜu, ᶜuc, ᶜλ, ᶜT, ᶜTc, ᶜdivT, ᶜdYt) = p
    (; ᶜthermo_completion, ᶜmomentum_completion) = p

    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe

    @. ᶜT = dg_momentum_flux(ᶜρ, ᶜu, ᶜp)
    Operators.cartesian_tensor_divergence!(
        ᶜdivT,
        ᶜTc,
        ᶜT,
        ᶜmomentum_completion,
        ᶜρ,
        ᶜuc,
        ᶜλ,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(Geometry.UVAxis(), ᶜdivT)
    return Yₜ
end

# Flux differencing: one volume term and one interface flux over the whole
# state, accumulated into a mass-weighted residual.
function dg_horizontal_tendency!(volume2pt::V, Yₜ, Y, p, geometry, ᶜcoords) where {V}
    (; ᶜp, ᶜu, ᶜuc, ᶜλ, ᶜfluxstate, ᶜresidual, numflux) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜfluxstate.ρ = Y.c.ρ
    @. ᶜfluxstate.ρe = Y.c.ρe
    @. ᶜfluxstate.e = Y.c.ρe / Y.c.ρ
    @. ᶜfluxstate.p = ᶜp
    @. ᶜfluxstate.λ = ᶜλ
    @. ᶜfluxstate.uv = Geometry.project(Geometry.UVAxis(), ᶜu)
    @. ᶜfluxstate.u1 = ᶜuc.components.data.:1
    @. ᶜfluxstate.u2 = ᶜuc.components.data.:2
    @. ᶜfluxstate.u3 = ᶜuc.components.data.:3
    has_moisture(Y.c) && (@. ᶜfluxstate.q = Y.c.ρq_tot / Y.c.ρ)

    fill!(parent(ᶜresidual), zero(FT))
    Operators.add_flux_differencing_divergence!(
        volume2pt,
        ᶜresidual,
        ᶜfluxstate,
    )
    Operators.add_numerical_flux_interior!(numflux, ᶜresidual, ᶜfluxstate)

    @. Yₜ.c.ρ += ᶜresidual.ρ / ᶜWJ
    @. Yₜ.c.ρe += ᶜresidual.ρe / ᶜWJ
    has_moisture(Y.c) && (@. Yₜ.c.ρq_tot += ᶜresidual.ρq_tot / ᶜWJ)
    @. Yₜ.c.ρuₕ += dg_cartesian_momentum_tendency(
        ᶜresidual,
        ᶜWJ,
        geometry,
        ᶜcoords,
    )
    return Yₜ
end

# Cartesian momentum residual, back in the local frame, horizontal part.
dg_cartesian_momentum_tendency(r, WJ, geometry, coord) = Geometry.project(
    Geometry.UVAxis(),
    Geometry.LocalVector(
        Geometry.UVWVector(r.ρu1, r.ρu2, r.ρu3) / WJ,
        geometry,
        coord,
    ),
)
