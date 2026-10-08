# DG flux-form explicit tendency: `ρuₕ` at centers (in Cartesian components,
# so `∇·(ρu⊗u)` has no Christoffel terms) and `ρw` at faces. `DG_FLUX` selects
# the horizontal fluxes; optional `ρq_tot` moves with the mass flux.

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

# Without pressure, for the vertical divergence.
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

# Rusanov flux for momentum (Cartesian-rotated flux tensor).
dg_momentum_numflux(normal, (T⁻, ρ⁻, u_cart⁻, λ⁻), (T⁺, ρ⁺, u_cart⁺, λ⁺)) =
    ((T⁻ + T⁺) / 2)' * normal +
    (max(λ⁻, λ⁺) / 2) * (ρ⁻ * u_cart⁻ - ρ⁺ * u_cart⁺)

##
## The vertical momentum equation
##

# Kennedy-Gruber flux for horizontal `ρw` advection, advective-speed penalty.
dg_vertical_momentum_numflux(normal, (m⁻, u⁻, λ⁻), (m⁺, u⁺, λ⁺)) =
    ((m⁻ + m⁺) / 2) * (((u⁻ + u⁺) / 2)' * normal) +
    max(λ⁻, λ⁺) / 2 * (m⁻ - m⁺)

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
    ē_tot = (y_a.e_tot + y_b.e_tot) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ū_n = (y_a.uₕ' * nvec_a + y_b.uₕ' * nvec_b) / 2
    ū_x = (y_a.u_x + y_b.u_x) / 2
    ū_y = (y_a.u_y + y_b.u_y) / 2
    ū_z = (y_a.u_z + y_b.u_z) / 2
    n̄_x = (y_a.x̂' * nvec_a + y_b.x̂' * nvec_b) / 2
    n̄_y = (y_a.ŷ' * nvec_a + y_b.ŷ' * nvec_b) / 2
    n̄_z = (y_a.ẑ' * nvec_a + y_b.ẑ' * nvec_b) / 2
    F = (;
        ρ = ρ̄ * ū_n,
        ρe = (ρ̄ * ē_tot + p̄) * ū_n,
        ρu_x = ρ̄ * ū_x * ū_n + p̄ * n̄_x,
        ρu_y = ρ̄ * ū_y * ū_n + p̄ * n̄_y,
        ρu_z = ρ̄ * ū_z * ū_n + p̄ * n̄_z,
    )
    return dg_with_water(F, y_a, y_b)
end

# Water moves with the mass flux at the mean specific humidity.
@inline dg_with_water(F, y_a, y_b) =
    haskey(y_a, :q_tot) ?
    (; F..., ρq_tot = F.ρ * (y_a.q_tot + y_b.q_tot) / 2) : F

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
        ρu_x = y⁺.ρ * y⁺.u_x - y⁻.ρ * y⁻.u_x,
        ρu_y = y⁺.ρ * y⁺.u_y - y⁻.ρ * y⁻.u_y,
        ρu_z = y⁺.ρ * y⁺.u_z - y⁻.ρ * y⁻.u_z,
    )
    if haskey(y⁻, :q_tot)
        Δ = (; Δ..., ρq_tot = y⁺.ρ * y⁺.q_tot - y⁻.ρ * y⁻.q_tot)
    end
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
    n_x = y⁻.x̂' * normal
    n_y = y⁻.ŷ' * normal
    n_z = y⁻.ẑ' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û_x = a⁻ * y⁻.u_x + a⁺ * y⁺.u_x
    û_y = a⁻ * y⁻.u_y + a⁺ * y⁺.u_y
    û_z = a⁻ * y⁻.u_z + a⁺ * y⁺.u_z
    Ĥ = a⁻ * (y⁻.e_tot + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e_tot + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    û_n = û_x * n_x + û_y * n_y + û_z * n_z
    # jumps and wave amplitudes
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.p - y⁻.p
    Δu_x = y⁺.u_x - y⁻.u_x
    Δu_y = y⁺.u_y - y⁻.u_y
    Δu_z = y⁺.u_z - y⁻.u_z
    Δu_n = Δu_x * n_x + Δu_y * n_y + Δu_z * n_z
    α₊ = (Δp + ρ̂ * ĉ * Δu_n) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δu_n) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(û_n + ĉ)
    s₋ = abs(û_n - ĉ)
    s₀ = max(abs(û_n), ĉ / 20)
    Δu_tan_x = Δu_x - Δu_n * n_x
    Δu_tan_y = Δu_y - Δu_n * n_y
    Δu_tan_z = Δu_z - Δu_n * n_z
    # `B` absorbs the geopotential and vertical kinetic parts of `ρe`
    B = Ĥ - ĉ^2 / (γd - 1)
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu_x =
        s₊ * α₊ * (û_x + ĉ * n_x) + s₋ * α₋ * (û_x - ĉ * n_x) +
        s₀ * (α₀ * û_x + ρ̂ * Δu_tan_x)
    Dρu_y =
        s₊ * α₊ * (û_y + ĉ * n_y) + s₋ * α₋ * (û_y - ĉ * n_y) +
        s₀ * (α₀ * û_y + ρ̂ * Δu_tan_y)
    Dρu_z =
        s₊ * α₊ * (û_z + ĉ * n_z) + s₋ * α₋ * (û_z - ĉ * n_z) +
        s₀ * (α₀ * û_z + ρ̂ * Δu_tan_z)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * û_n) + s₋ * α₋ * (Ĥ - ĉ * û_n) +
        s₀ *
        (α₀ * B + ρ̂ * (û_x * Δu_tan_x + û_y * Δu_tan_y + û_z * Δu_tan_z))
    D = (; ρ = Dρ, ρe = Dρe, ρu_x = Dρu_x, ρu_y = Dρu_y, ρu_z = Dρu_z)
    # water rides every wave at its Roe average; its own jump is a contact
    D = if haskey(y⁻, :q_tot)
        q̂ = a⁻ * y⁻.q_tot + a⁺ * y⁺.q_tot
        (; D..., ρq_tot = q̂ * Dρ + s₀ * ρ̂ * (y⁺.q_tot - y⁻.q_tot))
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
    ᶠdYt_ρw = similar(ᶠlocal_geometry, FT)
    return (;
        ᶜf_coriolis = ᶜf,
        ᶜuₕ = similar(ᶜlocal_geometry, UV),
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜu_cart = similar(ᶜlocal_geometry, UVW),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        dg_horizontal_cache(scheme.volume2pt, ᶜlocal_geometry, scheme, Y)...,
        ᶠu = similar(ᶠlocal_geometry, UVW),
        ᶠT_cart = similar(ᶠlocal_geometry, Tensor),
        ᶠw = similar(ᶠlocal_geometry, C3{FT}),
        ᶠuₕ = similar(ᶠlocal_geometry, UV),
        ᶠρw_value = similar(ᶠlocal_geometry, FT),
        ᶠλ = similar(ᶠlocal_geometry, FT),
        ᶠdYt_ρw,
        ᶠρw_completion = Operators.tendency_completion(
            ᶠdYt_ρw;
            numflux = dg_vertical_momentum_numflux,
        ),
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
        ᶜT_cart = similar(ᶜlocal_geometry, Tensor),
        ᶜdivT,
        ᶜdYt,
        ᶜthermo_completion =
        Operators.tendency_completion(ᶜdYt; numflux = scheme.numflux),
        ᶜmomentum_completion =
        Operators.tendency_completion(ᶜdivT; numflux = dg_momentum_numflux),
        volume2pt = nothing,
    )
end

# Flux-differencing scratch: node state and mass-weighted residual.
function dg_horizontal_cache(volume2pt::V, ᶜlocal_geometry, scheme, Y) where {V}
    UV = Geometry.UVVector{FT}
    state_names = (:ρ, :ρe, :e_tot, :p, :λ, :uₕ, :u_x, :u_y, :u_z, :x̂, :ŷ, :ẑ)
    state_types = (FT, FT, FT, FT, FT, UV, FT, FT, FT, UV, UV, UV)
    residual_names = (:ρ, :ρe, :ρu_x, :ρu_y, :ρu_z)
    if has_moisture(Y.c)
        state_names = (state_names..., :q_tot)
        state_types = (state_types..., FT)
        residual_names = (residual_names..., :ρq_tot)
    end
    ᶜfluxstate = similar(
        ᶜlocal_geometry,
        NamedTuple{state_names, Tuple{state_types...}},
    )
    # Cartesian unit vectors in the local (u, v) basis.
    space = axes(ᶜlocal_geometry)
    geometry = Spaces.global_geometry(space)
    coords = Fields.coordinate_field(space)
    for (Ec, ê) in (
        (ᶜfluxstate.x̂, Geometry.Cartesian123Vector(FT(1), FT(0), FT(0))),
        (ᶜfluxstate.ŷ, Geometry.Cartesian123Vector(FT(0), FT(1), FT(0))),
        (ᶜfluxstate.ẑ, Geometry.Cartesian123Vector(FT(0), FT(0), FT(1))),
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
    ᶠw = face_velocity(Y, p)
    (; ᶜK, ᶜΦ, ᶜp) = p
    (; ᶜf_coriolis, ᶜuₕ, ᶜu, ᶜu_cart, ᶜλ) = p
    (; ᶠu, ᶠT_cart, ᶜdivᵥT) = p
    ᶜspace = axes(Y.c)
    ᶠspace = axes(Y.f)
    geometry = Spaces.global_geometry(ᶜspace)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)

    @. ᶜuₕ = Y.c.ρuₕ / ᶜρ
    @. ᶜu = Geometry.UVWVector(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
    @. ᶜu_cart = Geometry.CartesianVector(ᶜu, geometry, ᶜcoords)
    @. ᶜK = norm_sqr(ᶜu) / 2
    thermo_pressure!(ᶜp, Y.c, ᶜK, ᶜΦ, p.moisture)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    dg_horizontal_tendency!(p.volume2pt, Yₜ, Y, p, geometry, ᶜcoords)

    # Vertical transport by `uₕ` (by `ρw`: implicit).
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))
    if has_moisture(Y.c)
        @. Yₜ.c.ρq_tot -= ᶜdivᵥ(ᶠinterp(Y.c.ρq_tot * ᶜuₕ))
        dg_vertical_water_tendency!(Yₜ, Y, p, ᶠw)
    end
    # Vertical flux of `ρuₕ`, rotated to Cartesian and back.
    @. ᶠu = Geometry.UVWVector(C123(ᶠinterp(ᶜuₕ)) + C123(ᶠw))
    @. ᶠT_cart = Geometry.CartesianTensor(
        dg_momentum_transport(ᶠinterp(ᶜρ), ᶠu),
        geometry,
        ᶠcoords,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(
        Geometry.UVAxis(),
        Geometry.LocalVector(ᶜdivᵥT(ᶠT_cart), geometry, ᶜcoords),
    )

    @. Yₜ.c.ρuₕ += dg_coriolis(ᶜf_coriolis, Y.c.ρuₕ)
    # `Φ` is continuous: no face flux needed.
    @. Yₜ.c.ρuₕ -= ᶜρ * Geometry.project(Geometry.UVAxis(), gradₕ(ᶜΦ))

    dg_vertical_momentum_tendency!(Yₜ, Y, p, ᶠw)
    return Yₜ
end

# Vertical water transport by `w` (explicit): mass flux times a monotone
# (Lin-van Leer) face `q_tot`, keeping element means of `ρq_tot` non-negative.
const ᶠmonotone_product = Operators.LinVanLeerC2F(
    constraint = Operators.MonotoneLocalExtrema(),
)
function dg_vertical_water_tendency!(Yₜ, Y, p, ᶠw)
    ᶜρ = Y.c.ρ
    @. Yₜ.c.ρq_tot -= ᶜdivᵥ(
        ᶠinterp(ᶜρ) * ᶠmonotone_product(ᶠw, Y.c.ρq_tot / ᶜρ, p.dt),
    )
    return Yₜ
end

# Explicit `ρw` advection, `-∇ᵥ·(ρw w) - ∇ₕ·(uₕ ρw)`; zero at the boundaries.
function dg_vertical_momentum_tendency!(Yₜ, Y, p, ᶠw)
    ᶠρw = Y.f.ρw
    (; ᶜuₕ, ᶠuₕ, ᶠρw_value, ᶠλ, ᶠdYt_ρw, ᶠρw_completion) = p
    @. Yₜ.f.ρw -= C3(
        ᶠdivᵥ_tensor(ᶜinterp(Geometry.WVector(ᶠρw) ⊗ Geometry.WVector(ᶠw))),
    )
    @. ᶠuₕ = ᶠinterp(ᶜuₕ)
    @. ᶠρw_value = vertical_component(Geometry.WVector(ᶠρw))
    @. ᶠλ = norm(ᶠuₕ)
    @. ᶠdYt_ρw = -wdivₕ(ᶠuₕ * ᶠρw_value)
    Operators.complete_tendency!(ᶠρw_completion, ᶠdYt_ρw, ᶠρw_value, ᶠuₕ, ᶠλ)
    @. Yₜ.f.ρw += C3(Geometry.WVector(ᶠdYt_ρw))
    @. Yₜ.f.ρw = ᶠno_momentum_flux(Yₜ.f.ρw)
    return Yₜ
end

const ᶠdivᵥ_tensor = Operators.DivergenceC2F(
    bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
    top = Operators.SetDivergence(Geometry.WVector(FT(0))),
)
const ᶠno_momentum_flux = Operators.SetBoundaryOperator(
    bottom = Operators.SetValue(C3(FT(0))),
    top = Operators.SetValue(C3(FT(0))),
)

@inline vertical_component(w) = w.components.data.:1

# Vector-invariant `w` equation (DG vector-invariant form).
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
    (; ᶜp, ᶜu, ᶜu_cart, ᶜλ, ᶜT, ᶜT_cart, ᶜdivT, ᶜdYt) = p
    (; ᶜthermo_completion, ᶜmomentum_completion) = p

    @. ᶜdYt = -wdivₕ(dg_thermo_flux(ᶜρ, ᶜρe, ᶜu, ᶜp))
    Operators.complete_tendency!(ᶜthermo_completion, ᶜdYt, Y.c, ᶜu, ᶜp, ᶜλ)
    @. Yₜ.c.ρ += ᶜdYt.ρ
    @. Yₜ.c.ρe += ᶜdYt.ρe

    @. ᶜT = dg_momentum_flux(ᶜρ, ᶜu, ᶜp)
    Operators.cartesian_tensor_divergence!(
        ᶜdivT,
        ᶜT_cart,
        ᶜT,
        ᶜmomentum_completion,
        ᶜρ,
        ᶜu_cart,
        ᶜλ,
    )
    @. Yₜ.c.ρuₕ -= Geometry.project(Geometry.UVAxis(), ᶜdivT)
    return Yₜ
end

# Flux differencing over the whole state, into a mass-weighted residual.
function dg_horizontal_tendency!(volume2pt::V, Yₜ, Y, p, geometry, ᶜcoords) where {V}
    (; ᶜp, ᶜu, ᶜu_cart, ᶜλ, ᶜfluxstate, ᶜresidual, numflux) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜfluxstate.ρ = Y.c.ρ
    @. ᶜfluxstate.ρe = Y.c.ρe
    @. ᶜfluxstate.e_tot = Y.c.ρe / Y.c.ρ
    @. ᶜfluxstate.p = ᶜp
    @. ᶜfluxstate.λ = ᶜλ
    @. ᶜfluxstate.uₕ = Geometry.project(Geometry.UVAxis(), ᶜu)
    @. ᶜfluxstate.u_x = ᶜu_cart.components.data.:1
    @. ᶜfluxstate.u_y = ᶜu_cart.components.data.:2
    @. ᶜfluxstate.u_z = ᶜu_cart.components.data.:3
    has_moisture(Y.c) && (@. ᶜfluxstate.q_tot = Y.c.ρq_tot / Y.c.ρ)

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
        Geometry.UVWVector(r.ρu_x, r.ρu_y, r.ρu_z) / WJ,
        geometry,
        coord,
    ),
)
