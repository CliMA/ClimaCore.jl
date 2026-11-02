# DG flux-form explicit tendency: `ρuₕ` at centers (in Cartesian components,
# so `∇·(ρu⊗u)` has no Christoffel terms) and `ρw` at faces. `DG_FLUX` selects
# the horizontal fluxes.

import LinearAlgebra
import ClimaCore: Fields, Geometry, Grids, Operators, Spaces
import ClimaCore.Geometry: ⊗

include("dg_fluxes.jl")

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

##
## Flux schemes
##

const dg_flux_name = get(ENV, "DG_FLUX", "kg-roe")

"""
    dg_flux_scheme(name)

The horizontal assembly named by `DG_FLUX`, as
`(; volume2pt, numflux, face_lift)`:

  - `"rusanov"`: weak-form volume divergence with Rusanov interface fluxes
    (`volume2pt = nothing`).
  - `"kg-rusanov"`, `"kg-roe"`: Kennedy-Gruber flux differencing with the
    two-point Exner force, its face lift, and a Rusanov or Roe interface
    dissipation.
"""
function dg_flux_scheme(name)
    name == "rusanov" && return (;
        volume2pt = nothing,
        numflux = dg_thermo_numflux,
        face_lift = nothing,
    )
    dissipation = if name == "kg-rusanov"
        dg_rusanov_dissipation
    elseif name == "kg-roe"
        DGRoeDissipation(γ)
    else
        error("DG_FLUX must be one of \"rusanov\", \"kg-rusanov\", \
               \"kg-roe\"; got $(repr(name))")
    end
    return (;
        volume2pt = DGKennedyGruberKPEPFlux(cp_d),
        numflux = DGCentralMinusDissipation(
            dg_kennedy_gruber_advective_flux,
            dissipation,
        ),
        face_lift = DGKPEPFaceLift(cp_d),
    )
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
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    ᶠdYt_ρw = similar(ᶠlocal_geometry, FT)
    return (;
        ᶜf_coriolis = ᶜf,
        ᶜuₕ = similar(ᶜlocal_geometry, UV),
        ᶜu = similar(ᶜlocal_geometry, UVW),
        ᶜu_cart = similar(ᶜlocal_geometry, UVW),
        ᶜλ = similar(ᶜlocal_geometry, FT),
        dg_horizontal_cache(scheme.volume2pt, scheme, ᶜlocal_geometry)...,
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
function dg_horizontal_cache(::Nothing, (; numflux), ᶜlocal_geometry)
    UVW = Geometry.UVWVector{FT}
    Tensor = typeof(dg_momentum_transport(zero(FT), zero(UVW)))
    ᶜdYt = similar(ᶜlocal_geometry, NamedTuple{(:ρ, :ρe), Tuple{FT, FT}})
    ᶜdivT = similar(ᶜlocal_geometry, UVW)
    return (;
        ᶜT = similar(ᶜlocal_geometry, Tensor),
        ᶜT_cart = similar(ᶜlocal_geometry, Tensor),
        ᶜdivT,
        ᶜdYt,
        ᶜthermo_completion = Operators.tendency_completion(ᶜdYt; numflux),
        ᶜmomentum_completion =
        Operators.tendency_completion(ᶜdivT; numflux = dg_momentum_numflux),
        volume2pt = nothing,
    )
end

# Flux-differencing scratch: node state, mass-weighted residual and the face
# lift of the two-point Exner force.
function dg_horizontal_cache(
    volume2pt,
    (; numflux, face_lift),
    ᶜlocal_geometry,
)
    UV = Geometry.UVVector{FT}
    ᶜfluxstate = similar(
        ᶜlocal_geometry,
        NamedTuple{
            (
                :ρ, :ρe, :e_tot, :p, :pm, :λ, :uₕ, :u_x, :u_y, :u_z,
                :x̂, :ŷ, :ẑ, :ψ, :θ, :Π,
            ),
            Tuple{
                FT, FT, FT, FT, FT, FT, UV, FT, FT, FT,
                UV, UV, UV, FT, FT, FT,
            },
        },
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
    return (;
        ᶜfluxstate,
        ᶜp_ref = (@. pressure_ref(grav * ᶜlocal_geometry.coordinates.z)),
        ᶜresidual = similar(
            ᶜlocal_geometry,
            NamedTuple{(:ρ, :ρe, :ρu_x, :ρu_y, :ρu_z), NTuple{5, FT}},
        ),
        ᶜL_kpep = similar(ᶜlocal_geometry, UV),
        volume2pt,
        numflux,
        face_lift,
    )
end

##
## Tendency
##

function default_remaining_tendency!(::Grids.DG, ::FluxForm, Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    ᶜuₕ, ᶠw = flux_form_velocities!(Y, p)
    (; ᶜK, ᶜΦ, ᶜp) = p
    (; ᶜf_coriolis, ᶜu, ᶜu_cart, ᶜλ) = p
    (; ᶠu, ᶠT_cart, ᶜdivᵥT) = p
    ᶜspace = axes(Y.c)
    ᶠspace = axes(Y.f)
    geometry = Spaces.global_geometry(ᶜspace)
    ᶜcoords = Fields.coordinate_field(ᶜspace)
    ᶠcoords = Fields.coordinate_field(ᶠspace)

    @. ᶜu = Geometry.UVWVector(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw)))
    @. ᶜu_cart = Geometry.CartesianVector(ᶜu, geometry, ᶜcoords)
    @. ᶜK = norm_sqr(ᶜu) / 2
    @. ᶜp = pressure_ρe(Y.c.ρe, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜλ = dg_wavespeed(ᶜρ, ᶜu, ᶜp)

    dg_horizontal_tendency!(p.volume2pt, Yₜ, Y, p, geometry, ᶜcoords)

    # Vertical transport by `uₕ`.
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))
    @. Yₜ.c.ρe -= ᶜdivᵥ(ᶠinterp((ᶜρe + ᶜp) * ᶜuₕ))
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

    dg_vertical_momentum_tendency!(Yₜ, Y, p, ᶠw)
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

##
## The two horizontal assemblies
##

# Weak form: separate interface fluxes for mass/energy and for momentum, with
# the full pressure in the momentum flux and nodal gravity.
function dg_horizontal_tendency!(::Nothing, Yₜ, Y, p, geometry, ᶜcoords)
    ᶜρ = Y.c.ρ
    ᶜρe = Y.c.ρe
    (; ᶜΦ, ᶜp, ᶜu, ᶜu_cart, ᶜλ, ᶜT, ᶜT_cart, ᶜdivT, ᶜdYt) = p
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
    @. Yₜ.c.ρuₕ -= ᶜρ * Geometry.project(Geometry.UVAxis(), gradₕ(ᶜΦ))
    return Yₜ
end

# Flux differencing over the whole state, into a mass-weighted residual; the
# face part of the two-point Exner force is a lifting on `ρuₕ`.
function dg_horizontal_tendency!(volume2pt, Yₜ, Y, p, geometry, ᶜcoords)
    (; ᶜΦ, ᶜp, ᶜu, ᶜu_cart, ᶜλ, ᶜfluxstate, ᶜp_ref, ᶜresidual) = p
    (; ᶜL_kpep, numflux, face_lift) = p
    ᶜWJ = Fields.local_geometry_field(axes(Y.c)).WJ

    @. ᶜfluxstate.ρ = Y.c.ρ
    @. ᶜfluxstate.ρe = Y.c.ρe
    @. ᶜfluxstate.e_tot = Y.c.ρe / Y.c.ρ
    @. ᶜfluxstate.p = ᶜp
    @. ᶜfluxstate.pm = ᶜp - ᶜp_ref
    @. ᶜfluxstate.λ = ᶜλ
    @. ᶜfluxstate.uₕ = Geometry.project(Geometry.UVAxis(), ᶜu)
    @. ᶜfluxstate.u_x = ᶜu_cart.components.data.:1
    @. ᶜfluxstate.u_y = ᶜu_cart.components.data.:2
    @. ᶜfluxstate.u_z = ᶜu_cart.components.data.:3
    @. ᶜfluxstate.ψ = ᶜΦ - phi_ref(ᶜp)
    @. ᶜfluxstate.θ = theta_prime(Y.c.ρ, ᶜp)
    @. ᶜfluxstate.Π = exner(ᶜp)

    fill!(parent(ᶜresidual), zero(FT))
    Operators.add_flux_differencing_divergence!(
        volume2pt,
        ᶜresidual,
        ᶜfluxstate,
    )
    Operators.add_numerical_flux_interior!(numflux, ᶜresidual, ᶜfluxstate)

    @. Yₜ.c.ρ += ᶜresidual.ρ / ᶜWJ
    @. Yₜ.c.ρe += ᶜresidual.ρe / ᶜWJ
    @. Yₜ.c.ρuₕ += dg_cartesian_momentum_tendency(
        ᶜresidual,
        ᶜWJ,
        geometry,
        ᶜcoords,
    )
    fill!(parent(ᶜL_kpep), zero(FT))
    Operators.add_lifting_flux_interior!(
        face_lift,
        ᶜL_kpep,
        ᶜfluxstate.ψ,
        ᶜfluxstate.θ,
        ᶜfluxstate.Π,
        Y.c.ρ,
    )
    @. Yₜ.c.ρuₕ -= ᶜL_kpep / ᶜWJ
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
