# Two-point and interface fluxes of the flux-form DG horizontal dynamics for
# `(ρ, ρe, ρu⃗)` with Cartesian momentum. Parameters are fields of the
# callable structs.
#
# Flux state fields: `ρ`, `ρe`, `e_tot`, `p`, `pm = p − p_ref(Φ)`, `λ`, `uₕ`
# (orthonormal horizontal velocity), `u_x`, `u_y`, `u_z` (Cartesian components
# of `uₕ`), `x̂`, `ŷ`, `ẑ` (tangential projections of the Cartesian unit
# vectors), `ψ = Φ − Φ_r(p)`, `θ = θ′ = θ − θ_r(p)` and `Π` (Exner function).

import LinearAlgebra
import ClimaCore: Geometry, Operators, Fields, Spaces

##
## Two-point volume fluxes
##

"""
    dg_kennedy_gruber_advective_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux of the advection, with Cartesian momentum:
`F_ρ = {ρ}{u_n}`, `F_ρe = ({ρ}{e_tot} + {p}){u_n}`, `F_ρu_c = {ρ}{u_c}{u_n}`,
with `{u_n}` the average of the nodal contravariant velocities. The momentum
flux is `{u_c}` times a symmetric mass flux, so the advection is
kinetic-energy preserving. It carries no pressure.
"""
function dg_kennedy_gruber_advective_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē_tot = (y_a.e_tot + y_b.e_tot) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ū_n = (y_a.uₕ' * nvec_a + y_b.uₕ' * nvec_b) / 2
    ū_x = (y_a.u_x + y_b.u_x) / 2
    ū_y = (y_a.u_y + y_b.u_y) / 2
    ū_z = (y_a.u_z + y_b.u_z) / 2
    return (;
        ρ = ρ̄ * ū_n,
        ρe = (ρ̄ * ē_tot + p̄) * ū_n,
        ρu_x = ρ̄ * ū_x * ū_n,
        ρu_y = ρ̄ * ū_y * ū_n,
        ρu_z = ρ̄ * ū_z * ū_n,
    )
end

"""
    DGKennedyGruberKPEPFlux(cp_d)

Two-point volume flux, a callable `(nvec_a, nvec_b, y_a, y_b)`: the advective
Kennedy-Gruber flux plus, in the momentum slots, the Exner pressure force and
gravity in the two-point form of Souza et al. (2023),

    {ρ} [ (ψ_b − ψ_a)/2 + c_pd {θ′} (Π_b − Π_a)/2 ] (ê_c ⋅ J∇ξ)_a,

with the own-node metric. With the mass flux `{ρ}{u_n}` the geopotential
part exchanges kinetic and potential energy exactly (KPEP), and the pressure
part cancels it at the hydrostatic reference. The term vanishes for
`y_a == y_b` and has the face lifting [`DGKPEPFaceLift`](@ref).
"""
struct DGKennedyGruberKPEPFlux{FT}
    cp_d::FT
end
function (f::DGKennedyGruberKPEPFlux)(nvec_a, nvec_b, y_a, y_b)
    F = dg_kennedy_gruber_advective_flux(nvec_a, nvec_b, y_a, y_b)
    g =
        (y_a.ρ + y_b.ρ) / 2 * (
            (y_b.ψ - y_a.ψ) / 2 +
            f.cp_d * (y_a.θ + y_b.θ) / 2 * (y_b.Π - y_a.Π) / 2
        )
    return (;
        ρ = F.ρ,
        ρe = F.ρe,
        ρu_x = F.ρu_x + g * (y_a.x̂' * nvec_a),
        ρu_y = F.ρu_y + g * (y_a.ŷ' * nvec_a),
        ρu_z = F.ρu_z + g * (y_a.ẑ' * nvec_a),
    )
end

"""
    DGKPEPFaceLift(cp_d)

Face part of the two-point Exner force, a callable
`(normal, (ψ⁻, θ⁻, Π⁻, ρ⁻), (ψ⁺, θ⁺, Π⁺, ρ⁺))` applied on each side as a
lifting: `{ρ} [ (ψ⁺ − ψ⁻)/2 + c_pd {θ′} (Π⁺ − Π⁻)/2 ] n̂`, a `UVVector`.
"""
struct DGKPEPFaceLift{FT}
    cp_d::FT
end
(f::DGKPEPFaceLift)(normal, (ψ⁻, θ⁻, Π⁻, ρ⁻), (ψ⁺, θ⁺, Π⁺, ρ⁺)) =
    (
        (ρ⁻ + ρ⁺) / 2 *
        ((ψ⁺ - ψ⁻) / 2 + f.cp_d * (θ⁻ + θ⁺) / 2 * (Π⁺ - Π⁻) / 2)
    ) * normal

"""
    add_kpep_vertical_force!(ρuₕₜ, cp_d, ρ, ψ, θ, Π)

Subtract from `ρuₕₜ` the part of the two-point Exner force along `ξ³`, which
is horizontal on terrain-following levels:

    ρ / J ⋅ I_c( J (∇ξ³)ₕ [δψ + c_pd I_f(θ′) δΠ] ),

with the face differences `δ` and the interpolations `I_f`, `I_c`. With the
vertical mass flux `I_f(ρuₕ) ⋅ J∇ξ³` the geopotential part exchanges kinetic
and potential energy exactly, as the horizontal part does with `{ρ}{u_n}`.
The term vanishes on flat levels, at the hydrostatic reference and on a
uniform state; the boundary faces carry no force.
"""
function add_kpep_vertical_force!(ρuₕₜ, cp_d, ρ, ψ, θ, Π)
    FT = eltype(ρ)
    C3 = Geometry.Covariant3Vector
    ᶜinterp = Operators.InterpolateF2C()
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(C3(zero(FT))),
        top = Operators.SetGradient(C3(zero(FT))),
    )
    ᶜJ = Fields.local_geometry_field(axes(ρ)).J
    ᶠlg = Fields.local_geometry_field(Spaces.face_space(axes(ρ)))
    @. ρuₕₜ -=
        ρ / ᶜJ * ᶜinterp(
            ᶠlg.J * horizontal_part(
                ᶠgradᵥ(ψ) + cp_d * ᶠinterp(θ) * ᶠgradᵥ(Π),
                ᶠlg,
            ),
        )
    return ρuₕₜ
end
horizontal_part(v, lg) =
    Geometry.project(Geometry.UVAxis(), Geometry.LocalVector(v, lg))

##
## Interface fluxes
##

"""
    DGCentralMinusDissipation(central, dissipation)

Interface flux `central(normal, normal, y⁻, y⁺) − dissipation(normal, y⁻, y⁺) / 2`
over the slots `(ρ, ρe, ρu_x, ρu_y, ρu_z)`.
"""
struct DGCentralMinusDissipation{C, D}
    central::C
    dissipation::D
end
(f::DGCentralMinusDissipation)(normal, (y⁻,), (y⁺,)) = map(
    (F, D) -> F - D / 2,
    f.central(normal, normal, y⁻, y⁺),
    f.dissipation(normal, y⁻, y⁺),
)

"""
    dg_rusanov_dissipation(normal, y⁻, y⁺)

Rusanov dissipation `λ ⟦q⟧` of the conserved variables at `λ = max(λ⁻, λ⁺)`.
"""
function dg_rusanov_dissipation(normal, y⁻, y⁺)
    λ = max(y⁻.λ, y⁺.λ)
    return (;
        ρ = λ * (y⁺.ρ - y⁻.ρ),
        ρe = λ * (y⁺.ρe - y⁻.ρe),
        ρu_x = λ * (y⁺.ρ * y⁺.u_x - y⁻.ρ * y⁻.u_x),
        ρu_y = λ * (y⁺.ρ * y⁺.u_y - y⁻.ρ * y⁻.u_y),
        ρu_z = λ * (y⁺.ρ * y⁺.u_z - y⁻.ρ * y⁻.u_z),
    )
end

"""
    DGRoeDissipation(γ)

Roe dissipation, a callable `(normal, y⁻, y⁺)`: acoustic waves damped at
`|ûₙ ± ĉ|`, entropy and shear waves at `max(|ûₙ|, ĉ/20)`. The acoustic
amplitudes use the jump of `pm = p − p_ref`, so they vanish at the hydrostatic
reference.
"""
struct DGRoeDissipation{FT}
    γ::FT
end
function (d::DGRoeDissipation)(normal, y⁻, y⁺)
    γd = oftype(y⁻.ρ, d.γ)
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
    Δp = y⁺.pm - y⁻.pm
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
    return (; ρ = Dρ, ρe = Dρe, ρu_x = Dρu_x, ρu_y = Dρu_y, ρu_z = Dρu_z)
end
