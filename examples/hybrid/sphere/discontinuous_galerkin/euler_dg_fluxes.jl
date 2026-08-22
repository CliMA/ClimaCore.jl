#=
Equation-set-specific DG flux functions for the compressible-Euler examples:
Kennedy-Gruber two-point volume fluxes and Rusanov/Roe interface fluxes for
flux-form states with momentum in global Cartesian components (Souza et al.
2023, JAMES, doi:10.1029/2022MS003527), plus the (ρ, ρe) scalar subsystem,
the velocity-form fluctuation pair, and the pressure-hook
`EntropyConservingFlux`. These depend on the equation set (state field names,
thermodynamics, γ), so they live with the examples rather than in ClimaCore's
Operators module; the generic machinery and flux contracts they satisfy are
in src/Operators/numericalflux.jl and src/Operators/dg_fluxes.jl. In a model
that consumes ClimaCore (e.g. ClimaAtmos), the analogous fluxes belong to the
model, with pressure/sound speed from its own thermodynamics.

The `add_auto_broadcasters`/`unwrap` imports mirror the Operators module: on
the CPU path, flux arguments may arrive wrapped in `AutoBroadcaster`s.
=#

import ClimaCore: Geometry, Operators
import ClimaCore.Geometry: ⊗
import ClimaCore.Utilities: unwrap, add_auto_broadcasters
import ClimaCore.Operators: AbstractNumericalFlux
using LinearAlgebra: I

# Kinetic energy of a velocity AxisVector (2D UV / UW plane or 1D U).
@inline _specific_ke(u::Geometry.UVVector) = (u.u^2 + u.v^2) / 2
@inline _specific_ke(u::Geometry.UWVector) = (u.u^2 + u.w^2) / 2
@inline _specific_ke(u::Geometry.UVector) = (u.u^2) / 2

# Tangential unit for Roe shear wave: rotate n̂ 90° CCW in the horizontal
# plane. 1D `UVector` normals (extruded x–z hybrid) have no in-plane shear.
@inline _roe_tangent(n::Geometry.UVVector) = typeof(n)(-n.v, n.u)
@inline _roe_tangent(::Geometry.UVector) = nothing
@inline _roe_tangent(::Geometry.UWVector) = nothing

"""
    ideal_gas_pressure(state, params)

Default pressure for `EntropyConservingFlux`: `p = (γ-1)(ρe - ρKE)`.
"""
function ideal_gas_pressure(state, params)
    ρ, ρu, ρe = state.ρ, state.ρu, state.ρe
    u = ρu / ρ
    return (params.γ - 1) * (ρe - ρ * _specific_ke(u))
end

"""
    EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn[; pressure_fn, momentum_pressure_fn, roe_pressure_fn, sound_speed_fn])

Kennedy-Gruber kinetic energy preserving (KEP) interface flux with Roe entropy-stable
dissipation for compressible Euler equations, following Souza et al. (2023, JAMES, Eqs 40-42).

The central part uses arithmetic averages of primitive variables {ρ}, {u}, {p}, {e}, giving
the KEP property. The Roe dissipation uses a full characteristic decomposition of the Roe-averaged
Jacobian (4 waves in 2D: two acoustics, one entropy/contact, one shear).

  - `fluxfn(state, params...)`: physical flux tensor F(U)
  - `entropy_var_fn(state, params...)`: entropy variables v = ∂η/∂U (stored, not used in dissipation)
  - `roe_avg_fn(ρ⁻, ρ⁺, var⁻, var⁺)`: Roe-averaging function, e.g. density-weighted average
  - `pressure_fn(state, params...)`: thermodynamic pressure for enthalpy / energy (defaults to [`ideal_gas_pressure`](@ref))
  - `momentum_pressure_fn(state, params...)`: pressure in the K-G momentum flux (defaults to `pressure_fn`)
  - `roe_pressure_fn(state, params...)`: pressure in Roe wave amplitudes α₁, α₂, α₄ (defaults to `momentum_pressure_fn`, so stratified p′ formulations stay consistent with the volume flux)
  - `sound_speed_fn(state, params...)`: optional Roe sound speed; if `nothing`, uses `√((γ-1)(H̃-KẼ))`
"""
struct EntropyConservingFlux{F, V, A, P, MP, RP, S} <: AbstractNumericalFlux
    fluxfn::F
    entropy_var_fn::V
    roe_avg_fn::A
    pressure_fn::P
    momentum_pressure_fn::MP
    roe_pressure_fn::RP
    sound_speed_fn::S

    function EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn = ideal_gas_pressure,
        momentum_pressure_fn = nothing,
        roe_pressure_fn = nothing,
        sound_speed_fn = nothing,
    )
        F, V, A, P = typeof.((fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn))
        MP = momentum_pressure_fn === nothing ? pressure_fn : momentum_pressure_fn
        # Roe Δp must use the same pressure as the K-G / volume momentum flux (p′ for stratified).
        RP = roe_pressure_fn === nothing ? MP : roe_pressure_fn
        S = sound_speed_fn
        return new{F, V, A, P, typeof(MP), typeof(RP), typeof(S)}(
            fluxfn,
            entropy_var_fn,
            roe_avg_fn,
            pressure_fn,
            MP,
            RP,
            S,
        )
    end
end

# Positional `pressure_fn` for backward compatibility (e.g. Compressible Euler).
function EntropyConservingFlux(fluxfn, entropy_var_fn, roe_avg_fn, pressure_fn)
    return EntropyConservingFlux(
        fluxfn,
        entropy_var_fn,
        roe_avg_fn;
        pressure_fn,
    )
end

function (fn::EntropyConservingFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params = argvals⁻[2]

    ρ⁻, ρu⁻, ρe⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρe
    ρ⁺, ρu⁺, ρe⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρe

    u⁻ = ρu⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺
    γ = params.γ

    KE⁻ = _specific_ke(u⁻)
    KE⁺ = _specific_ke(u⁺)
    p⁻ = fn.pressure_fn(argvals⁻...)
    p⁺ = fn.pressure_fn(argvals⁺...)
    pm⁻ = fn.momentum_pressure_fn(argvals⁻...)
    pm⁺ = fn.momentum_pressure_fn(argvals⁺...)
    p_roe⁻ = fn.roe_pressure_fn(argvals⁻...)
    p_roe⁺ = fn.roe_pressure_fn(argvals⁺...)

    # Kennedy-Gruber KEP interface flux: arithmetic averages (Souza et al. 2023,
    # JAMES, Eqs 40–42 / App. A). Uses {ρ}{u}{p}{e} for the central flux.
    ρ̄ = (ρ⁻ + ρ⁺) / 2
    ū = (u⁻ + u⁺) / 2
    p̄ = (p⁻ + p⁺) / 2
    p̄m = (pm⁻ + pm⁺) / 2
    ē = (ρe⁻ / ρ⁻ + ρe⁺ / ρ⁺) / 2  # arithmetic mean of specific total energy

    Fc_ρ = (ρ̄ * ū)' * normal
    Fc_ρu = (ρ̄ * (ū ⊗ ū) + p̄m * I)' * normal
    Fc_ρe = (ū * (ρ̄ * ē + p̄))' * normal  # {u}({ρ}{e} + {p})

    # Roe-averaged state for compressible Euler (Roe 1981)
    # Guard non-positive densities at the face (does not floor prognostics).
    pos = ρ⁻ > 0 && ρ⁺ > 0
    ρ̃ = pos ? sqrt(ρ⁻ * ρ⁺) : abs(ρ̄)
    ũ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺) : ū
    H⁻ = (ρe⁻ + p⁻) / ρ⁻  # specific total enthalpy
    H⁺ = (ρe⁺ + p⁺) / ρ⁺
    H̃ = pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, H⁻, H⁺) : (H⁻ + H⁺) / 2
    KE_tilde = _specific_ke(ũ)
    c̃ = if fn.sound_speed_fn === nothing
        # Fall back to thermodynamic Roe c only when H̃ > KẼ.
        ΔH = H̃ - KE_tilde
        ΔH > 0 ? sqrt((γ - 1) * ΔH) : FT_zero(ΔH)
    else
        c⁻ = fn.sound_speed_fn(argvals⁻...)
        c⁺ = fn.sound_speed_fn(argvals⁺...)
        pos ? fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺) : max(c⁻, c⁺)
    end

    # Normal (and tangential, in 2D) directions.
    # Extruded 1D faces use `UVector` normals → no shear wave (Souza 1D Euler).
    ũₙ = ũ' * normal
    tang = _roe_tangent(normal)

    Δρ = ρ⁺ - ρ⁻
    Δu = u⁺ - u⁻
    Δuₙ = Δu' * normal
    # p′ jump for Roe amplitudes when momentum_pressure_fn = p′ (stratified)
    Δp = p_roe⁺ - p_roe⁻

    c̃⁻² = 1 / c̃^2
    α₁ = (Δp - ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # left-running acoustic
    α₂ = Δρ - Δp * c̃⁻²                        # entropy / contact
    α₄ = (Δp + ρ̃ * c̃ * Δuₙ) * 0.5 * c̃⁻²   # right-running acoustic

    λ₁ = abs(ũₙ - c̃)
    λ₂ = abs(ũₙ)
    λ₄ = abs(ũₙ + c̃)

    diss_ρ = λ₁ * α₁ + λ₂ * α₂ + λ₄ * α₄
    diss_ρu =
        (λ₁ * α₁) * (ũ - c̃ * normal) +
        (λ₂ * α₂) * ũ +
        (λ₄ * α₄) * (ũ + c̃ * normal)
    diss_ρe =
        λ₁ * α₁ * (H̃ - c̃ * ũₙ) +
        λ₂ * α₂ * KE_tilde +
        λ₄ * α₄ * (H̃ + c̃ * ũₙ)

    if tang !== nothing
        Δuₜ = Δu' * tang
        ũₜ = ũ' * tang
        α₃ = ρ̃ * Δuₜ                               # shear / vorticity
        diss_ρu = diss_ρu + (λ₂ * α₃) * tang
        diss_ρe = diss_ρe + λ₂ * α₃ * ũₜ
    end

    base = (
        ρ = Fc_ρ - diss_ρ / 2,
        ρu = Fc_ρu - diss_ρu / 2,
        ρe = Fc_ρe - diss_ρe / 2,
    )
    return merge(base, _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂))
end

@inline FT_zero(x) = zero(typeof(x))

# Handle passive tracer fields (ρθ) not part of the Euler entropy structure.
function _passive_tracer_fluxes(y⁻, y⁺, ū, normal, λ₂)
    nt⁻, nt⁺ = unwrap(y⁻), unwrap(y⁺)
    if !hasfield(typeof(nt⁻), :ρθ)
        return NamedTuple()
    end
    # Central advection + upwind dissipation for the passive tracer ρθ.
    Fc_ρθ = ((nt⁻.ρθ + nt⁺.ρθ) / 2 * ū)' * normal
    diss_ρθ = λ₂ * (nt⁺.ρθ - nt⁻.ρθ)
    return (ρθ = Fc_ρθ - diss_ρθ / 2,)
end

# ---------------------------------------------------------------------------
# Two-point (volume) and interface fluxes
# ---------------------------------------------------------------------------

"""
    kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the flux-form (ρ, ρe) subsystem (Souza et
al. 2023, JAMES, Eqs. 39 & 41): ``F_ρ = \\{ρ\\}\\{ũ\\}``,
``F_{ρe} = \\{ũ\\}(\\{ρ\\}\\{e\\} + \\{p\\})``, with `e` the specific total
energy and ``\\{ũ\\} = \\{u ⋅ nvec\\}`` the average of the **contravariant
nodal fluxes** (each node's velocity contracted with its own metric vector —
see [`add_flux_differencing_divergence!`](@ref) for why). Symmetric,
consistent, jointly linear in `(nvec_a, nvec_b)`.

State fields required: `ρ`, `ρe`, `e`, `p`, and `uv` (velocity in the local
orthonormal horizontal frame).
"""
function kennedy_gruber_scalars_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    return (ρ = ρ̄ * ūn, ρe = (ρ̄ * ē + p̄) * ūn)
end

"""
    kennedy_gruber_rusanov_scalars(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe) subsystem: [`kennedy_gruber_scalars_flux`](@ref)
as the central part plus a Rusanov penalty scaled by the state field `λ`
(the paper's interface choice, Souza et al. 2023).
"""
function kennedy_gruber_rusanov_scalars(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_scalars_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
    )
end

"""
    kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the full (ρ, ρe, ρu⃗) system with momentum
carried in GLOBAL CARTESIAN components (Souza et al. 2023): the basis is
constant, so component-wise flux differencing retains the KEP property with
no curvature source terms. Contravariant nodal fluxes are averaged (each
node's own metric vector).

State fields required: `ρ`, `e`, `p`, `uv` (velocity, local orthonormal
horizontal frame), `u1`, `u2`, `u3` (Cartesian velocity components), and
`E1`, `E2`, `E3` (the tangential projections of the Cartesian unit vectors
ê₁, ê₂, ê₃, each as a `UVVector` — position-dependent on the sphere but
state-independent). The pressure flux for component ``c`` is
``\\{p\\}\\{ê_c ⋅ nvec\\}``.
"""
function kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
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
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄ * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄ * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄ * Ē3n,
    )
end

"""
    kennedy_gruber_rusanov_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus a Rusanov
penalty scaled by the state field `λ` (jumps of the conserved variables;
momentum jumps via `ρ * u_c`). Additional state fields: `ρe`, `λ`.
"""
function kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    kennedy_gruber_roe_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus ROE-TYPE
wave-selective dissipation (Souza et al. 2023 interface choice): acoustic
waves are damped at ``|û_n ± ĉ|`` but entropy and shear jumps at
``max(|û_n|, ĉ/20)`` — so stationary balanced structure (contact/shear
jumps with ``u_n ≈ 0``) receives ~5% of Rusanov's uniform ``|u| + c``
dissipation (the Harten-type floor is required: see inline comment).
The energy eigen-component uses ``B = Ĥ - ĉ²/(γ-1)``, which absorbs the
geopotential and vertical-kinetic contributions of ``ρe`` without needing
them separately (Φ is single-valued at the face). Same state fields as
[`kennedy_gruber_rusanov_cartesian`](@ref); requires `γ` jumps consistent
with `p`/`e` (dry ideal gas).
"""
function kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    # keep the working precision of the state (Float32 fields stay Float32)
    γd = oftype(y⁻.ρ, γ_dry)
    # face normal in Cartesian components (E_c single-valued at the node)
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
    # Harten-type entropy floor on the contact/shear speed: pure |û_n|
    # leaves density jumps in near-stagnant columns (e.g. the model top)
    # undamped, and the min-ρ cell can drain unchecked (observed: secular
    # min-ρ collapse from day ~2.3 of a perturbed baroclinic wave at
    # zelem = 30). ε = 0.05 retains 5% of the Rusanov ρ-jump dissipation
    # while keeping the spurious forcing of balanced jets ~20× below
    # Rusanov. The acoustic speeds need no floor (|û_n| ≪ ĉ here).
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
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
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end
# dry-air ratio of specific heats used by the Roe linearization
const γ_dry = 7 / 5

"""
    kennedy_gruber_cartesian_advective_flux(nvec_a, nvec_b, y_a, y_b)

Advection-only variant of [`kennedy_gruber_cartesian_flux`](@ref): the momentum
flux omits the pressure term ``p̄ \\{ê_c ⋅ n\\}``, leaving the pure kinetic
Kennedy-Gruber flux ``ρ̄ ū_c ūn``. Used when the pressure-gradient force is
supplied separately in non-conservative (Exner-perturbation) form (Yatunin et
al. 2026): momentum conservation is traded for a well-balanced pressure
gradient, while the KEP property of the advective flux — and the mass and
energy (enthalpy) fluxes — are unchanged.
"""
function kennedy_gruber_cartesian_advective_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn,
        ρu2 = ρ̄ * ū2 * ūn,
        ρu3 = ρ̄ * ū3 * ūn,
    )
end

"""
    kennedy_gruber_rusanov_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`kennedy_gruber_rusanov_cartesian`](@ref): the
central part omits the momentum pressure flux (see
[`kennedy_gruber_cartesian_advective_flux`](@ref)); the Rusanov dissipation is
unchanged.
"""
function kennedy_gruber_rusanov_cartesian_advective(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_advective_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    kennedy_gruber_roe_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`kennedy_gruber_roe_cartesian`](@ref): the full
Roe flux minus its central momentum pressure term ``p̄ \\{ê_c ⋅ n\\}`` (mass,
energy and all wave-selective dissipation unchanged). `ê_c` is single-valued at
the shared node, so ``\\{ê_c ⋅ n\\} = ((E_c⁻ + E_c⁺)/2) ⋅ n``.
"""
function kennedy_gruber_roe_cartesian_advective(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    p̄ = (y⁻.p + y⁺.p) / 2
    return (
        ρ = F.ρ,
        ρe = F.ρe,
        ρu1 = F.ρu1 - p̄ * (((y⁻.E1 + y⁺.E1) / 2)' * normal),
        ρu2 = F.ρu2 - p̄ * (((y⁻.E2 + y⁺.E2) / 2)' * normal),
        ρu3 = F.ρu3 - p̄ * (((y⁻.E3 + y⁺.E3) / 2)' * normal),
    )
end

"""
    kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)

Non-symmetric two-point FLUCTUATION form for the advective operator
``(u·∇_h)u_c`` acting on velocity components in velocity (non-conservative)
form, driven by the Kennedy-Gruber mass flux:
``P^\\#_c(a, b) = F^\\#_ρ(a, b)\\,(u_{c,b} - u_{c,a})/2`` with
``F^\\#_ρ = \\{ρ\\}\\{u ⋅ nvec\\}`` (contravariant nodal fluxes averaged).

Pass to [`add_flux_differencing_divergence!`](@ref); the own-side boundary
lifts evaluate to zero (the jump vanishes for `y_a == y_b`), so the kernel
degenerates to the pure strong-form fluctuation sum. The mass-weighted
result divided by ``ρ\\,WJ`` is ``-(u·∇_h)u_c``, replacing BOTH the
relative-vorticity cross product and the horizontal-KE gradient of the
vector-invariant form.

KE compatibility with the KG mass flux (the fluctuation-form analog of the
KEP property): ``K_i F^\\#_ρ(i,j) + u_{c,i} P^\\#_c(i,j)`` (summed over
components) equals ``F^\\#_ρ(i,j)\\,(u_i · u_j)/2``, which is symmetric, so
the volume kinetic-energy production telescopes to face terms; complete them
with [`advective_fluctuation_lift`](@ref). The advected components must be
in a globally constant frame (e.g. Cartesian) — position-dependent frames
reintroduce curvature terms the jumps cannot see.

State fields required: `ρ`, `uv`, `u1`, `u2`, `u3`.
"""
function kg_massflux_fluctuation(nvec_a, nvec_b, y_a, y_b)
    F = ((y_a.ρ + y_b.ρ) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)
    return (
        u1 = F * (y_b.u1 - y_a.u1) / 2,
        u2 = F * (y_b.u2 - y_a.u2) / 2,
        u3 = F * (y_b.u3 - y_a.u3) / 2,
    )
end

"""
    advective_fluctuation_lift(normal, argvals⁻, argvals⁺)

Per-component face SAT completing [`kg_massflux_fluctuation`](@ref): the
EXACT velocity-variables transform of the flux-form KG central face
treatment, ``δu_c = (δ(ρ u_c) - u_c\\,δρ)/ρ`` applied to the central
interface fluxes — the own-flux terms cancel, leaving each side
``-\\{ρ\\}(\\{uv\\} ⋅ n̂_{side})\\,(u_c^{other} - u_c^{side})/2``.
Because it is the exact transform of a KE-consistent face treatment, the
face kinetic-energy bookkeeping is identical to the flux form's. (The same
transform of the Rusanov jumps reproduces the λ velocity-jump penalties
with a ``ρ^{other}/ρ^{side}`` weight; the plain penalties are their
constant-ρ limit and provide the face dissipation.) NOTE the sign: the
naive "central lifting" sign (+) is anti-consistent and exponentially
unstable at O(jump). Use through [`lifting_correction`](@ref) with argument
fields `(u_c, ρ, uv)`; divide the result by ``ρ`` for the velocity
tendency.
"""
function advective_fluctuation_lift(normal, argvals⁻, argvals⁺)
    u_c⁻, ρ⁻, uv⁻ = argvals⁻
    u_c⁺, ρ⁺, uv⁺ = argvals⁺
    F = ((ρ⁻ + ρ⁺) / 2) * (((uv⁻ + uv⁺) / 2)' * normal)
    return -F * (u_c⁺ - u_c⁻) / 2
end
