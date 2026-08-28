# Generic DG flux library: reference interface fluxes, face-lifting functions
# for non-conservative terms, and the LDG/SIPG Laplacian. Equation-set-specific
# fluxes (compressible Euler, Cartesian-momentum Kennedy-Gruber variants) are
# expected to live downstream of this module.

"""
    CentralNumericalFlux(fluxfn)

Evaluates the central numerical flux using `fluxfn`.
"""
struct CentralNumericalFlux{F} <: AbstractNumericalFlux
    fluxfn::F
end

function (fn::CentralNumericalFlux)(normal, argvals⁻, argvals⁺)
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    return ((F⁻ + F⁺) / 2)' * normal
end

"""
    RusanovNumericalFlux(fluxfn, wavespeedfn)

Evaluates the Rusanov numerical flux using `fluxfn` with wavespeed `wavespeedfn`
"""
struct RusanovNumericalFlux{F, W} <: AbstractNumericalFlux
    fluxfn::F
    wavespeedfn::W
end

function (fn::RusanovNumericalFlux)(normal, argvals⁻, argvals⁺)
    # AutoBroadcasters keep NamedTuple/vector arithmetic type-stable on GPU.
    y⁻ = add_auto_broadcasters(argvals⁻[1])
    y⁺ = add_auto_broadcasters(argvals⁺[1])
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    Favg = ((F⁻ + F⁺) / 2)' * normal
    return Favg + (λ / 2) * (y⁻ - y⁺)
end

# ---------------------------------------------------------------------------
# DG face-function library for non-conservative (vector-invariant) terms
# ---------------------------------------------------------------------------

"""
    central_gradient_lift(normal, (q⁻,), (q⁺,))

Symmetric central lifting completing the strong-form DG gradient of a scalar:
each side adds ``(q^* - q_{side}) n̂_{side}`` with central ``q^*``, i.e.
``((q⁺ - q⁻)/2)\\,n̂`` on the minus side. Use with
[`add_lifting_flux_internal!`](@ref) / [`lifting_correction`](@ref).
"""
central_gradient_lift(normal, (q⁻,), (q⁺,)) = ((q⁺ - q⁻) / 2) * normal

"""
    central_curl3_lift(normal, (u⁻, v⁻), (u⁺, v⁺))

Central lifting for the radial component of the horizontal curl:
``r̂ ⋅ (n̂ × (u^* - u_{side}))`` from the tangential jumps of the orthonormal
velocity components `(u, v)`.
"""
central_curl3_lift(normal, (u⁻, v⁻), (u⁺, v⁺)) =
    (
        normal.components.data.:1 * (v⁺ - v⁻) -
        normal.components.data.:2 * (u⁺ - u⁻)
    ) / 2

"""
    jump_penalty_lift(normal, (q⁻, λ⁻), (q⁺, λ⁺))

λ-scaled interface penalty: each side relaxes toward its neighbor at rate
``\\max(λ⁻, λ⁺)/2``.
"""
jump_penalty_lift(normal, (q⁻, λ⁻), (q⁺, λ⁺)) = max(λ⁻, λ⁺) / 2 * (q⁺ - q⁻)

# ---------------------------------------------------------------------------
# LDG / interior-penalty Laplacian face fluxes
# Volume term: WJ-weighted κ∇² via (−WJ)·κ·(−wdiv(G)); face −{{κG}}·n + τ[[q]]
# with G = ∇q.
# ---------------------------------------------------------------------------

"""
    ldg_laplacian_tendency(q, ρ_weight, κ, τ)

WJ-normalized interior-penalty Laplacian tendency approximating
``κ ∇⋅(ρ_{weight} ∇q)`` (or ``κ ∇²q`` when `ρ_weight === nothing`): weak-form
volume term plus the consistent numerical flux
``−\\{\\!\\{κ G\\}\\!\\}·n̂ + τ [\\![q]\\!]`` with ``G = ρ_{weight} ∇q``
(or ``G = ∇q``). See [`LDGLaplacianFlux`](@ref) and
[`ldg_penalty_parameter`](@ref).
"""
function ldg_laplacian_tendency(q, ρ_weight, κ, τ)
    wdiv = WeakDivergence()
    grad = Gradient()
    lgeom = Fields.local_geometry_field(axes(q))
    residual = similar(q)
    G = ρ_weight === nothing ? (@. grad(q)) : (@. ρ_weight * grad(q))
    @. residual = (-lgeom.WJ) * κ * (-wdiv(G))
    # Face normals are UVVector; raise G to the same basis for G·n̂.
    G_uv = @. Geometry.UVVector(G)
    add_ldg_laplacian_flux_internal!(residual, q, G_uv, κ, τ)
    return residual ./ lgeom.WJ
end

"""
    ldg_penalty_parameter(κ, space)

Interior-penalty scaling ``τ = κ (2N_q − 1)^2 / h`` using the horizontal
spectral-element length scale (works for extruded hybrid spaces).
"""
function ldg_penalty_parameter(κ, space)
    hspace =
        space isa Spaces.ExtrudedFiniteDifferenceSpace ?
        Spaces.horizontal_space(space) : space
    h = Spaces.node_horizontal_length_scale(hspace)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
    return κ * (2 * Nq - 1)^2 / h
end

"""
    LDGLaplacianFlux(τ)

Consistent interior-penalty flux for the LDG/SIPG Laplacian. Called through
[`add_numerical_flux_internal!`](@ref) on a WJ-weighted residual of
``−∇·F`` with ``F = −κ G`` and ``G = ∇q`` (or ``ρ_{weight} ∇q``). Arguments
are `(q, G, κ)` on each side, where `G` must share the face-normal
basis (typically [`Geometry.UVVector`](@ref)); returns
``−\\{\\!\\{κ G\\}\\!\\}·n̂ + τ[[q]]`` with ``[[q]] = q⁻ − q⁺``.
"""
struct LDGLaplacianFlux{T} <: AbstractNumericalFlux
    τ::T
end

function (fn::LDGLaplacianFlux)(normal, argvals⁻, argvals⁺)
    q⁻, G⁻, κ⁻ = argvals⁻[1], argvals⁻[2], argvals⁻[3]
    q⁺, G⁺, κ⁺ = argvals⁺[1], argvals⁺[2], argvals⁺[3]
    Favg = -((κ⁻ * G⁻ + κ⁺ * G⁺) / 2)' * normal
    return Favg + fn.τ * (q⁻ - q⁺)
end

"""
    add_ldg_laplacian_flux_internal!(dydt, q, G, κ, τ)

Add consistent LDG/SIPG face coupling
``−\\{\\!\\{κ G\\}\\!\\}·n̂ + τ[[q]]`` to a WJ-weighted Laplacian residual.
Accepts a shared [`start_dg_ghost_exchange`](@ref) handle started on
`(q, G, κ)`.
"""
add_ldg_laplacian_flux_internal!(dydt, q, G, κ, τ; ghost_exchange = nothing) =
    add_numerical_flux_internal!(
        LDGLaplacianFlux(τ),
        dydt,
        q,
        G,
        κ;
        ghost_exchange,
    )

# ===========================================================================
# Flux families grafted from as/moisture-0M (numericalflux.jl) during the
# origin/ts/mpi merge. ts/mpi split DG code into numericalflux.jl (infra) +
# dg_fluxes.jl (fluxes) and carried only Central/Rusanov/LDG; these are the
# moisture-0M flux families (Roe/EC + kennedy_gruber/ranocha/waruszewski/es/
# lmars/tracer/advective + entropy & KE helpers). Calling conventions match
# ts/mpi's face loops (fn(normal, argvals⁻, argvals⁺); two-point fn2pt).
# ===========================================================================

# --- Roe numerical flux (struct) ---
"""
    RoeNumericalFlux(fluxfn, roe_avg_fn)

Evaluates the Roe numerical flux using `fluxfn` and Roe-averaging function `roe_avg_fn`.

The Roe flux computes a central flux plus an entropy-stable dissipation term based on
the characteristic decomposition of the jump in conserved variables.
"""
struct RoeNumericalFlux{F, A} <: AbstractNumericalFlux
    fluxfn::F
    roe_avg_fn::A
end

function (fn::RoeNumericalFlux)(normal, argvals⁻, argvals⁺)
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    params⁻ = argvals⁻[2]
    params⁺ = argvals⁺[2]

    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    Favg = (F⁻ + F⁺) / 2

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * normal

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * normal

    λ = sqrt(params⁻.g)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    ρ̄ = sqrt(ρ⁻ * ρ⁺)
    ū = fn.roe_avg_fn(ρ⁻, ρ⁺, u⁻, u⁺)
    θ̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c̄ = fn.roe_avg_fn(ρ⁻, ρ⁺, c⁻, c⁺)

    ūₙ = ū' * normal

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * normal

    c̄⁻² = 1 / c̄^2
    w1 = abs(ūₙ - c̄) * (Δp - ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w2 = abs(ūₙ + c̄) * (Δp + ρ̄ * c̄ * Δuₙ) * 0.5 * c̄⁻²
    w3 = abs(ūₙ) * (Δρ - Δp * c̄⁻²)
    w4 = abs(ūₙ) * ρ̄
    w5 = abs(ūₙ) * (Δρθ - θ̄ * Δp * c̄⁻²)

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (ū - c̄ * normal) + w2 * (ū + c̄ * normal) + w3 * ū + w4 * (Δu - Δuₙ * normal)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ̄ + w5) * 0.5

    return (
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal - fluxᵀn_ρ,
        ρu = ((F⁻.ρu + F⁺.ρu) / 2)' * normal - fluxᵀn_ρu,
        ρθ = ((F⁻.ρθ + F⁺.ρθ) / 2)' * normal - fluxᵀn_ρθ,
    )
end


# --- KE/Roe helpers, ideal-gas pressure, entropy-conserving flux, passive tracer ---
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
    kennedy_gruber_tracer_flux(nvec_a, nvec_b, y_a, y_b)

Two-point flux for a passive tracer ``ρq`` advected by the SAME Kennedy-Gruber
mass flux as continuity: ``F_{ρq} = \\{ρ\\}\\{ũ\\}\\{q\\}`` (the mass flux
``\\{ρ\\}\\{ũ\\}`` times the arithmetic-mean specific tracer ``\\{q\\}``). This
is free-stream-preserving for the tracer — with ``q`` uniform, ``F_{ρq} = q F_ρ``
so the tracer equation reduces to ``q``×continuity and a constant ``q`` stays
constant. State fields required: `ρ`, `uv`, `q` (specific tracer, e.g. total
specific humidity `q_tot`).
"""
function kennedy_gruber_tracer_flux(nvec_a, nvec_b, y_a, y_b)
    Fρ = ((y_a.ρ + y_b.ρ) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)
    q̄ = (y_a.q + y_b.q) / 2
    return (ρq = Fρ * q̄,)
end

"""
    kennedy_gruber_rusanov_tracer(normal, argvals⁻, argvals⁺)

Interface flux for a passive tracer: [`kennedy_gruber_tracer_flux`](@ref) central
part plus a Rusanov penalty on the conserved tracer jump ``⟦ρq⟧`` scaled by the
state field `λ`. State fields: `ρ`, `uv`, `q`, `λ` (and `ρq = ρ·q`).
"""
function kennedy_gruber_rusanov_tracer(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_tracer_flux(normal, normal, y⁻, y⁺)
    return (ρq = F.ρq - λ / 2 * (y⁺.ρ * y⁺.q - y⁻.ρ * y⁻.q),)
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
    # Momentum pressure: `pm` = p (full conservative) or p' = p − p_ref
    # (stratified conservative, well-balanced over topography). Energy keeps
    # the full thermodynamic p in the enthalpy flux.
    p̄m = (y_a.pm + y_b.pm) / 2
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
        ρu1 = ρ̄ * ū1 * ūn + p̄m * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄m * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄m * Ē3n,
    )
end

"""
    ln_mean(x, y)

Numerically-stable logarithmic mean ``(x-y)/(\\log x - \\log y)`` (Ismail & Roe
2009): switches to the convergent Taylor series in ``f^2=((x-y)/(x+y))^2`` when
``x≈y`` to avoid the ``0/0`` cancellation. The log mean is the building block of
entropy-conservative fluxes (it is what makes ``⟦w⟧·F^\\# = ⟦ψ⟧`` hold exactly).
"""
@inline function ln_mean(x, y)
    ε = oftype(x, 1e-4)
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)  # ((x−y)/(x+y))²
    return f² < ε ?
           (x + y) / (2 + f² * (2 / 3 + f² * (2 / 5 + f² * 2 / 7))) :
           (y - x) / log(y / x)
end

"""
    ranocha_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Ranocha (2018, 2020) two-point flux for the (ρ, ρe, ρu⃗) system in GLOBAL
CARTESIAN momentum components — the *entropy-conservative* counterpart of
[`kennedy_gruber_cartesian_flux`](@ref). Unlike Kennedy-Gruber (which is only
kinetic-energy- and pressure-equilibrium-preserving), the Ranocha flux is
SIMULTANEOUSLY entropy-conservative (Tadmor `⟦w⟧·F# = ⟦ψ⟧`), kinetic-energy-
preserving, and pressure-equilibrium-preserving, so — paired with an
entropy-dissipative interface — it yields a discrete entropy inequality that
Kennedy-Gruber cannot.

It differs from KG in three places: the mass flux uses the logarithmic mean
``ρ^{ln}`` instead of ``ρ̄``; the internal energy uses ``1/((γ-1)(ρ/p)^{ln})``;
and the pressure-work uses the cross term ``½(p_a u_{n,b}+p_b u_{n,a})`` rather
than ``p̄ ū_n``. The kinetic part is the KEP cross term ``½\\,u_a·u_b``. The
geopotential (``Φ = e - e_{int} - K``, single-valued at a shared node, varying
horizontally only over terrain) is advected as a passive potential ``ρ^{ln}
ū_n\\,\\{Φ\\}``. Momentum pressure uses `pm` (= p, or p' for the stratified /
well-balanced split) exactly as KG, so it drops into the same volume-flux slot
and inherits the same reference-deviation well-balancedness. Consistency check:
for `y_a == y_b` it collapses to the physical fluxes ``ρu_n``,
``(ρe+p)u_n``, ``ρu_c u_n + pm\\,ê_c·n``. Same state fields as
[`kennedy_gruber_cartesian_flux`](@ref).
"""
function ranocha_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ_dry)
    ρln = ln_mean(y_a.ρ, y_b.ρ)
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn                                   # entropy-consistent mass flux
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p̄m = (y_a.pm + y_b.pm) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    # internal energy: 1/((γ−1)(ρ/p)^ln); KEP kinetic cross term; pressure work
    e_int = 1 / (ln_mean(y_a.ρ / y_a.p, y_b.ρ / y_b.p) * (γd - 1))
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    una = y_a.uv' * nvec_a
    unb = y_b.uv' * nvec_b
    pv = (y_a.p * unb + y_b.p * una) / 2            # ½(p_a u_{n,b}+p_b u_{n,a})
    # geopotential per node (Φ = e − e_int − K), advected as a passive potential
    Φa = y_a.e - y_a.p / ((γd - 1) * y_a.ρ) - (y_a.u1^2 + y_a.u2^2 + y_a.u3^2) / 2
    Φb = y_b.e - y_b.p / ((γd - 1) * y_b.ρ) - (y_b.u1^2 + y_b.u2^2 + y_b.u3^2) / 2
    Φ̄ = (Φa + Φb) / 2
    return (
        ρ = mn,
        ρe = mn * (K̃ + e_int + Φ̄) + pv,
        ρu1 = mn * ū1 + p̄m * Ē1n,
        ρu2 = mn * ū2 + p̄m * Ē2n,
        ρu3 = mn * ū3 + p̄m * Ē3n,
    )
end

"""
    waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Waruszewski et al. (2022, JCP 468:111507) entropy-conservative + WELL-BALANCED
two-point flux for the (ρ, ρe, ρu⃗) system WITH GRAVITY, in global Cartesian
momentum components. This is the only flux here that is EC *and* machine-precision
well-balanced over terrain SIMULTANEOUSLY: the geopotential is handled by a
non-conservative fluctuation term ``½ρ̂⟦φ⟧`` in the momentum flux — NOT by a
reference split. It satisfies the generalized (non-conservative) Tadmor condition
``β⁻·D(a;b) − β⁺·D(b;a) = ⟦u_kη⟧`` with the geopotential-augmented entropy
variables (β₁ carries the ``+2φb`` term; see [`entropy_variables`](@ref)).

Differs from Ranocha: the EC pressure is Chandrashekar's ``p* = {{ρ}}/(2{{b}})``,
``b = ρ/(2p)`` (not ``{{p}}``); the internal energy uses the log-mean of ``b``;
and the momentum pressure slot is ``p* + ½ρ̂⟦φ⟧`` with ``ρ̂ = {{b}}{{ρ}}_log/b⁻``
(NON-symmetric — uses the own/self state ``b⁻``, which is well-defined here since
the kernel passes the self node first). Verified: at ``y_a=y_b`` it reduces to the
physical fluxes, and the Tadmor residual over a geopotential jump is ~1e-15.

Hybrid adaptation: the horizontal DG advects only the horizontal momentum, so the
vertical kinetic energy ``w_c²/2`` rides as a passive potential bundled with ``φ``
in ``e*`` (via ``Ψ = e − e_int − K_h``), while the gravity fluctuation uses the
geopotential ``φ`` alone (state field `φ`). State fields: `ρ`, `e`, `p`, `uv`,
`u1`,`u2`,`u3`, `E1`,`E2`,`E3`, `φ`.
"""
function waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ_dry)
    ba = y_a.ρ / (2 * y_a.p)                         # inverse temperature b⁻ (self)
    bb = y_b.ρ / (2 * y_b.p)
    ρln = ln_mean(y_a.ρ, y_b.ρ)
    bln = ln_mean(ba, bb)
    b̄ = (ba + bb) / 2
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn                                    # (ρuₖ)* = ρ^ln {{u}}
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p_star = ρ̄ / (2 * b̄)                             # Chandrashekar p* = {{ρ}}/2{{b}}
    ρ̂ = b̄ * ρln / ba                                # NON-symmetric (self b⁻)
    jφ = y_b.φ - y_a.φ                               # ⟦φ⟧
    pgrav = p_star + ρ̂ * jφ / 2                      # momentum pressure slot
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    # internal energy log-mean 1/(2(γ−1)b^ln); horizontal KEP kinetic cross term;
    # passive potential Ψ = φ + w_c²/2 = e − e_int − K_h (advected like {{φ}}).
    e_int = 1 / (2 * (γd - 1) * bln)
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    Ψa = y_a.e - y_a.p / ((γd - 1) * y_a.ρ) - (y_a.u1^2 + y_a.u2^2 + y_a.u3^2) / 2
    Ψb = y_b.e - y_b.p / ((γd - 1) * y_b.ρ) - (y_b.u1^2 + y_b.u2^2 + y_b.u3^2) / 2
    e_star = e_int + (Ψa + Ψb) / 2 + K̃
    return (
        ρ = mn,
        ρe = e_star * mn + ūn * p_star,
        ρu1 = mn * ū1 + pgrav * Ē1n,
        ρu2 = mn * ū2 + pgrav * Ē2n,
        ρu3 = mn * ū3 + pgrav * Ē3n,
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
    # jumps and wave amplitudes. The pressure jump uses the momentum pressure
    # `pm` (= p for full conservative, = p' for stratified) so the acoustic
    # amplitudes vanish at rest even over topography. (The entropy amplitude α₀
    # still uses the full Δρ, so stratified Roe leaves an O(Δρ_ref) contact-wave
    # residual over terrain — stable, not machine-precision; LMARS avoids it.)
    Δρ = y⁺.ρ - y⁻.ρ
    Δp = y⁺.pm - y⁻.pm
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

"""
    ranocha_rusanov_cartesian(normal, argvals⁻, argvals⁺)
    ranocha_roe_cartesian(normal, argvals⁻, argvals⁺)

Entropy-stable interface fluxes: the entropy-conservative
[`ranocha_cartesian_flux`](@ref) central part plus the same Rusanov / Roe
dissipation used by the Kennedy-Gruber interfaces. The dissipation is recovered
as ``(F_{diss} - F_{KG,central})`` (a cheap extra KG eval) and added to the
Ranocha central flux, so the tested wave-selective penalties are reused verbatim
while the volume/interface central pair is now entropy-conservative. Paired with
[`ranocha_cartesian_flux`](@ref) as the volume flux this gives an EC-volume +
dissipative-interface scheme — the ingredient Kennedy-Gruber lacks for a discrete
entropy inequality. (The dissipation is in conserved, not entropy, variables, so
this is entropy-stable in the sense of an EC volume flux + a positive dissipation,
not a certified entropy-variable dissipation matrix.)
"""
function ranocha_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    Fr = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fr.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fr.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fr.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fr.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fr.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function ranocha_roe_cartesian(normal, (y⁻,), (y⁺,))
    Fr = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fr.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fr.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fr.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fr.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fr.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

"""
    waruszewski_rusanov_cartesian(normal, argvals⁻, argvals⁺)
    waruszewski_roe_cartesian(normal, argvals⁻, argvals⁺)
    waruszewski_es_cartesian(normal, argvals⁻, argvals⁺)

Interface fluxes pairing the well-balanced entropy-conservative
[`waruszewski_cartesian_flux`](@ref) central part with Rusanov / Roe / entropy-
variable ([`entropy_stable_dissipation`](@ref)) dissipation. The dissipation is
recovered as ``(F_{diss} − F_{KG,central})`` (a cheap KG eval) so the tested
penalties are reused verbatim; the WB-EC central flux carries the pressure and
gravity. With the entropy-variable (`es`) dissipation this is the genuinely
entropy-stable AND well-balanced-over-terrain scheme.
"""
function waruszewski_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fw.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fw.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fw.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fw.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fw.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function waruszewski_roe_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fw.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fw.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fw.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fw.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fw.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function waruszewski_es_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = Fw.ρ - D.ρ,
        ρe = Fw.ρe - D.ρe,
        ρu1 = Fw.ρu1 - D.ρu1,
        ρu2 = Fw.ρu2 - D.ρu2,
        ρu3 = Fw.ρu3 - D.ρu3,
    )
end
# dry-air ratio of specific heats used by the Roe linearization
const γ_dry = 7 / 5

"""
    entropy_variables(ρ, u1, u2, u3, p)

Entropy variables ``w = ∂S/∂U`` for the ideal-gas Euler system with the
mathematical (convex) entropy ``S = -ρs/(γ-1)``, ``s = \\log p - γ\\log ρ``
(thermal frame). With ``β = ρ/(2p)``,

    w = ((γ-s)/(γ-1) - β|u|²,  2βu1,  2βu2,  2βu3,  -2β).

Additive constants in `s` drop under the jump `⟦w⟧`, so they are irrelevant to
the dissipation built from these.
"""
@inline function entropy_variables(ρ, u1, u2, u3, p)
    γd = oftype(ρ, γ_dry)
    β = ρ / (2 * p)
    s = log(p) - γd * log(ρ)
    wρ = (γd - s) / (γd - 1) - β * (u1^2 + u2^2 + u3^2)
    return (wρ, 2 * β * u1, 2 * β * u2, 2 * β * u3, -2 * β)
end

"""
    entropy_stable_dissipation(y⁻, y⁺)

Lax-Friedrichs dissipation in ENTROPY variables, ``½ λ Ĥ ⟦w⟧``, where
``Ĥ = ∂U/∂w`` is the (symmetric positive-definite) entropy Jacobian at the
arithmetic-mean state and ``λ = \\max(|u|+c)``. Because `Ĥ` is SPD,
``⟦w⟧·(Ĥ⟦w⟧) ≥ 0``, so subtracting this from ANY entropy-conservative
([`ranocha_cartesian_flux`](@ref)) or kinetic-energy-preserving
([`kennedy_gruber_cartesian_flux`](@ref)) central flux gives a discrete entropy
inequality (entropy stability) — the guarantee that conserved-variable
Rusanov/Roe penalties do not provide. To leading order `Ĥ⟦w⟧ = ⟦U⟧`, so this is
an entropy-consistent Rusanov. The geopotential (single-valued at the shared
node, `⟦Φ⟧ = 0`) is handled by forming `Ĥ⟦w⟧` in the thermal frame and shifting
the energy component by `Φ·(mass dissipation)` — an identity-preserving change of
variables. Returns the conserved-variable dissipation `(ρ, ρe, ρu1, ρu2, ρu3)`.
The `Ĥ = ∂U/∂w` form is verified numerically (symmetry, SPD, `Ĥ·(∂w/∂U)=I`).
"""
@inline function entropy_stable_dissipation(y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ_dry)
    w⁻ = entropy_variables(y⁻.ρ, y⁻.u1, y⁻.u2, y⁻.u3, y⁻.p)
    w⁺ = entropy_variables(y⁺.ρ, y⁺.u1, y⁺.u2, y⁺.u3, y⁺.p)
    v1 = w⁺[1] - w⁻[1]
    v2 = w⁺[2] - w⁻[2]
    v3 = w⁺[3] - w⁻[3]
    v4 = w⁺[4] - w⁻[4]
    v5 = w⁺[5] - w⁻[5]
    # arithmetic-mean state for Ĥ = ∂U/∂w
    ρ = (y⁻.ρ + y⁺.ρ) / 2
    u1 = (y⁻.u1 + y⁺.u1) / 2
    u2 = (y⁻.u2 + y⁺.u2) / 2
    u3 = (y⁻.u3 + y⁺.u3) / 2
    p = (y⁻.p + y⁺.p) / 2
    k = (u1^2 + u2^2 + u3^2) / 2
    E = p / ((γd - 1) * ρ) + k            # thermal total energy per mass
    H = E + p / ρ                         # thermal enthalpy per mass
    c2 = γd * p / ρ
    # Ĥ v (thermal frame), Ĥ = ∂U/∂w SPD
    HvR = ρ * v1 + ρ * u1 * v2 + ρ * u2 * v3 + ρ * u3 * v4 + ρ * E * v5
    Hv1 =
        ρ * u1 * v1 + (ρ * u1^2 + p) * v2 + ρ * u1 * u2 * v3 +
        ρ * u1 * u3 * v4 + ρ * u1 * H * v5
    Hv2 =
        ρ * u2 * v1 + ρ * u1 * u2 * v2 + (ρ * u2^2 + p) * v3 +
        ρ * u2 * u3 * v4 + ρ * u2 * H * v5
    Hv3 =
        ρ * u3 * v1 + ρ * u1 * u3 * v2 + ρ * u2 * u3 * v3 +
        (ρ * u3^2 + p) * v4 + ρ * u3 * H * v5
    HvE =
        ρ * E * v1 + ρ * u1 * H * v2 + ρ * u2 * H * v3 + ρ * u3 * H * v4 +
        (ρ * H^2 - c2 * p / (γd - 1)) * v5
    λ = max(y⁻.λ, y⁺.λ)
    # geopotential (single-valued at the node ⇒ Φ⁻ = Φ⁺); shift thermal→total
    Φ = y⁻.e - y⁻.p / ((γd - 1) * y⁻.ρ) - (y⁻.u1^2 + y⁻.u2^2 + y⁻.u3^2) / 2
    half = λ / 2
    Dρ = half * HvR
    return (
        ρ = Dρ,
        ρe = half * HvE + Φ * Dρ,
        ρu1 = half * Hv1,
        ρu2 = half * Hv2,
        ρu3 = half * Hv3,
    )
end

"""
    kennedy_gruber_es_cartesian(normal, argvals⁻, argvals⁺)
    ranocha_es_cartesian(normal, argvals⁻, argvals⁺)

Entropy-stable interface fluxes: a central two-point flux (Kennedy-Gruber or
Ranocha) minus [`entropy_stable_dissipation`](@ref) (dissipation in the entropy
variables). With the Ranocha EC central flux this is a genuinely entropy-stable
scheme (discrete `dS/dt ≤` boundary); with the KG (KEP, not EC) central flux the
dissipation is still entropy-decreasing but the KG volume error remains. Both
share the identical dissipation, so the penalty is decoupled from the choice of
central flux.
"""
function kennedy_gruber_es_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ,
        ρe = F.ρe - D.ρe,
        ρu1 = F.ρu1 - D.ρu1,
        ρu2 = F.ρu2 - D.ρu2,
        ρu3 = F.ρu3 - D.ρu3,
    )
end

function ranocha_es_cartesian(normal, (y⁻,), (y⁺,))
    F = ranocha_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ,
        ρe = F.ρe - D.ρe,
        ρu1 = F.ρu1 - D.ρu1,
        ρu2 = F.ρu2 - D.ρu2,
        ρu3 = F.ρu3 - D.ρu3,
    )
end

"""
    lmars_cartesian(normal, argvals⁻, argvals⁺)

Low-Mach Approximate Riemann Solver (LMARS; Chen et al. 2013, the FV3 flux) for
the conservative (ρ, ρe, ρu⃗-Cartesian) system. A two-wave acoustic Riemann
solve gives an interface normal velocity and pressure from the reference
impedance ``C = ρ̄ ĉ`` (ĉ = mean of a state-provided, floorable sound speed
`c`):

    u* = ½(uₙ⁻+uₙ⁺) − (p⁺−p⁻)/(2C),   p* = ½(p⁻+p⁺) − ½C(uₙ⁺−uₙ⁻),

then every advected quantity is upwinded at `u*` (flow speed, NOT `|u|+c`), so
acoustic dissipation scales with the impedance `C` while advective dissipation
scales with `|u*|` — wave-selective like Roe, but with no eigen-decomposition
and no `sqrt(γp/ρ)` (robust where `p` dips negative). State fields: `ρ`, `ρe`,
`p`, `u1`,`u2`,`u3` (Cartesian velocity), `E1`,`E2`,`E3` (Cartesian projections
of the face normal, single-valued at the node), and `c` (sound speed). It is a
complete numerical flux (no separate central+penalty), consistent with the
Kennedy-Gruber volume flux `kennedy_gruber_cartesian_flux`.
"""
function lmars_cartesian(normal, (y⁻,), (y⁺,))
    # face normal in Cartesian components (ê_c single-valued at the node)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    # Acoustic solve on the momentum pressure `pm` (= p full / p' stratified) so
    # u*, p* vanish at rest even over topography; enthalpy below keeps full p.
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    pstar = (y⁻.pm + y⁺.pm) / 2 - C * (unR - unL) / 2
    # upwind (branchless) the advected quantities at u*
    pos = ustar >= 0
    ρup = ifelse(pos, y⁻.ρ, y⁺.ρ)
    ρeup = ifelse(pos, y⁻.ρe, y⁺.ρe)
    pup = ifelse(pos, y⁻.p, y⁺.p)
    u1up = ifelse(pos, y⁻.u1, y⁺.u1)
    u2up = ifelse(pos, y⁻.u2, y⁺.u2)
    u3up = ifelse(pos, y⁻.u3, y⁺.u3)
    return (
        ρ = ustar * ρup,
        ρe = ustar * (ρeup + pup),                 # enthalpy flux (full p)
        ρu1 = ustar * (ρup * u1up) + pstar * n1,
        ρu2 = ustar * (ρup * u2up) + pstar * n2,
        ρu3 = ustar * (ρup * u3up) + pstar * n3,
    )
end

"""
    lmars_tracer(normal, argvals⁻, argvals⁺)

Interface flux for a passive tracer `ρq` that is **consistent with the LMARS mass
flux**: the tracer is upwinded at the SAME low-Mach contact velocity `u*` that
[`lmars_cartesian`](@ref) uses for continuity/momentum, so a uniform `q` reproduces
`q·(u*·ρ_up)` = `q` × the LMARS continuity flux (free-stream / constancy
preserving). Use this for `ρq_tot` whenever the dynamics use `INTERFACE_FLUX=lmars`,
so mass and tracer share one interface velocity (a `kennedy_gruber_rusanov_tracer`
here would advect moisture with a *different* mass flux and inject spurious tracer).
`u*` is `√(γp/ρ)`-free and vanishes at rest over terrain. State fields required:
`ρ`, `c`, `pm`, `u1/u2/u3`, `E1/E2/E3`, `q`.
"""
function lmars_tracer(normal, (y⁻,), (y⁺,))
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    ρqup = ifelse(ustar >= 0, y⁻.ρ * y⁻.q, y⁺.ρ * y⁺.q)
    return (ρq = ustar * ρqup,)
end

"""
    lmars_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`lmars_cartesian`](@ref): keeps LMARS's low-Mach
contact velocity `u* = ½(uₙ⁻+uₙ⁺) − (pm⁺−pm⁻)/(2C)` and upwinds the advected
quantities (`ρ`, `ρe`, `ρu_c`) at `u*`, but OMITS the conservative pressure flux
`p* n`. Used with a non-conservative (Exner-perturbation) pressure-gradient force
(`kennedy_gruber_cartesian_advective_flux` volume flux): the interface supplies
LMARS's wave-selective, `sqrt(γp/ρ)`-free advective dissipation (impedance
`C = ρ̄ĉ`, `ĉ = √(γR_d T_ref)`) while the PGF is handled separately, exactly as
the Roe/Rusanov advective counterparts. Well-balanced: at a shared node `pm⁻=pm⁺`
at rest ⇒ `u*=0` ⇒ zero interface flux.
"""
function lmars_cartesian_advective(normal, (y⁻,), (y⁺,))
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    unL = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    unR = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    C = (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2      # reference impedance ρ̄ĉ
    ustar = (unL + unR) / 2 - (y⁺.pm - y⁻.pm) / (2 * C)
    pos = ustar >= 0
    ρup = ifelse(pos, y⁻.ρ, y⁺.ρ)
    ρeup = ifelse(pos, y⁻.ρe, y⁺.ρe)
    pup = ifelse(pos, y⁻.p, y⁺.p)
    u1up = ifelse(pos, y⁻.u1, y⁺.u1)
    u2up = ifelse(pos, y⁻.u2, y⁺.u2)
    u3up = ifelse(pos, y⁻.u3, y⁺.u3)
    return (
        ρ = ustar * ρup,
        ρe = ustar * (ρeup + pup),                 # enthalpy flux (full p)
        ρu1 = ustar * (ρup * u1up),                # NO pressure flux (Exner PGF)
        ρu2 = ustar * (ρup * u2up),
        ρu3 = ustar * (ρup * u3up),
    )
end

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

# Harten–Lax–van Leer (HLL, 1983) two-wave approximate Riemann solver.
# Davis (1988) signal-speed estimates S_L = min(uₙ⁻−c⁻, uₙ⁺−c⁺),
# S_R = max(uₙ⁻+c⁻, uₙ⁺+c⁺) (c = √(γp/ρ)). Well-posed here: S_R − S_L ≥
# 2·min(c⁻,c⁺) > 0 whenever p > 0 (the Zhang–Shu limiter guarantees it), so the
# middle-state division never blows up and the 3-way select can be a branchless
# `ifelse` (GPU-safe — both arms always evaluate).
@inline function _hll_signal_speeds(normal, y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ_dry)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    un⁻ = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    un⁺ = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    c⁻ = sqrt(γd * y⁻.p / y⁻.ρ)
    c⁺ = sqrt(γd * y⁺.p / y⁺.ρ)
    S_L = min(un⁻ - c⁻, un⁺ - c⁺)
    S_R = max(un⁻ + c⁻, un⁺ + c⁺)
    return S_L, S_R
end

# One HLL flux component from the left/right physical fluxes (FL, FR) and
# conserved states (UL, UR): F⁻ if S_L≥0, F⁺ if S_R≤0, else the HLL average.
@inline _hll(FL, FR, UL, UR, S_L, S_R, invΔS) = ifelse(
    S_L >= 0,
    FL,
    ifelse(S_R <= 0, FR, (S_R * FL - S_L * FR + S_L * S_R * (UR - UL)) * invΔS),
)

"""
    kennedy_gruber_hll_cartesian(normal, argvals⁻, argvals⁺)

HLL interface flux for the (ρ, ρe, ρu⃗-Cartesian) conservative system. Unlike the
Roe/Rusanov fluxes (central two-point flux + a dissipation penalty), HLL is a
genuine two-wave Riemann solver built directly from the LEFT and RIGHT *physical*
fluxes `F(y⁻)`, `F(y⁺)` — here `kennedy_gruber_cartesian_flux(y, y)`, which equals
the consistent physical flux and carries the momentum pressure `pm` (= p, or p′
for the stratified/well-balanced split). Signal speeds from
[`_hll_signal_speeds`](@ref) (Davis). The conserved momentum jump uses `ρ u_c`.

More dissipative than Roe on contact/shear waves (a single [S_L, S_R] star region,
no wave-by-wave selection), but robust and positivity-friendly, and — with the
Zhang–Shu limiter — a strong, simple baseline. Volume-scheme-independent (uses the
physical flux, not the two-point volume flux), so it pairs with any VOLUME_FLUX.
Same state fields as [`kennedy_gruber_roe_cartesian`](@ref).
"""
function kennedy_gruber_hll_cartesian(normal, (y⁻,), (y⁺,))
    S_L, S_R = _hll_signal_speeds(normal, y⁻, y⁺)
    invΔS = 1 / (S_R - S_L)
    F⁻ = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁻)
    F⁺ = kennedy_gruber_cartesian_flux(normal, normal, y⁺, y⁺)
    return (
        ρ = _hll(F⁻.ρ, F⁺.ρ, y⁻.ρ, y⁺.ρ, S_L, S_R, invΔS),
        ρe = _hll(F⁻.ρe, F⁺.ρe, y⁻.ρe, y⁺.ρe, S_L, S_R, invΔS),
        ρu1 = _hll(F⁻.ρu1, F⁺.ρu1, y⁻.ρ * y⁻.u1, y⁺.ρ * y⁺.u1, S_L, S_R, invΔS),
        ρu2 = _hll(F⁻.ρu2, F⁺.ρu2, y⁻.ρ * y⁻.u2, y⁺.ρ * y⁺.u2, S_L, S_R, invΔS),
        ρu3 = _hll(F⁻.ρu3, F⁺.ρu3, y⁻.ρ * y⁻.u3, y⁺.ρ * y⁺.u3, S_L, S_R, invΔS),
    )
end

"""
    kennedy_gruber_hll_cartesian_advective(normal, argvals⁻, argvals⁺)

Advection-only counterpart of [`kennedy_gruber_hll_cartesian`](@ref) for the
non-conservative (Exner-perturbation) path: the left/right physical fluxes come
from [`kennedy_gruber_cartesian_advective_flux`](@ref) (momentum flux omits the
pressure term), so HLL supplies wave-selective advective dissipation while the
pressure-gradient force is handled separately by the Exner PGF — exactly as the
Roe/Rusanov/LMARS advective counterparts. Same signal speeds and conserved-state
jumps as the conservative HLL.
"""
function kennedy_gruber_hll_cartesian_advective(normal, (y⁻,), (y⁺,))
    S_L, S_R = _hll_signal_speeds(normal, y⁻, y⁺)
    invΔS = 1 / (S_R - S_L)
    F⁻ = kennedy_gruber_cartesian_advective_flux(normal, normal, y⁻, y⁻)
    F⁺ = kennedy_gruber_cartesian_advective_flux(normal, normal, y⁺, y⁺)
    return (
        ρ = _hll(F⁻.ρ, F⁺.ρ, y⁻.ρ, y⁺.ρ, S_L, S_R, invΔS),
        ρe = _hll(F⁻.ρe, F⁺.ρe, y⁻.ρe, y⁺.ρe, S_L, S_R, invΔS),
        ρu1 = _hll(F⁻.ρu1, F⁺.ρu1, y⁻.ρ * y⁻.u1, y⁺.ρ * y⁺.u1, S_L, S_R, invΔS),
        ρu2 = _hll(F⁻.ρu2, F⁺.ρu2, y⁻.ρ * y⁻.u2, y⁺.ρ * y⁺.u2, S_L, S_R, invΔS),
        ρu3 = _hll(F⁻.ρu3, F⁺.ρu3, y⁻.ρ * y⁻.u3, y⁺.ρ * y⁺.u3, S_L, S_R, invΔS),
    )
end

"""
    hll_tracer(normal, argvals⁻, argvals⁺)

HLL interface flux for a passive tracer `ρq`, consistent with the HLL mass flux:
the tracer physical fluxes are `ρq·uₙ` on each side and the same Davis signal
speeds are used, so `q ≡ const` reproduces the HLL continuity flux (free-stream
preserving). State fields required: `ρ`, `p`, `u1/u2/u3`, `E1/E2/E3`, `q`.
"""
function hll_tracer(normal, (y⁻,), (y⁺,))
    S_L, S_R = _hll_signal_speeds(normal, y⁻, y⁺)
    invΔS = 1 / (S_R - S_L)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    un⁻ = y⁻.u1 * n1 + y⁻.u2 * n2 + y⁻.u3 * n3
    un⁺ = y⁺.u1 * n1 + y⁺.u2 * n2 + y⁺.u3 * n3
    ρq⁻ = y⁻.ρ * y⁻.q
    ρq⁺ = y⁺.ρ * y⁺.q
    return (ρq = _hll(ρq⁻ * un⁻, ρq⁺ * un⁺, ρq⁻, ρq⁺, S_L, S_R, invΔS),)
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

"""
    kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber-style two-point mass flux ``\\{h\\}\\{u ⋅ nvec\\}`` for the
shallow-water height equation (contravariant nodal fluxes averaged). State
fields required: `h`, `uv`.
"""
kennedy_gruber_height_flux(nvec_a, nvec_b, y_a, y_b) =
    ((y_a.h + y_b.h) / 2) * ((y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2)

"""
    kennedy_gruber_rusanov_height(normal, argvals⁻, argvals⁺)

Interface flux for the shallow-water height equation:
[`kennedy_gruber_height_flux`](@ref) central part plus a Rusanov penalty
scaled by the state field `λ`.
"""
function kennedy_gruber_rusanov_height(normal, (y⁻,), (y⁺,))
    λ = max(y⁻.λ, y⁺.λ)
    return kennedy_gruber_height_flux(normal, normal, y⁻, y⁺) -
           λ / 2 * (y⁺.h - y⁻.h)
end

