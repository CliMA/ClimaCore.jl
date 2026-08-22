# Generic DG flux library: reference interface fluxes, two-point fluxes for
# advected scalars, face-lifting functions for non-conservative terms, and
# the LDG/SIPG Laplacian. Equation-set-specific fluxes (compressible Euler,
# Cartesian-momentum Kennedy-Gruber variants) belong downstream (see
# examples/hybrid/sphere/discontinuous_galerkin/euler_dg_fluxes.jl); the
# fluxes here depend only on the state fields named in their docstrings.

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
    y⁻ = argvals⁻[1]
    y⁺ = argvals⁺[1]
    F⁻ = add_auto_broadcasters(fn.fluxfn(argvals⁻...))
    F⁺ = add_auto_broadcasters(fn.fluxfn(argvals⁺...))
    λ = max(fn.wavespeedfn(argvals⁻...), fn.wavespeedfn(argvals⁺...))
    Favg = ((F⁻ + F⁺) / 2)' * normal
    return Favg + (λ / 2) * (y⁻ - y⁺)
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
# Volume term: WJ-weighted κ∇² via (−WJ)·κ·(−wdiv(grad q)); face τ[[q]].
# ---------------------------------------------------------------------------

"""
    ldg_laplacian_tendency(q, ρ_weight, κ, τ)

WJ-normalized LDG / SIPG Laplacian tendency approximating
``κ ∇⋅(ρ_{weight} ∇q)`` (or ``κ ∇²q`` when `ρ_weight === nothing`): weak-form
volume term plus the interior-penalty face flux ``τ [\\![q]\\!]`` (see
[`LDGLaplacianFlux`](@ref) and [`ldg_penalty_parameter`](@ref)).
"""
function ldg_laplacian_tendency(q, ρ_weight, κ, τ)
    wdiv = WeakDivergence()
    grad = Gradient()
    lgeom = Fields.local_geometry_field(axes(q))
    residual = similar(q)
    if ρ_weight === nothing
        @. residual = (-lgeom.WJ) * κ * (-wdiv(grad(q)))
    else
        @. residual = (-lgeom.WJ) * κ * (-wdiv(ρ_weight * grad(q)))
    end
    add_ldg_laplacian_flux_internal!(residual, q, τ)
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

Scalar interior-penalty flux for the LDG/SIPG Laplacian. Called through
[`add_numerical_flux_internal!`](@ref) on a WJ-weighted residual of
``−∇·F`` with ``F = −κ∇q``. Returns ``τ[[q]]`` with ``[[q]] = q⁻ − q⁺``.
"""
struct LDGLaplacianFlux{T} <: AbstractNumericalFlux
    τ::T
end

function (fn::LDGLaplacianFlux)(normal, argvals⁻, argvals⁺)
    q⁻, q⁺ = argvals⁻[1], argvals⁺[1]
    return fn.τ * (q⁻ - q⁺)
end

"""
    add_ldg_laplacian_flux_internal!(dydt, q, τ)

Add LDG interior-penalty face coupling to a WJ-weighted Laplacian residual.
"""
add_ldg_laplacian_flux_internal!(dydt, q, τ) =
    add_numerical_flux_internal!(LDGLaplacianFlux(τ), dydt, q)
