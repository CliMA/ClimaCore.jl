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
