# Generic DG flux library: reference interface fluxes, face-lifting functions
# for non-conservative terms, and the SIPG Laplacian. Equation-set-specific
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
[`add_lifting_flux_interior!`](@ref) / [`lifting_correction`](@ref).
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
# The DG Laplacian
#
# `q` jumps across element faces, so a second derivative has to be assembled
# from one piece inside each element and three at the faces. Testing
# −∇·(κ∇q) against `v` gives
#
#     Σ_K ∫_K κ∇q·∇v             inside each element
#   − Σ_f ∫_f {{κ∇q}}·n̂ [[v]]    the gradient averaged across the face
#   − Σ_f ∫_f {{κ∇v}}·n̂ [[q]]    the same with q and v swapped
#   + Σ_f ∫_f τ [[q]][[v]]       a penalty pulling the two sides together
#
# where {{·}} is the average of the two sides of a face and [[·]] is their
# difference.
#
# The face loops add the second and fourth pieces directly. The third they
# cannot, since it differentiates the test function `v`, which they never see;
# it is handled from inside the element instead, by spreading the face jump
# back over the element (`R` below) and differentiating that along with the
# gradient. Leaving it out is simpler and still usable, but the operator is
# then not symmetric and one order less accurate.
# ---------------------------------------------------------------------------

"""
    sipg_laplacian_tendency(q, ρ_weight, κ, τ)

Horizontal Laplacian `κ ∇⋅(ρ_weight ∇q)` on a DG space, or `κ ∇²q` when
`ρ_weight === nothing`. Neighbouring elements are coupled by
[`SIPGLaplacianFlux`](@ref), with `τ` from [`sipg_penalty_parameter`](@ref).

Allocates the result and two scratch fields;
[`sipg_laplacian_tendency!`](@ref) takes all three as arguments.
"""
sipg_laplacian_tendency(q, ρ_weight, κ, τ) = sipg_laplacian_tendency!(
    similar(q),
    Fields.Field(Geometry.UVVector{eltype(q)}, axes(q)),
    Fields.Field(Geometry.UVVector{eltype(q)}, axes(q)),
    q,
    ρ_weight,
    κ,
    τ,
)

"""
    sipg_laplacian_tendency!(out, G_uv, R_uv, q, ρ_weight, κ, τ)

In-place [`sipg_laplacian_tendency`](@ref): writes into `out`, using `G_uv` and
`R_uv` as scratch. None of the three may alias `q` or `ρ_weight`. Returns
`out`.
"""
function sipg_laplacian_tendency!(out, G_uv, R_uv, q, ρ_weight, κ, τ)
    wdiv = Divergence{WeakForm}()
    grad = Gradient()
    lgeom = Fields.local_geometry_field(axes(q))
    # G is the gradient each element sees on its own. Face normals are
    # UVVector, so G is built in that basis for the G·n̂ in the face flux, and
    # the volume term differentiates the same field (converting the gradient
    # to UV is exact, so this costs only roundoff).
    if ρ_weight === nothing
        @. G_uv = Geometry.UVVector(grad(q))
    else
        @. G_uv = Geometry.UVVector(ρ_weight * grad(q))
    end
    # R spreads each face jump in `q` back over the element; adding it to the
    # gradient is how the third face piece above gets included. The weight
    # multiplies after the spreading rather than before, because the piece
    # being reproduced carries the weight on the *test* function's gradient.
    fill!(parent(R_uv), zero(Spaces.undertype(axes(q))))
    add_lifting_flux_interior!(central_gradient_lift, R_uv, q)
    if ρ_weight === nothing
        @. R_uv = R_uv / lgeom.WJ
    else
        @. R_uv = ρ_weight * R_uv / lgeom.WJ
    end
    # Volume term: gradient plus spread-out jump. Face flux below: plain
    # gradient.
    @. out = (-lgeom.WJ) * κ * (-wdiv(G_uv + R_uv))
    add_sipg_laplacian_flux_interior!(out, q, G_uv, κ, τ)
    @. out = out / lgeom.WJ
    return out
end

"""
    sipg_penalty_parameter(κ, space; weight = nothing)

Penalty field `τ = κ w (2Nq − 1)^2 / h` for the DG Laplacian, where `h` is the
node spacing and `w` is `weight` (1 when `nothing`).

`τ` sets how hard neighbouring elements are pushed to agree at their shared
face. Too small and the Laplacian stops damping — it can amplify the
grid-scale noise it exists to remove. It is a `Field` rather than one number
because both inputs vary over the mesh: the node spacing by 1.4x on a cubed
sphere, and `weight` by however much the caller's does. A face uses the larger
of its two sides.

Allocates; [`sipg_penalty_parameter!`](@ref) writes into a field you own.
"""
sipg_penalty_parameter(κ, space; weight = nothing) = sipg_penalty_parameter!(
    Fields.Field(Spaces.undertype(space), space),
    κ;
    weight,
)

"""
    sipg_penalty_parameter!(τ, κ; weight = nothing)

In-place [`sipg_penalty_parameter`](@ref): fills `τ`, whose space supplies the
geometry, and returns it.
"""
sipg_penalty_parameter!(τ, κ; weight = nothing) =
    _sipg_penalty_parameter!(τ, κ, weight)

function _sipg_penalty_parameter!(τ, κ, weight)
    space = axes(τ)
    FT = Spaces.undertype(space)
    hspace = Spaces.horizontal_space(space)
    quadrature_style = Spaces.quadrature_style(hspace)
    Nq = Quadratures.degrees_of_freedom(quadrature_style)
    # `Nu` and the constant become the space's float type, and the weight is
    # applied in a second pass, so that no integer and no `nothing` ends up
    # inside a broadcast. Julia puts either of those on the heap there, which
    # is enough to cost the caller an allocation-free tendency.
    Nu = FT(Quadratures.unique_degrees_of_freedom(quadrature_style))
    C = FT((2 * Nq - 1)^2)
    lgeom = Fields.local_geometry_field(space)
    # Choose the 1D or 2D version out here, so each broadcast has a single
    # known function to compile against.
    if hspace isa Spaces.SpectralElementSpace1D
        _sipg_penalty_geometry!(τ, lgeom, κ, C, Nu, _node_length_scale_1d)
    else
        _sipg_penalty_geometry!(τ, lgeom, κ, C, Nu, _node_length_scale_2d)
    end
    weight === nothing || (@. τ = weight * τ)
    return τ
end

# `node_h::F` so the broadcast compiles against the one function it was handed
# rather than looking it up at run time.
_sipg_penalty_geometry!(τ, lgeom, κ, C, Nu, node_h::F) where {F} =
    @. τ = κ * C / node_h(lgeom, Nu)

# Node length scales
@inline function _node_length_scale_1d(lg, Nu)
    ∂x∂ξ = lg.∂x∂ξ
    return 2 * hypot(∂x∂ξ[1, 1], ∂x∂ξ[2, 1]) / Nu
end
@inline function _node_length_scale_2d(lg, Nu)
    ∂x∂ξ = lg.∂x∂ξ
    h₁ = hypot(∂x∂ξ[1, 1], ∂x∂ξ[2, 1])
    h₂ = hypot(∂x∂ξ[1, 2], ∂x∂ξ[2, 2])
    return 2 * min(h₁, h₂) / Nu
end

"""
    SIPGLaplacianFlux()

Interface flux for the DG Laplacian, used with
[`add_numerical_flux_interior!`](@ref). Each side supplies `(q, G, κ, τ)`,
where `G` is the gradient in the same basis as the face normal (usually
[`Geometry.UVVector`](@ref)) and `τ` is a scalar or a `Field`. Returns the
gradient averaged across the face plus a penalty on the jump in `q`:
`-{{κG}}·n̂ + max(τ⁻, τ⁺) * (q⁻ - q⁺)`.

Two of the three face terms; the third rides along in the volume divergence of
[`sipg_laplacian_tendency!`](@ref), which is what makes the operator symmetric.
"""
struct SIPGLaplacianFlux <: AbstractNumericalFlux end

function (::SIPGLaplacianFlux)(normal, argvals⁻, argvals⁺)
    q⁻, G⁻, κ⁻, τ⁻ = argvals⁻[1], argvals⁻[2], argvals⁻[3], argvals⁻[4]
    q⁺, G⁺, κ⁺, τ⁺ = argvals⁺[1], argvals⁺[2], argvals⁺[3], argvals⁺[4]
    Favg = -((κ⁻ * G⁻ + κ⁺ * G⁺) / 2)' * normal
    return Favg + max(τ⁻, τ⁺) * (q⁻ - q⁺)
end

"""
    add_sipg_laplacian_flux_interior!(dydt, q, G, κ, τ)

Add [`SIPGLaplacianFlux`](@ref) at interior faces to a mass-weighted (`WJ`)
Laplacian residual. `τ` may be a scalar or a `Field`. The method with a leading
`ghost_exchange` reuses a [`start_dg_ghost_exchange`](@ref) handle started on
`(q, G, κ, τ)`.
"""
add_sipg_laplacian_flux_interior!(dydt, q, G, κ, τ) =
    add_numerical_flux_interior!(SIPGLaplacianFlux(), dydt, q, G, κ, τ)
add_sipg_laplacian_flux_interior!(
    ghost_exchange::DGGhostExchange,
    dydt,
    q,
    G,
    κ,
    τ,
) = add_numerical_flux_interior!(
    ghost_exchange,
    SIPGLaplacianFlux(),
    dydt,
    q,
    G,
    κ,
    τ,
)
