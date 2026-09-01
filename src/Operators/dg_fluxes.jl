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
    central_curl3_lift(normal, (u⁻,), (u⁺,))

Vector-argument form of the above: the horizontal vector `u` in the same
orthonormal basis as the normal (usually [`Geometry.UVVector`](@ref)), rather
than its two components as separate fields. One field to index and to stage in
the halo exchange instead of two.
"""
function central_curl3_lift(
    normal,
    argvals⁻::Tuple{Any},
    argvals⁺::Tuple{Any},
)
    Δu = (argvals⁺[1] - argvals⁻[1]) / 2
    return normal.components.data.:1 * Δu.components.data.:2 -
           normal.components.data.:2 * Δu.components.data.:1
end

"""
    central_divergence_lift(normal, (u⁻,), (u⁺,))

Symmetric central lifting completing the strong-form DG divergence of a
horizontal vector: each side adds ``(u^* - u_{side}) ⋅ n̂_{side}`` with central
``u^*``, i.e. ``((u⁺ - u⁻)/2) ⋅ n̂`` on the minus side. `u` must be in the same
basis as the face normal (usually [`Geometry.UVVector`](@ref)). Use with
[`add_lifting_flux_interior!`](@ref) / [`lifting_correction`](@ref).
"""
central_divergence_lift(normal, (u⁻,), (u⁺,)) = ((u⁺ - u⁻) / 2)' * normal

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
    ldg_laplacian_tendency(q, ρ_weight, κ, τ)

Horizontal Laplacian `κ ∇⋅(ρ_weight ∇q)` on a DG space, or `κ ∇²q` when
`ρ_weight === nothing`. Neighbouring elements are coupled by
[`LDGLaplacianFlux`](@ref), with `τ` from [`ldg_penalty_parameter`](@ref).

Allocates the result and two scratch fields;
[`ldg_laplacian_tendency!`](@ref) takes all three as arguments.
"""
ldg_laplacian_tendency(q, ρ_weight, κ, τ) = ldg_laplacian_tendency!(
    similar(q),
    Fields.Field(Geometry.UVVector{eltype(q)}, axes(q)),
    Fields.Field(Geometry.UVVector{eltype(q)}, axes(q)),
    q,
    ρ_weight,
    κ,
    τ,
)

"""
    ldg_laplacian_tendency!(out, G_uv, R_uv, q, ρ_weight, κ, τ)

In-place [`ldg_laplacian_tendency`](@ref): writes into `out`, using `G_uv` and
`R_uv` as scratch. None of the three may alias `q` or `ρ_weight`. Returns
`out`.
"""
function ldg_laplacian_tendency!(out, G_uv, R_uv, q, ρ_weight, κ, τ)
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
    add_ldg_laplacian_flux_interior!(out, q, G_uv, κ, τ)
    @. out = out / lgeom.WJ
    return out
end

"""
    ldg_penalty_parameter(κ, space; weight = nothing)

Penalty field `τ = κ w (2Nq − 1)^2 / h` for the DG Laplacian, where `h` is the
node spacing and `w` is `weight` (1 when `nothing`).

`τ` sets how hard neighbouring elements are pushed to agree at their shared
face. Too small and the Laplacian stops damping — it can amplify the
grid-scale noise it exists to remove. It is a `Field` rather than one number
because both inputs vary over the mesh: the node spacing by 1.4x on a cubed
sphere, and `weight` by however much the caller's does. A face uses the larger
of its two sides.

Allocates; [`ldg_penalty_parameter!`](@ref) writes into a field you own.
"""
ldg_penalty_parameter(κ, space; weight = nothing) = ldg_penalty_parameter!(
    Fields.Field(Spaces.undertype(space), space),
    κ;
    weight,
)

"""
    ldg_penalty_parameter!(τ, κ; weight = nothing)

In-place [`ldg_penalty_parameter`](@ref): fills `τ`, whose space supplies the
geometry, and returns it.
"""
ldg_penalty_parameter!(τ, κ; weight = nothing) =
    _ldg_penalty_parameter!(τ, κ, weight)

function _ldg_penalty_parameter!(τ, κ, weight)
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
        _ldg_penalty_geometry!(τ, lgeom, κ, C, Nu, _node_length_scale_1d)
    else
        _ldg_penalty_geometry!(τ, lgeom, κ, C, Nu, _node_length_scale_2d)
    end
    weight === nothing || (@. τ = weight * τ)
    return τ
end

# `node_h::F` so the broadcast compiles against the one function it was handed
# rather than looking it up at run time.
_ldg_penalty_geometry!(τ, lgeom, κ, C, Nu, node_h::F) where {F} =
    @. τ = κ * C / node_h(lgeom, Nu)

# How far apart the nodes are here: the element's horizontal width at this
# node, divided by the number of distinct quadrature points across it. The
# local version of `Spaces.node_horizontal_length_scale`, which averages over
# the whole mesh.
#
# Columns 1 and 2 of the (identity-padded) `∂x∂ξ` are the tangent vectors along
# the element's two horizontal directions, and the reference element runs from
# -1 to 1, so the width along direction `r` is `2‖∂x/∂ξʳ‖`. Only the (U, V)
# rows are read, so on a terrain-following grid a steeply tilted coordinate
# line does not come out as extra horizontal width.
#
# Taking the smaller of the two widths errs towards a larger penalty. For a
# square element it is the right answer; for a long thin one it over-penalizes
# the long direction, which costs accuracy and time-step size but never
# stability. Getting that case right too would need a τ that depends on which
# face you are looking at, which one value per node cannot express.
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
    LDGLaplacianFlux()

Interface flux for the DG Laplacian, used with
[`add_numerical_flux_interior!`](@ref). Each side supplies `(q, G, κ, τ)`,
where `G` is the gradient in the same basis as the face normal (usually
[`Geometry.UVVector`](@ref)) and `τ` is a scalar or a `Field`. Returns the
gradient averaged across the face plus a penalty on the jump in `q`:
`-{{κG}}·n̂ + max(τ⁻, τ⁺) * (q⁻ - q⁺)`.

Two of the three face terms; the third rides along in the volume divergence of
[`ldg_laplacian_tendency!`](@ref), which is what makes the operator symmetric.
"""
struct LDGLaplacianFlux <: AbstractNumericalFlux end

function (::LDGLaplacianFlux)(normal, argvals⁻, argvals⁺)
    q⁻, G⁻, κ⁻, τ⁻ = argvals⁻[1], argvals⁻[2], argvals⁻[3], argvals⁻[4]
    q⁺, G⁺, κ⁺, τ⁺ = argvals⁺[1], argvals⁺[2], argvals⁺[3], argvals⁺[4]
    Favg = -((κ⁻ * G⁻ + κ⁺ * G⁺) / 2)' * normal
    return Favg + max(τ⁻, τ⁺) * (q⁻ - q⁺)
end

"""
    add_ldg_laplacian_flux_interior!(dydt, q, G, κ, τ)

Add [`LDGLaplacianFlux`](@ref) at interior faces to a mass-weighted (`WJ`)
Laplacian residual. `τ` may be a scalar or a `Field`. The method with a leading
`ghost_exchange` reuses a [`start_dg_ghost_exchange`](@ref) handle started on
`(q, G, κ, τ)`.
"""
add_ldg_laplacian_flux_interior!(dydt, q, G, κ, τ) =
    add_numerical_flux_interior!(LDGLaplacianFlux(), dydt, q, G, κ, τ)
add_ldg_laplacian_flux_interior!(
    ghost_exchange::DGGhostExchange,
    dydt,
    q,
    G,
    κ,
    τ,
) = add_numerical_flux_interior!(
    ghost_exchange,
    LDGLaplacianFlux(),
    dydt,
    q,
    G,
    κ,
    τ,
)

# ---------------------------------------------------------------------------
# The DG vector Laplacian
#
# ∇²u = ∇(∇⋅u) − ∇×(∇×u) splits the smoothing into the divergence ∇⋅u and the
# vertical component of the curl, ẑ⋅(∇×u).
# For a test function `v`:
#     Σ_K ∫_K [α (∇⋅u)(∇⋅v) + (ẑ⋅(∇×u))(ẑ⋅(∇×v))]  inside each element
#   − Σ_f ∫_f [α {{∇⋅u}} [[v]]⋅n̂ + {{ẑ⋅(∇×u)}} ẑ⋅(n̂ × [[v]])]
#   − Σ_f ∫_f [the same with u and v swapped]
#   + Σ_f ∫_f τ [[u]]⋅[[v]]                       a penalty term
# ---------------------------------------------------------------------------

"""
    VectorLaplacianFlux(divergence_factor)

Interface flux for the DG vector Laplacian, used with
[`add_numerical_flux_interior!`](@ref). Each side supplies
`(u, ∇⋅u, ẑ⋅(∇×u), τ)`: the vector in the face normal's basis (usually
[`Geometry.UVVector`](@ref)), the divergence and the vertical component of the
curl as that side's element computed them, and the penalty, a scalar or a
`Field`. With one horizontal dimension there is no curl term, and each side
supplies `(u, ∇⋅u, τ)`. Returns

    -α {{∇⋅u}} n̂ - {{ẑ⋅(∇×u)}} (ẑ × n̂) + max(τ⁻, τ⁺) * (α [[u]]⋅n̂ n̂ + [[u]]ₜ)

where `α = divergence_factor` and `[[u]]ₜ` is the tangential part of the jump.
`α` multiplies the grad-div part and the normal-jump penalty that holds it
together.
"""
struct VectorLaplacianFlux{FT} <: AbstractNumericalFlux
    divergence_factor::FT
end

function (fn::VectorLaplacianFlux)(
    normal,
    argvals⁻::NTuple{4, Any},
    argvals⁺::NTuple{4, Any},
)
    u⁻, divu⁻, curlu⁻, τ⁻ = argvals⁻[1], argvals⁻[2], argvals⁻[3], argvals⁻[4]
    u⁺, divu⁺, curlu⁺, τ⁺ = argvals⁺[1], argvals⁺[2], argvals⁺[3], argvals⁺[4]
    α = fn.divergence_factor
    αdiv = α * (divu⁻ + divu⁺) / 2
    curlavg = (curlu⁻ + curlu⁺) / 2
    n₁ = normal.components.data.:1
    n₂ = normal.components.data.:2
    # -α {{∇⋅u}} n̂ - {{ẑ⋅(∇×u)}} (ẑ × n̂), with ẑ × n̂ = (-n₂, n₁) in the
    # (U, V) frame.
    Favg =
        Geometry.UVVector(curlavg * n₂ - αdiv * n₁, -curlavg * n₁ - αdiv * n₂)
    # The normal part of the jump goes with the divergence and carries its
    # factor; what is left is tangential and goes with the curl.
    Δ = u⁻ - u⁺
    pen = Δ + (α - 1) * (Δ' * normal) * normal
    return Favg + max(τ⁻, τ⁺) * pen
end

function (fn::VectorLaplacianFlux)(
    normal,
    argvals⁻::NTuple{3, Any},
    argvals⁺::NTuple{3, Any},
)
    u⁻, divu⁻, τ⁻ = argvals⁻[1], argvals⁻[2], argvals⁻[3]
    u⁺, divu⁺, τ⁺ = argvals⁺[1], argvals⁺[2], argvals⁺[3]
    α = fn.divergence_factor
    αdiv = α * (divu⁻ + divu⁺) / 2
    # With one horizontal dimension every jump is normal to the face, so all of
    # it goes with the divergence.
    return -αdiv * normal + α * max(τ⁻, τ⁺) * (u⁻ - u⁺)
end

"""
    ldg_vector_laplacian_tendency(u, divergence_factor, τ)

Horizontal vector Laplacian of `u` on a DG space,
`divergence_factor * ∇(∇⋅u) − ∇×(∇×u)`, in the basis of `u`. Neighbouring
elements are coupled by [`VectorLaplacianFlux`](@ref), with `τ` from
[`ldg_penalty_parameter`](@ref) at unit diffusivity; the flux applies
`divergence_factor` to the part of the penalty that carries it.

Allocates the result and the scratch fields;
[`ldg_vector_laplacian_tendency!`](@ref) takes all of them as arguments.
"""
function ldg_vector_laplacian_tendency(u, divergence_factor, τ)
    space = axes(u)
    FT = Spaces.undertype(space)
    if Spaces.horizontal_space(space) isa Spaces.SpectralElementSpace1D
        V = Geometry.UVector{FT}
        curlu = nothing
        R_curlu = nothing
    else
        V = Geometry.UVVector{FT}
        curlu = Fields.Field(FT, space)
        R_curlu = Fields.Field(FT, space)
    end
    return ldg_vector_laplacian_tendency!(
        similar(u),
        Fields.Field(V, space),
        Fields.Field(V, space),
        Fields.Field(FT, space),
        Fields.Field(FT, space),
        curlu,
        R_curlu,
        u,
        divergence_factor,
        τ,
    )
end

"""
    ldg_vector_laplacian_tendency!(out, r, u_loc, divu, R_divu, curlu, R_curlu, u, divergence_factor, τ)

In-place [`ldg_vector_laplacian_tendency`](@ref): writes into `out`, using as
scratch the mass-weighted residual `r` and the vector `u_loc` (both in the face
normals' basis), the divergence `divu` and the lifting `R_divu` of the face
jumps in `u`, and the vertical curl `curlu` and its `R_curlu`. Pass
`curlu = R_curlu = nothing` with one horizontal dimension, where there is no
curl-curl part. No scratch field may alias `u`; `out` may. Returns `out`.
"""
function ldg_vector_laplacian_tendency!(
    out,
    r,
    u_loc,
    divu,
    R_divu,
    curlu,
    R_curlu,
    u,
    divergence_factor,
    τ,
)
    space = axes(u)
    FT = Spaces.undertype(space)
    α = FT(divergence_factor)
    lgeom = Fields.local_geometry_field(space)
    div = Divergence()
    curl = Curl()
    wgrad = Gradient{WeakForm}()
    wcurl = Curl{WeakForm}()
    # `u` in the frame the face terms work in, which the two sides of a face
    # node share. The volume operators read the same field: converting to it is
    # exact, so it costs only roundoff.
    _project_into!(u_loc, u, lgeom)
    # Both liftings read `u_loc` alone, so one halo exchange serves both.
    ghost = start_dg_ghost_exchange(space, u_loc)
    # The divergence each element sees on its own, and the same with the face
    # jumps spread back over it. The face flux averages the first; the volume
    # term differentiates the second, which is what keeps the operator
    # symmetric.
    @. divu = div(u_loc)
    fill!(parent(R_divu), zero(FT))
    add_lifting_flux_interior!(ghost, central_divergence_lift, R_divu, u_loc)
    @. R_divu = divu + R_divu / lgeom.WJ
    if curlu === nothing
        @. r = lgeom.WJ * Geometry.UVector(α * wgrad(R_divu), lgeom)
        add_numerical_flux_interior!(VectorLaplacianFlux(α), r, u_loc, divu, τ)
    else
        # ẑ⋅(∇×u), the same way: covariant components in, since a curl is free
        # of the metric in those, and the physical component out, since that is
        # what the face terms measure.
        @. curlu = _vertical_curl(
            curl(Geometry.Covariant12Vector(u_loc, lgeom)),
            lgeom,
        )
        fill!(parent(R_curlu), zero(FT))
        add_lifting_flux_interior!(ghost, central_curl3_lift, R_curlu, u_loc)
        @. R_curlu = curlu + R_curlu / lgeom.WJ
        @. r =
            lgeom.WJ * Geometry.UVVector(
                α * wgrad(R_divu) - Geometry.Covariant12Vector(
                    wcurl(
                        Geometry.Covariant3Vector(
                            Geometry.WVector(R_curlu),
                            lgeom,
                        ),
                    ),
                ),
                lgeom,
            )
        add_numerical_flux_interior!(
            VectorLaplacianFlux(α),
            r,
            u_loc,
            divu,
            curlu,
            τ,
        )
    end
    return _unweight_into!(out, r, lgeom)
end

# The vertical component of a curl, as a scalar rate. Orthonormal rather than
# contravariant, so that the volume and face terms measure the same thing where
# the two differ: a grid whose third coordinate is not a length, such as an
# extruded one.
@inline _vertical_curl(curl_u, lg) = Geometry.WVector(curl_u, lg).w

# Write `v` into `dest`, converting to the basis `dest`'s element type asks
# for: the caller chooses the basis of its own field, and the operators here
# work in the face normals'.
_project_into!(dest, v, lgeom) =
    _project_into!(eltype(dest), dest, v, lgeom)
function _project_into!(::Type{T}, dest, v, lgeom) where {T}
    @. dest = T(v, lgeom)
    return dest
end

# The same, for a mass-weighted residual: unweight, then convert.
_unweight_into!(dest, r, lgeom) = _unweight_into!(eltype(dest), dest, r, lgeom)
function _unweight_into!(::Type{T}, dest, r, lgeom) where {T}
    @. dest = T(r / lgeom.WJ, lgeom)
    return dest
end
