import ClimaComms
import ForwardDiff

"""
    PositivityLimiter{FT, G, F} <: AbstractLimiter

Mean-preserving positivity limiter of [ZhangShu2010](@cite) for the conserved
state of a DG discretization, applied with [`apply_limiter!`](@ref).

In each element, with `Ū` the `WJ`-weighted mean:

 1. Each field with a linear floor is scaled toward its own mean,
    `x_j ← x̄ + θ₁ (x_j − x̄)`, by the largest `θ₁ ∈ [0, 1]` that keeps its
    nodal minimum at the floor (per component for `NamedTuple`-valued
    fields). Fields without a floor are untouched.

 2. If `g` is given, the whole conserved vector is then scaled toward the
    mean, `U_j ← Ū + θ₂ (U_j − Ū)`, by the largest `θ₂ ∈ [0, 1]` with
    `g(U_j) ≥ g_min` at every node (bisection, `maxiter` steps per node).

Both steps are convex combinations with the element mean, so every element
mean is preserved exactly.

`FT` is the floating-point type of the limited states, `G` the type of `g`,
and `F` the type of `floors`.

# Fields

  - `g`: Concave constraint `g(U, aux)`, or `nothing` to skip step 2. `U` is
    one node's state as a `NamedTuple` with the names of the limited states,
    and `aux` the node's value of the auxiliary field.
  - `floors`: `NamedTuple` of linear floors keyed by state names, in the
    units of each state. A floor applies to every component of a
    `NamedTuple`-valued field.
  - `g_min`: Lower bound of the nonlinear constraint `g(U, aux) ≥ g_min`, in
    the units of `g`.
  - `maxiter`: Bisection steps per node for `θ₂` [-]; `θ₂` is accurate to
    `2^-maxiter` and always on the admissible side.

# Constructor

    PositivityLimiter(FT; floors = (;), g = nothing, g_min = 0, maxiter = 10)

Each keyword sets the field of the same name; `floors` and `g_min` are
converted to `FT`.

# Examples

```julia
# Shallow water: depth floor only.
limiter = PositivityLimiter(Float64; floors = (; h = 0))

# Compressible Euler: density and tracer floors, and a pressure floor, with
# the geopotential Φ as the auxiliary field.
pressure(U, Φ) = 0.4 * (U.ρe - (U.ρu1^2 + U.ρu2^2 + U.ρu3^2) / 2U.ρ - U.ρ * Φ)
limiter = PositivityLimiter(
    Float64;
    floors = (; ρ = 1e-6, ρq = 0),
    g = pressure,
    g_min = 1e-3,
)
```

# Notes

  - The limiter only redistributes within an element, so it cannot fix a bad mean.
    Such an element is set to its mean (`θ = 0`) and stays below the floor. With an
    SSP time integrator, the mean stays admissible under the CFL condition of
    [ZhangShu2010](@cite).
  - The bounds hold at the nodes only; the interpolating polynomial may still
    dip below a floor between nodes.
  - `g` must be concave in `U` (as pressure and depth are), so that the
    admissible `θ`s at a node form an interval `[0, θ*]` and bisection finds
    `θ*`.
  - States may hold `ForwardDiff.Dual` values. Derivatives are exact except
    where a floor switches on or off; the bisected `θ₂` is differentiated
    implicitly through `g(U(θ₂)) = g_min`.

See also [`apply_limiter!`](@ref) and the
[Zhang–Shu positivity limiter (DG)](@ref) how-to.
"""
struct PositivityLimiter{FT, G, F} <: AbstractLimiter
    g::G
    floors::F
    g_min::FT
    maxiter::Int
end

PositivityLimiter(
    ::Type{FT};
    floors::NamedTuple = (;),
    g = nothing,
    g_min = FT(0),
    maxiter::Int = 10,
) where {FT} = _positivity_limiter(FT, g, map(f -> FT(f), floors), g_min, maxiter)

_positivity_limiter(::Type{FT}, g::G, floors::F, g_min, maxiter) where {FT, G, F} =
    PositivityLimiter{FT, G, F}(g, floors, FT(g_min), maxiter)

# The limiter bound to the names of `states`: `floors` becomes a tuple in
# state order (`nothing` for unconstrained states; `merge` keeps the order of
# `states`), and `g` is wrapped to rebuild the NamedTuple from plain tuples.
_bind(lim::PositivityLimiter{FT}, states::NamedTuple) where {FT} =
    _positivity_limiter(
        FT,
        _named_state_function(lim.g, states),
        values(merge(map(_ -> nothing, states), lim.floors)),
        lim.g_min,
        lim.maxiter,
    )

# Convex combination toward the element mean, elementwise on scalars and on
# NamedTuples of scalars (a NamedTuple-valued field holding several tracer variables).
@inline _θmix(θ::Number, x::Number, m::Number) = m + θ * (x - m)
@inline _θmix(θ, ::Nothing, ::Nothing) = nothing
@inline _θmix(θ::Number, x::NamedTuple, m::NamedTuple) =
    map((xc, mc) -> _θmix(θ, xc, mc), x, m)
# One θ per variable (step 1 on a NamedTuple-valued field).
@inline _θmix(θ::NamedTuple, x::NamedTuple, m::NamedTuple) =
    map((θc, xc, mc) -> _θmix(θc, xc, mc), θ, x, m)

# Elementwise map over a field value: a `Number` or a `NamedTuple` of them.
@inline _tmap(f::F, xs::Number...) where {F} = f(xs...)
@inline _tmap(f::F, xs::NamedTuple...) where {F} = map(f, xs...)
@inline _tmin(x::Number) = x
@inline _tmin(x::NamedTuple) = min(values(x)...)

# Largest θ ∈ [0, 1] keeping the scaled nodal minimum `xmin` at the floor
# `fmin` (equation (2.5) in [ZhangShu2010]); 1 if already admissible. Clamped
# so that an inadmissible mean (`m < fmin`) or roundoff (`m ≤ xmin`) collapses
# the element to its mean (θ = 0) instead of extrapolating past it.
@inline function _θ_floor(m, xmin, fmin)
    xmin < fmin || return one(m)
    d = m - xmin
    return d > 0 ? clamp((m - fmin) / d, zero(m), one(m)) : zero(m)
end

# g(U(θ)) at one node: every conserved value mixed toward its mean by θ.
@inline _g_scaled(gfn::F, θ, vals, means, aux) where {F} =
    gfn(map((x, m) -> _θmix(θ, x, m), vals, means), aux)

# The kernel works on plain tuples (Base's `map` over NamedTuples does not
# specialize on the closure, which breaks GPU compilation);
# `_NamedStateFunction` gives the user's `g` the node state back as a
# NamedTuple with the state names `K`.
struct _NamedStateFunction{K, G}
    g::G
end
_NamedStateFunction{K}(g::G) where {K, G} = _NamedStateFunction{K, G}(g)
@inline (f::_NamedStateFunction{K})(vals::Tuple, aux) where {K} =
    f.g(NamedTuple{K}(vals), aux)
_named_state_function(::Nothing, ::NamedTuple) = nothing
_named_state_function(g, ::NamedTuple{K}) where {K} = _NamedStateFunction{K}(g)

# WJ-weighted element mean of one slab field: a scalar, or elementwise for a
# flat NamedTuple of scalars (`_tmap` has no methods for vectors or nesting).
@inline function _slab_mean(s, sWJ, Wtot, Ni, Nj)
    acc = _tmap(zero, s[1, 1, 1, 1])
    for j in 1:Nj, i in 1:Ni
        w = sWJ[1, i, j, 1]
        acc = _tmap((a, x) -> muladd(x, w, a), acc, s[1, i, j, 1])
    end
    return _tmap(a -> a / Wtot, acc)
end

# Largest θ₁ ∈ [0, 1] keeping every node of slab field `s` at or above
# `floor`, per component for a NamedTuple-valued field; `nothing` if the
# field has no floor.
@inline _θ_linear(::Type{FT}, s, m, ::Nothing, Ni, Nj) where {FT} = nothing
@inline function _θ_linear(::Type{FT}, s, m, floor, Ni, Nj) where {FT}
    xmin = s[1, 1, 1, 1]
    for j in 1:Nj, i in 1:Ni
        xmin = _tmap(min, xmin, s[1, i, j, 1])
    end
    fmin = FT(floor)
    return _tmap((mc, xc) -> _θ_floor(mc, xc, fmin), m, xmin)
end

# Scale slab field `s` toward its mean `m` by θ (scalar or per-component),
# in place; a no-op when θ is `nothing` or 1 everywhere.
@inline _scale_slab!(s, m, ::Nothing, Ni, Nj) = nothing
@inline function _scale_slab!(s, m, θ, Ni, Nj)
    _tmin(θ) < 1 || return nothing
    for j in 1:Nj, i in 1:Ni
        s[1, i, j, 1] = _θmix(θ, s[1, i, j, 1], m)
    end
    return nothing
end

# Largest θ ∈ [0, θ_hi] with g(U(θ)) ≥ g_min at one node, by bisection
# (paper's (2.12)). The bracket is valid because θ = 0 is the mean state,
# assumed admissible.
@inline function _θ_bisect(gfn::F, vals, means, aux, θ_hi, g_min, maxiter) where {F}
    _g_scaled(gfn, θ_hi, vals, means, aux) >= g_min && return θ_hi
    lo, hi = zero(θ_hi), θ_hi
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        if _g_scaled(gfn, mid, vals, means, aux) >= g_min
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

# Total WJ of one slab.
@inline function _slab_total(::Type{FT}, sWJ, Ni, Nj) where {FT}
    Wtot = zero(FT)
    for j in 1:Nj, i in 1:Ni
        Wtot += sWJ[1, i, j, 1]
    end
    return Wtot
end

# Primal (non-Dual) part of a node value.
@inline _primal(x::Number) = ForwardDiff.value(x)
@inline _primal(x::Union{Tuple, NamedTuple}) = map(_primal, x)
@inline _primal(::Nothing) = nothing

# Bisection drops the derivative of θ; for Dual states, restore it at the
# active node from g(U(θ)) = g_min: dθ = −dG / (∂g/∂θ), with `dG` the
# partials of G = g(U(θ)) at fixed θ. Identity for plain-float states.
@inline _θ_with_partials(gfn, θ, θ_a, G::Real, vals, means, aux) = θ
@inline function _θ_with_partials(
    gfn::F,
    θ,
    θ_a,
    G::ForwardDiff.Dual,
    vals,
    means,
    aux,
) where {F}
    dG = G - ForwardDiff.value(G)
    θ < θ_a || return θ + zero(dG)
    pvals, pmeans, paux = _primal(vals), _primal(means), _primal(aux)
    gθ = ForwardDiff.derivative(t -> _g_scaled(gfn, t, pvals, pmeans, paux), θ)
    return iszero(gθ) ? θ + zero(dG) : θ - dG / gθ
end

# State and aux values at node (i, j); a function, since a closure over the
# loop's running indices would box them.
@inline _node_values(slabs, saux, i, j) =
    (map(s -> s[1, i, j, 1], slabs), saux === nothing ? nothing : saux[1, i, j, 1])

# Min over the slab nodes of the bisected θ for `gfn ≥ g_min`, capped at θ_a;
# for Dual states, the derivative comes from the node that sets the min.
@inline _θ_nonlinear(::Nothing, slabs, means, saux, θ_a, g_min, maxiter, Ni, Nj) =
    θ_a
@inline function _θ_nonlinear(
    gfn::F,
    slabs,
    means,
    saux,
    θ_a,
    g_min,
    maxiter,
    Ni,
    Nj,
) where {F}
    θ, iθ, jθ = θ_a, 1, 1
    for j in 1:Nj, i in 1:Ni
        (vals, aux) = _node_values(slabs, saux, i, j)
        θn = _θ_bisect(gfn, vals, means, aux, θ_a, g_min, maxiter)
        if θn < θ
            θ, iθ, jθ = θn, i, j
        end
    end
    (vals, aux) = _node_values(slabs, saux, iθ, jθ)
    G = _g_scaled(gfn, θ, vals, means, aux)
    return _θ_with_partials(gfn, θ, θ_a, G, vals, means, aux)
end

"""
    apply_positivity_slab!(lim, slabs, saux, sWJ)

Apply the bound limiter `lim` to the slabs of one element (fixed `(v, h)`),
in place. Called from the CPU and CUDA methods of [`apply_limiter!`](@ref).
"""
function apply_positivity_slab!(
    lim::PositivityLimiter{FT, <:Any, <:NTuple{N, Any}},
    slabs::NTuple{N, Any},
    saux,
    sWJ,
) where {FT, N}
    (_, Ni, Nj, _) = size(first(slabs))
    (; g_min, floors) = lim
    gfn = lim.g

    # 1) WJ-weighted element means (the conserved quantities to preserve).
    Wtot = _slab_total(FT, sWJ, Ni, Nj)
    means = map(s -> _slab_mean(s, sWJ, Wtot, Ni, Nj), slabs)

    # 2) Step 1, per-field θ₁ on the constrained fields, in place. Tuple
    #    `map` rather than `foreach`: `foreach` zips and allocates.
    map(
        (s, m, f) ->
            _scale_slab!(s, m, _θ_linear(FT, s, m, f, Ni, Nj), Ni, Nj),
        slabs,
        means,
        floors,
    )

    # 3) Step 2, θ₂ = min over nodes of the per-node bisection on the
    #    intermediate state. `gfn === nothing` compiles the pass away.
    θ₂ = _θ_nonlinear(
        gfn,
        slabs,
        means,
        saux,
        one(FT),
        g_min,
        lim.maxiter,
        Ni,
        Nj,
    )

    # 4) The common θ₂ scales every field toward its mean.
    map((s, m) -> _scale_slab!(s, m, θ₂, Ni, Nj), slabs, means)
    return nothing
end

"""
    apply_limiter!(states::NamedTuple, aux, limiter::PositivityLimiter)

Apply the [`PositivityLimiter`](@ref) `limiter` to `states`, element by
element. Mutates the fields of `states`; returns `nothing`.

# Arguments

  - `states`: `NamedTuple` of the conserved `Field`s on one space, e.g.
    `(; ρ, ρe, ρu1, ρu2, ρu3)`, with the limiter's eltype. Its names must
    include every key of `limiter.floors`, and are the names `limiter.g`
    reads. Field elements must be scalars or flat `NamedTuple`s of scalars
    (several tracers in one field); pass vector quantities such as momentum
    as separate scalar component fields.
  - `aux`: `Field` of non-conserved data passed unscaled to `limiter.g`, or
    `nothing`; e.g. the geopotential [m²/s²].
  - `limiter`: The [`PositivityLimiter`](@ref).

# Examples

```julia
# h, hu1, hu2: depth and momentum `Field`s on one spectral-element space.
limiter = PositivityLimiter(Float64; floors = (; h = 0))
apply_limiter!((; h, hu1, hu2), nothing, limiter)
```

See [Zhang–Shu positivity limiter (DG)](@ref) for a runnable shallow-water
example and the compressible Euler setup.
"""
function apply_limiter!(
    states::NamedTuple,
    aux,
    limiter::PositivityLimiter{FT},
) where {FT}
    # Checked on the host, before the CPU/CUDA split: a misspelled floor
    # would otherwise be silently ignored.
    issubset(keys(limiter.floors), keys(states)) || throw(
        ArgumentError(
            "floors $(keys(limiter.floors)) must all name states $(keys(states))",
        ),
    )
    all(s -> eltype(parent(s)) == FT, values(states)) ||
        throw(ArgumentError("states must have the limiter's eltype $FT"))
    return apply_limiter!(
        values(states),
        aux,
        _bind(limiter, states),
        ClimaComms.device(first(states)),
    )
end

function apply_limiter!(
    states::Tuple,
    aux,
    lim::PositivityLimiter,
    dev::ClimaComms.AbstractCPUDevice,
)
    dstates = map(Fields.field_values, states)
    daux = aux === nothing ? nothing : Fields.field_values(aux)
    dWJ = Spaces.local_geometry_data(axes(first(states))).WJ
    (Nv, _, _, Nh) = size(first(dstates))
    for h in 1:Nh, v in 1:Nv
        apply_positivity_slab!(
            lim,
            map(d -> slab(d, v, h), dstates),
            daux === nothing ? nothing : slab(daux, v, h),
            slab(dWJ, v, h),
        )
    end
    # Once per state: the callback inspects one `Field` result.
    call_post_op_callback() &&
        foreach(s -> post_op_callback(s, s, aux, lim, dev), states)
    return nothing
end
