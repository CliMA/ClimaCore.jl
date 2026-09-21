import ClimaComms

"""
    PositivityLimiter(FT; ρ_min = 0, p_min = 0, maxiter = 10)

Zhang–Shu (2010) positivity-preserving limiter for a coupled
conservation-law system on spectral elements.

At each node the whole conserved vector is scaled toward the `WJ`-weighted
element mean,

    U_j ← Ū + θ (U_j − Ū),

by the largest single factor `θ ∈ [0, 1]` that enforces

  - a linear floor on each conserved field given one, and
  - `g ≥ p_min` for a nonlinear admissibility functional `g` of the conserved
    state (e.g. pressure), by per-node bisection in `θ` (`maxiter` steps),

so every element mean is preserved exactly.

# Usage

The limiter is not tied to a prognostic variable family:

    lim = PositivityLimiter(FT; p_min, maxiter)
    apply_positivity_limiter!(lim, gfn, states, floors, off)

`states`: a tuple of conserved `Field`s (any number; element types may be
scalars or `NamedTuple`s of scalars). `floors`: a matching tuple of `nothing`
(scaled but unconstrained) or a number (applied componentwise to a
`NamedTuple`-valued field). `off`: an auxiliary scalar `Field` passed
unscaled to `gfn`, or `nothing`. `gfn(states_node..., off) -> g` must be
GPU-compatible; `nothing` disables the nonlinear floor.

Convenience forms for the compressible Euler system with total energy `ρe`
supply the floors (`ρ_min` on the density, `0` on the tracer):

    apply_positivity_limiter!(lim, pressure_fn, (ρ, ρe, ρu1, ρu2, ρu3, ρq), off)
    apply_positivity_limiter!(lim, pressure_fn, (ρ, ρe, ρu1, ρu2, ρu3), off)

with `pressure_fn(ρ, ρe, ρu1, ρu2, ρu3, ρq, off) -> p` and `off` carrying the
unscaled part of the energy (e.g. `w_c²/2 + Φ`). The element type of `ρq` may
be a scalar or a `NamedTuple` of tracer densities (one `≥ 0` constraint per
species). The dry 5-field form has no tracer constraint and `pressure_fn`
receives `ρq = nothing`.
"""
struct PositivityLimiter{FT} <: AbstractLimiter
    ρ_min::FT
    p_min::FT
    maxiter::Int
end

PositivityLimiter(
    ::Type{FT};
    ρ_min = FT(0),
    p_min = FT(0),
    maxiter::Int = 10,
) where {FT} = PositivityLimiter{FT}(FT(ρ_min), FT(p_min), maxiter)

# Convex combination toward the element mean, elementwise on scalars and on
# NamedTuples of scalars (multi-species tracer fields).
@inline _θmix(θ, x, m) = m + θ * (x - m)
@inline _θmix(θ, ::Nothing, ::Nothing) = nothing
@inline _θmix(θ, x::NamedTuple, m::NamedTuple) =
    map((xc, mc) -> _θmix(θ, xc, mc), x, m)

# Elementwise map over a field value: a `Number` or a `NamedTuple` of them.
@inline _tmap(f::F, xs::Number...) where {F} = f(xs...)
@inline _tmap(f::F, xs::NamedTuple...) where {F} = map(f, xs...)
@inline _tmin(x::Number) = x
@inline _tmin(x::NamedTuple) = min(values(x)...)

# Largest θ keeping the scaled nodal minimum `xmin` at the floor `fmin`: 1 if
# already admissible, ≤ 0 (clamped later) if even the element mean is not.
@inline _θ_floor(m, xmin, fmin) =
    xmin < fmin ? ((m - xmin) > 0 ? (m - fmin) / (m - xmin) : zero(m)) : one(m)

# g(U(θ)) at one node: every conserved value mixed toward its mean by θ.
@inline _g_scaled(gfn::F, θ, vals, means, off) where {F} =
    gfn(map((x, m) -> _θmix(θ, x, m), vals, means)..., off)

# WJ-weighted element mean of one slab field (elementwise for NamedTuples).
@inline function _slab_mean(s, sWJ, Wtot, Ni, Nj)
    acc = _tmap(zero, s[1, 1, 1, 1])
    for j in 1:Nj, i in 1:Ni
        w = sWJ[1, i, j, 1]
        acc = _tmap((a, x) -> muladd(x, w, a), acc, s[1, i, j, 1])
    end
    return _tmap(a -> a / Wtot, acc)
end

# Largest θ keeping every node of slab field `s` at or above `floor`;
# an unconstrained field (`floor === nothing`) imposes no restriction.
@inline _θ_linear(::Type{FT}, s, m, ::Nothing, Ni, Nj) where {FT} = one(FT)
@inline function _θ_linear(::Type{FT}, s, m, floor, Ni, Nj) where {FT}
    xmin = s[1, 1, 1, 1]
    for j in 1:Nj, i in 1:Ni
        xmin = _tmap(min, xmin, s[1, i, j, 1])
    end
    return _tmin(_tmap((mc, xc) -> _θ_floor(mc, xc, floor), m, xmin))
end

# Largest θ ∈ [0, θ_hi] with g(U(θ)) ≥ g_min at one node, by bisection.
# g at θ = 0 is the mean-state value (with this node's off); an admissible
# element mean ⇒ that is ≥ the floor and the bracket is valid.
@inline function _θ_bisect(gfn::F, vals, means, off, θ_hi, g_min, maxiter) where {F}
    _g_scaled(gfn, θ_hi, vals, means, off) >= g_min && return θ_hi
    _g_scaled(gfn, zero(θ_hi), vals, means, off) < g_min && return zero(θ_hi)
    lo, hi = zero(θ_hi), θ_hi
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        if _g_scaled(gfn, mid, vals, means, off) >= g_min
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

"""
    apply_positivity_slab!(lim, gfn, slabs, floors, soff, sWJ)

Apply the [`PositivityLimiter`](@ref) to one element slab (fixed `(v, h)`),
in place. Shared by the CPU and CUDA paths.
"""
function apply_positivity_slab!(
    lim::PositivityLimiter,
    gfn::F,
    slabs::Tuple,
    floors::Tuple,
    soff,
    sWJ,
) where {F}
    (_, Ni, Nj, _) = size(first(slabs))
    FT = eltype(parent(first(slabs)))
    g_min = lim.p_min

    # 1) WJ-weighted element means (the conserved quantities to preserve).
    Wtot = zero(FT)
    for j in 1:Nj, i in 1:Ni
        Wtot += sWJ[1, i, j, 1]
    end
    means = map(s -> _slab_mean(s, sWJ, Wtot, Ni, Nj), slabs)

    # 2) θ from the linear floors: the smallest per-field θ keeping each
    #    constrained field's nodal minimum at its floor.
    θ_fields =
        map((s, m, f) -> _θ_linear(FT, s, m, f, Ni, Nj), slabs, means, floors)
    θ_a = max(reduce(min, θ_fields; init = one(FT)), zero(FT))

    # 3) nonlinear floor `gfn ≥ p_min`: per-node bisection in [0, θ_a], take
    #    the min θ_node. `gfn === nothing` compiles the whole pass away.
    θ = θ_a
    if gfn !== nothing
        for j in 1:Nj, i in 1:Ni
            vals = map(s -> s[1, i, j, 1], slabs)
            off = soff === nothing ? nothing : soff[1, i, j, 1]
            θ_node = _θ_bisect(gfn, vals, means, off, θ_a, g_min, lim.maxiter)
            θ = min(θ, θ_node)
        end
    end

    # 4) apply the common θ to every conserved field (mean-preserving).
    if θ < one(FT)
        for j in 1:Nj, i in 1:Ni
            foreach(
                (s, m) -> s[1, i, j, 1] = _θmix(θ, s[1, i, j, 1], m),
                slabs,
                means,
            )
        end
    end
    return nothing
end

"""
    apply_positivity_limiter!(lim, gfn, states, floors, off)
    apply_positivity_limiter!(lim, pressure_fn, states, off)

Apply the [`PositivityLimiter`](@ref), in place. The first (generic) form
takes any tuple of conserved `Field`s with a matching tuple of linear floors;
the second is the compressible-Euler convenience form for the 6-tuple
`(ρ, ρe, ρu1, ρu2, ρu3, ρq)` or the dry 5-tuple (see
[`PositivityLimiter`](@ref)).
"""
apply_positivity_limiter!(
    lim::PositivityLimiter,
    gfn,
    states::Tuple,
    floors::Tuple,
    off,
) = apply_positivity_limiter!(
    lim,
    gfn,
    states,
    floors,
    off,
    ClimaComms.device(first(states)),
)

# Compressible-Euler convenience forms: `ρ_min` on the density, `≥ 0` on the
# tracer (componentwise for a NamedTuple-valued multi-species field).
apply_positivity_limiter!(
    lim::PositivityLimiter,
    pfn,
    states::Tuple{Any, Any, Any, Any, Any, Any},
    off,
) = apply_positivity_limiter!(
    lim,
    pfn,
    states,
    (lim.ρ_min, nothing, nothing, nothing, nothing, zero(lim.ρ_min)),
    off,
)
apply_positivity_limiter!(
    lim::PositivityLimiter,
    pfn,
    states::Tuple{Any, Any, Any, Any, Any},
    off,
) = apply_positivity_limiter!(
    lim,
    _DryPressure(pfn),
    states,
    (lim.ρ_min, nothing, nothing, nothing, nothing),
    off,
)

# Adapts the Euler pressure functor to the dry 5-field state: the tracer
# argument is pinned to `nothing`, so one `pressure_fn` serves both forms.
struct _DryPressure{P}
    pfn::P
end
@inline (d::_DryPressure)(ρ, ρe, u1, u2, u3, off) =
    d.pfn(ρ, ρe, u1, u2, u3, nothing, off)

@inline _positivity_slab(x, v, h) = slab(x, v, h)
@inline _positivity_slab(::Nothing, v, h) = nothing

function apply_positivity_limiter!(
    lim::PositivityLimiter,
    gfn::F,
    states::Tuple,
    floors::Tuple,
    off,
    ::ClimaComms.AbstractCPUDevice,
) where {F}
    dstates = map(Fields.field_values, states)
    doff = off === nothing ? nothing : Fields.field_values(off)
    dWJ = Spaces.local_geometry_data(axes(first(states))).WJ
    (Nv, _, _, Nh) = size(first(dstates))
    for h in 1:Nh, v in 1:Nv
        apply_positivity_slab!(
            lim,
            gfn,
            map(d -> slab(d, v, h), dstates),
            floors,
            _positivity_slab(doff, v, h),
            slab(dWJ, v, h),
        )
    end
    return nothing
end
