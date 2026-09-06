import ClimaComms

"""
    PositivityLimiter(FT; ρ_min = 0, p_min = 0, maxiter = 10)

A Zhang–Shu (2010) positivity-preserving limiter for a coupled
conservation-law system on spectral elements.

Unlike [`QuasiMonotoneLimiter`](@ref) (two-sided, neighbor-based shape
preservation for a tracer), this limiter enforces a one-sided physical floor
on the dynamics: it scales the whole conserved vector at each node toward the
`WJ`-weighted element mean by a single factor `θ ∈ [0, 1]`,

    U_j ← Ū + θ (U_j − Ū),

so the limited state is a convex combination of the (admissible) element mean and
the nodal value — every element mean (mass, energy, water, …) is preserved
exactly. `θ` is the smallest factor that simultaneously enforces

  - `ρ ≥ ρ_min`               (density positivity),
  - `ρq ≥ 0`                  (tracer/moisture positivity),
  - `p ≥ p_min`               (pressure positivity, via a user pressure functor),

the last found by a per-node bisection in `θ` (pressure is a nonlinear functional
of the conserved vector). The theory guarantees a valid `θ` exists whenever the
element mean is admissible, which a conservative update preserves under a CFL
condition.

# Usage

    lim = PositivityLimiter(FT; ρ_min, p_min, maxiter)
    apply_positivity_limiter!(lim, pressure_fn, (ρ, ρe, ρu1, ρu2, ρu3, ρq), off)
    apply_positivity_limiter!(lim, pressure_fn, (ρ, ρe, ρu1, ρu2, ρu3), off)

where `(ρ, ρe, ρu1, ρu2, ρu3, ρq)` are the conserved scalar `Field`s to scale
(the first is the density used for the `ρ_min` constraint; the last is the tracer
used for the `≥ 0` constraint), `off` is an auxiliary scalar `Field` (e.g.
`w_c²/2 + Φ`) passed unscaled to the pressure functor, and

    pressure_fn(ρ, ρe, ρu1, ρu2, ρu3, ρq, off) -> p

returns the pressure the `p_min` floor is applied to. `pressure_fn` must be
GPU-compatible (it is called inside the device kernel).

The 5-field form is the dry case: there is no tracer constraint, and the
pressure functor receives `ρq = nothing`, so a dry `pressure_fn` should accept
(and ignore) that argument.
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

# Convex combination toward the element mean; `nothing` (the dry case's
# absent tracer) passes through, so one `_p_scaled` serves both state shapes.
@inline _θmix(θ, x, m) = m + θ * (x - m)
@inline _θmix(θ, ::Nothing, ::Nothing) = nothing

@inline function _p_scaled(
    pfn,
    θ,
    ρ0, ρe0, u10, u20, u30, ρq0,
    mρ, mρe, mu1, mu2, mu3, mρq,
    off,
)
    return pfn(
        _θmix(θ, ρ0, mρ),
        _θmix(θ, ρe0, mρe),
        _θmix(θ, u10, mu1),
        _θmix(θ, u20, mu2),
        _θmix(θ, u30, mu3),
        _θmix(θ, ρq0, mρq),
        off,
    )
end

"""
    apply_positivity_slab!(lim, pfn, sρ, sρe, su1, su2, su3, sρq, soff, sWJ)

Apply the [`PositivityLimiter`](@ref) to one element slab (fixed `(v, h)`),
in place. Shared by the CPU and CUDA paths. `sρq === nothing` is the dry
(tracer-less) case.
"""
function apply_positivity_slab!(
    lim::PositivityLimiter,
    pfn::F,
    sρ, sρe, su1, su2, su3, sρq, soff, sWJ,
) where {F}
    (_, Ni, Nj, _) = size(sρ)
    FT = eltype(parent(sρ))
    ρ_min = lim.ρ_min
    p_min = lim.p_min

    # 1) WJ-weighted element means (the conserved quantities to preserve).
    #    `sρq === nothing` is the dry case: every tracer branch below compiles
    #    away and `mρq = nothing` flows through `_θmix` into the pressure call.
    Wtot = zero(FT)
    mρ = zero(FT); mρe = zero(FT)
    mu1 = zero(FT); mu2 = zero(FT); mu3 = zero(FT)
    mρq = sρq === nothing ? nothing : zero(FT)
    for j in 1:Nj, i in 1:Ni
        w = sWJ[1, i, j, 1]
        Wtot += w
        mρ += sρ[1, i, j, 1] * w
        mρe += sρe[1, i, j, 1] * w
        mu1 += su1[1, i, j, 1] * w
        mu2 += su2[1, i, j, 1] * w
        mu3 += su3[1, i, j, 1] * w
        if sρq !== nothing
            mρq += sρq[1, i, j, 1] * w
        end
    end
    mρ /= Wtot; mρe /= Wtot
    mu1 /= Wtot; mu2 /= Wtot; mu3 /= Wtot
    if mρq !== nothing
        mρq /= Wtot
    end

    # 2) θ from the two linear floors (density, tracer). (num)/(num − min):
    #    the largest θ keeping the scaled min at the floor. If the mean itself
    #    is inadmissible the ratio is ≤ 0 ⇒ θ collapses to the mean (θ = 0).
    θ = one(FT)
    ρmin_node = FT(Inf)
    for j in 1:Nj, i in 1:Ni
        ρmin_node = min(ρmin_node, sρ[1, i, j, 1])
    end
    if ρmin_node < ρ_min
        d = mρ - ρmin_node
        θ = min(θ, d > 0 ? (mρ - ρ_min) / d : zero(FT))
    end
    if sρq !== nothing
        ρqmin_node = FT(Inf)
        for j in 1:Nj, i in 1:Ni
            ρqmin_node = min(ρqmin_node, sρq[1, i, j, 1])
        end
        if ρqmin_node < 0
            d = mρq - ρqmin_node
            θ = min(θ, d > 0 ? mρq / d : zero(FT))
        end
    end
    θ = max(θ, zero(FT))
    θ_a = θ

    # 3) pressure floor: per-node bisection in [0, θ_a], take the min θ_node.
    #    p at θ=0 is the mean-state pressure (with this node's off); admissible
    #    element mean ⇒ that is ≥ p_min and the bracket is valid.
    θ_final = θ_a
    for j in 1:Nj, i in 1:Ni
        ρ0 = sρ[1, i, j, 1]; ρe0 = sρe[1, i, j, 1]
        u10 = su1[1, i, j, 1]; u20 = su2[1, i, j, 1]; u30 = su3[1, i, j, 1]
        ρq0 = sρq === nothing ? nothing : sρq[1, i, j, 1]
        off = soff[1, i, j, 1]
        p_hi = _p_scaled(pfn, θ_a, ρ0, ρe0, u10, u20, u30, ρq0, mρ, mρe, mu1, mu2, mu3, mρq, off)
        if p_hi < p_min
            p0 = pfn(mρ, mρe, mu1, mu2, mu3, mρq, off)
            if p0 < p_min
                θ_final = zero(FT)
            else
                lo = zero(FT); hi = θ_a
                for _ in 1:lim.maxiter
                    mid = (lo + hi) / 2
                    pm = _p_scaled(pfn, mid, ρ0, ρe0, u10, u20, u30, ρq0, mρ, mρe, mu1, mu2, mu3, mρq, off)
                    if pm >= p_min
                        lo = mid
                    else
                        hi = mid
                    end
                end
                θ_final = min(θ_final, lo)
            end
        end
    end
    θ = θ_final

    # 4) apply the common θ to every conserved component (mean-preserving).
    if θ < one(FT)
        for j in 1:Nj, i in 1:Ni
            sρ[1, i, j, 1] = _θmix(θ, sρ[1, i, j, 1], mρ)
            sρe[1, i, j, 1] = _θmix(θ, sρe[1, i, j, 1], mρe)
            su1[1, i, j, 1] = _θmix(θ, su1[1, i, j, 1], mu1)
            su2[1, i, j, 1] = _θmix(θ, su2[1, i, j, 1], mu2)
            su3[1, i, j, 1] = _θmix(θ, su3[1, i, j, 1], mu3)
            if sρq !== nothing
                sρq[1, i, j, 1] = _θmix(θ, sρq[1, i, j, 1], mρq)
            end
        end
    end
    return nothing
end

"""
    apply_positivity_limiter!(lim, pressure_fn, states, off)

Apply the [`PositivityLimiter`](@ref). `states` is the 6-tuple of conserved
scalar `Field`s `(ρ, ρe, ρu1, ρu2, ρu3, ρq)`, or the 5-tuple without the
tracer for a dry state; `off` is the auxiliary scalar `Field`; `pressure_fn`
is the pressure functor (see [`PositivityLimiter`](@ref)).
"""
apply_positivity_limiter!(lim::PositivityLimiter, pfn, states, off) =
    apply_positivity_limiter!(lim, pfn, states, off, ClimaComms.device(off))

# The tracer slot of a dry (5-field) state; the `nothing` disables every
# tracer branch in the slab kernel at compile time.
@inline _positivity_tracer(states::Tuple{Any, Any, Any, Any, Any}) = nothing
@inline _positivity_tracer(states::Tuple{Any, Any, Any, Any, Any, Any}) =
    Fields.field_values(states[6])

@inline _positivity_slab(x, v, h) = slab(x, v, h)
@inline _positivity_slab(::Nothing, v, h) = nothing

function apply_positivity_limiter!(
    lim::PositivityLimiter,
    pfn::F,
    states,
    off,
    ::ClimaComms.AbstractCPUDevice,
) where {F}
    (ρ, ρe, u1, u2, u3) = states
    dρ = Fields.field_values(ρ)
    dρe = Fields.field_values(ρe)
    du1 = Fields.field_values(u1)
    du2 = Fields.field_values(u2)
    du3 = Fields.field_values(u3)
    dρq = _positivity_tracer(states)
    doff = Fields.field_values(off)
    dWJ = Spaces.local_geometry_data(axes(ρ)).WJ
    (Nv, _, _, Nh) = size(dρ)
    for h in 1:Nh, v in 1:Nv
        apply_positivity_slab!(
            lim, pfn,
            slab(dρ, v, h), slab(dρe, v, h),
            slab(du1, v, h), slab(du2, v, h), slab(du3, v, h),
            _positivity_slab(dρq, v, h), slab(doff, v, h), slab(dWJ, v, h),
        )
    end
    return nothing
end
