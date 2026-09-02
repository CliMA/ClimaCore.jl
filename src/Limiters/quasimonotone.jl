import ClimaComms
import ..Operators
import Adapt


abstract type AbstractLimiterConvergenceStats end

struct NoConvergenceStats <: AbstractLimiterConvergenceStats end

Base.@kwdef mutable struct LimiterConvergenceStats{FT} <:
                           AbstractLimiterConvergenceStats
    n_times_unconverged::Int = 0
    max_rel_err::FT = 0
    min_tracer_mass::FT = Inf
end

print_convergence_stats(io::IO, lcs::AbstractLimiterConvergenceStats) =
    print_convergence_stats(io, lcs)

print_convergence_stats(io::IO, ::NoConvergenceStats) = nothing
print_convergence_stats(io::IO, lcs::LimiterConvergenceStats) =
    print(io, convergence_stats_str(lcs))

function convergence_stats_str(lcs::LimiterConvergenceStats)
    return string(
        "Limiter convergence stats:\n",
        "     `n_times_unconverged = $(lcs.n_times_unconverged)`\n",
        "     `max_rel_err = $(lcs.max_rel_err)`\n",
        "     `min_tracer_mass = $(lcs.min_tracer_mass)`\n",
    )
end

function update_convergence_stats!(
    lcs::LimiterConvergenceStats,
    max_rel_err,
    min_tracer_mass,
)
    lcs.max_rel_err = max(lcs.max_rel_err, max_rel_err)
    lcs.min_tracer_mass = min(lcs.min_tracer_mass, min_tracer_mass)
    lcs.n_times_unconverged += 1
    return nothing
end

function reset_convergence_stats!(lcs::LimiterConvergenceStats)
    lcs.max_rel_err = 0
    lcs.min_tracer_mass = Inf
    lcs.n_times_unconverged = 0
    return nothing
end

"""
    QuasiMonotoneLimiter

Quasi-monotone limiter for tracer densities on spectral element spaces, after the OP1
limiter of [GubaOpt2014](@cite), eqs. (37)-(40). Quasi-monotone means monotone with
respect to the spectral element nodal values.

For each element, the limiter finds the nodal field closest, in the mass-weighted `l2`
norm, to the input field that satisfies min/max bounds on the concentration `q = ρq / ρ`.
As in HOMME, it clips the nodal values that violate the bounds to the nearest bound, which
changes the tracer mass of the element, and then redistributes the mass change over the
nodes that are not at a bound so that the `l2` error is smallest. Redistribution can
violate the bounds again, so the two steps are iterated until
`abs(Δtracer_mass) <= rtol * tracer_mass` or `Nq^2` iterations have been done. The
optimization is local to each element; the neighbors enter only through the bounds.

# Fields

  - `q_bounds`: min and max of `q` in each element.
  - `q_bounds_nbr`: min and max of `q` over each element and its neighbors.
  - `ghost_buffer`: buffer for exchanging `q_bounds` with neighboring processes.
  - `rtol`: relative tolerance for the tracer mass change per element [-].
  - `convergence_stats`: `LimiterConvergenceStats` (or `NoConvergenceStats`) accumulated by
    `apply_limiter!`.

# Constructor

    QuasiMonotoneLimiter(
        ρq::Field;
        rtol = eps(eltype(parent(ρq))),
        convergence_stats = LimiterConvergenceStats{eltype(parent(ρq))}(),
    )

Create a limiter for the tracer density field `ρq`, where `q` is the tracer concentration
per unit mass; `ρq` can be a scalar-valued or a struct-valued `Field`. `convergence_stats`
records the number of `apply_limiter!` calls in which some element failed to converge,
the largest relative mass error, and the smallest tracer mass;
`print_convergence_stats(limiter)` prints them.

# Examples

Call [`compute_bounds!`](@ref) on the fields at the start of the step, then
[`apply_limiter!`](@ref) on the updated fields:

```julia
limiter = QuasiMonotoneLimiter(ρq)
compute_bounds!(limiter, ρq, ρ)
# ... advance ρq and ρ ...
apply_limiter!(ρq, ρ, limiter)
```
"""
struct QuasiMonotoneLimiter{D, G, FT, CS}
    q_bounds::D
    q_bounds_nbr::D
    ghost_buffer::G
    rtol::FT
    convergence_stats::CS
end

print_convergence_stats(lim::QuasiMonotoneLimiter) =
    print_convergence_stats(stdout, lim.convergence_stats)

Adapt.adapt_structure(to, lim::QuasiMonotoneLimiter) = QuasiMonotoneLimiter(
    Adapt.adapt(to, lim.q_bounds),
    Adapt.adapt(to, lim.q_bounds_nbr),
    Adapt.adapt(to, lim.ghost_buffer),
    lim.rtol,
    Adapt.adapt(to, lim.convergence_stats),
)

function QuasiMonotoneLimiter(
    ρq::Fields.Field;
    rtol = eps(eltype(parent(ρq))),
    convergence_stats = LimiterConvergenceStats{eltype(parent(ρq))}(),
)
    q_bounds = make_q_bounds(Fields.field_values(ρq))
    ghost_buffer =
        Topologies.create_ghost_buffer(q_bounds, Spaces.topology(axes(ρq)))
    return QuasiMonotoneLimiter(
        q_bounds,
        similar(q_bounds),
        ghost_buffer,
        rtol,
        convergence_stats,
    )
end

function make_q_bounds(ρq::DataLayouts.VIJHWithF{S}) where {S}
    (; Nv, Nh, F) = DataLayouts.shape_params(ρq)
    Nf = DataLayouts.ncomponents(ρq)
    array = similar(parent(ρq), DataLayouts.add_f_dim((Nv, 2, 1, size(ρq, 4)), Nf, Val(F)))
    return DataLayouts.VIJHWithF{S, Nv, 2, 1, Nh, F}(array)
end

"""
    compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)

Compute the min and max of `q = ρq / ρ` over the nodes of each element and store them in
`limiter.q_bounds`.

Part of [`compute_bounds!`](@ref).
"""
function compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)
    compute_element_bounds!(limiter, ρq, ρ, ClimaComms.device(ρ))
end

function compute_element_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq,
    ρ,
    dev::ClimaComms.AbstractCPUDevice,
)
    ρ_data = Base.broadcastable(Fields.field_values(ρ))
    ρq_data = Base.broadcastable(Fields.field_values(ρq))
    q_bounds = limiter.q_bounds
    (Nv, Ni, Nj, Nh) = size(ρq_data)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρq = slab(ρq_data, v, h)
            slab_ρ = slab(ρ_data, v, h)
            local q_min, q_max
            for j in 1:Nj
                for i in 1:Ni
                    q = slab_ρq[1, i, j, 1] / slab_ρ[1, i, j, 1]
                    if i == 1 && j == 1
                        q_min = q
                        q_max = q
                    else
                        q_min = min(q_min, q)
                        q_max = max(q_max, q)
                    end
                end
            end
            slab_q_bounds = slab(q_bounds, v, h)
            slab_q_bounds[1] = q_min
            slab_q_bounds[2] = q_max
        end
    end
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds, limiter, ρq, ρ, dev)
    return nothing
end

"""
    compute_neighbor_bounds_local!(limiter::QuasiMonotoneLimiter, ρ)

Set `limiter.q_bounds_nbr` in each element to the min and max of `limiter.q_bounds` over
the element and its process-local neighbors in the topology of `axes(ρ)`.

Part of [`compute_bounds!`](@ref).
"""
compute_neighbor_bounds_local!(limiter::QuasiMonotoneLimiter, ρ) =
    compute_neighbor_bounds_local!(limiter, ρ, ClimaComms.device(ρ))

function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    dev::ClimaComms.AbstractCPUDevice,
)
    topology = Spaces.topology(axes(ρ))
    q_bounds = Base.broadcastable(limiter.q_bounds)
    q_bounds_nbr = limiter.q_bounds_nbr
    (Nv, _, _, Nh) = size(q_bounds_nbr)
    for h in 1:Nh
        for v in 1:Nv
            slab_q_bounds = slab(q_bounds, v, h)
            q_min = slab_q_bounds[1]
            q_max = slab_q_bounds[2]
            for h_nbr in Topologies.local_neighboring_elements(topology, h)
                slab_q_bounds = slab(q_bounds, v, h_nbr)
                q_min = min(q_min, slab_q_bounds[1])
                q_max = max(q_max, slab_q_bounds[2])
            end
            slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
            slab_q_bounds_nbr[1] = q_min
            slab_q_bounds_nbr[2] = q_max
        end
    end
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds_nbr, limiter, ρ, dev)
    return nothing
end

"""
    compute_neighbor_bounds_ghost!(limiter::QuasiMonotoneLimiter, topology)

Widen `limiter.q_bounds_nbr` in each element with the min and max of `limiter.q_bounds` in
its ghost neighbors, read from `limiter.ghost_buffer.recv_data`. Call it after the ghost
exchange has completed; it does nothing when `limiter.ghost_buffer` is not a
`GhostBuffer`.

!!! note

    This function indexes slabs of the receive buffer from the host, so the distributed
    limiter runs on CPUs only. With a `CUDADevice` and an `MPICommsContext`, it triggers
    scalar indexing of a `CuArray`.

Part of [`compute_bounds!`](@ref).
"""
function compute_neighbor_bounds_ghost!(
    limiter::QuasiMonotoneLimiter,
    topology::Topologies.AbstractTopology,
)
    q_bounds_nbr = limiter.q_bounds_nbr
    (Nv, _, _, Nh) = size(q_bounds_nbr)
    if limiter.ghost_buffer isa Topologies.GhostBuffer
        q_bounds_ghost = Base.broadcastable(limiter.ghost_buffer.recv_data)
        for h in 1:Nh
            for v in 1:Nv
                slab_q_bounds = slab(q_bounds_nbr, v, h)
                q_min = slab_q_bounds[1]
                q_max = slab_q_bounds[2]
                for gidx in Topologies.ghost_neighboring_elements(topology, h)
                    ghost_slab_q_bounds = slab(q_bounds_ghost, v, gidx)
                    q_min = min(q_min, ghost_slab_q_bounds[1])
                    q_max = max(q_max, ghost_slab_q_bounds[2])
                end
                slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
                slab_q_bounds_nbr[1] = q_min
                slab_q_bounds_nbr[2] = q_max
            end
        end
    end
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds_nbr, limiter, topology)
    return nothing
end

"""
    compute_bounds!(limiter::QuasiMonotoneLimiter, ρq::Field, ρ::Field)

Compute the bounds on the tracer concentration `q = ρq / ρ` that [`apply_limiter!`](@ref)
enforces, from the tracer density `ρq` and the density `ρ`, and store them in
`limiter.q_bounds_nbr`.

The steps are:

 1. [`compute_element_bounds!`](@ref) computes the min and max of `q` in each element.
 2. If distributed, start the ghost exchange of the element bounds.
 3. [`compute_neighbor_bounds_local!`](@ref) widens the bounds with the local neighbors.
 4. If distributed, complete the ghost exchange and widen the bounds with the ghost
    neighbors in [`compute_neighbor_bounds_ghost!`](@ref).
"""
function compute_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq::Fields.Field,
    ρ::Fields.Field,
)
    compute_element_bounds!(limiter, ρq, ρ)
    if limiter.ghost_buffer isa Topologies.GhostBuffer
        Spaces.fill_send_buffer!(
            Spaces.topology(axes(ρq)),
            limiter.q_bounds,
            limiter.ghost_buffer,
        )
        ClimaComms.start(limiter.ghost_buffer.graph_context)
    end
    compute_neighbor_bounds_local!(limiter, ρ)
    if limiter.ghost_buffer isa Topologies.GhostBuffer
        ClimaComms.finish(limiter.ghost_buffer.graph_context)
        compute_neighbor_bounds_ghost!(limiter, Spaces.topology(axes(ρq)))
    end
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds, limiter, ρq, ρ)
end




"""
    apply_limiter!(ρq, ρ, limiter::QuasiMonotoneLimiter; warn = true)

Limit the tracer density `ρq` in place so that the concentration `q = ρq / ρ` at each
node lies within the bounds computed by [`compute_bounds!`](@ref), while preserving the
tracer mass of each element up to the relative tolerance `limiter.rtol`.

Each element is processed by [`apply_limit_slab!`](@ref), with the density `ρ` times the
quadrature weights as the weights of the least-squares redistribution. When some element
fails to converge, `limiter.convergence_stats` is updated. On CPU devices with
`warn = true`, the accumulated convergence statistics are printed with `@warn` after each
call.

Return `ρq` on CPU devices and `nothing` on CUDA devices.
"""
apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter;
    warn::Bool = true,
) = apply_limiter!(ρq, ρ, limiter, ClimaComms.device(ρ); warn)

function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
    dev::ClimaComms.AbstractCPUDevice;
    warn::Bool = true,
)
    (; q_bounds_nbr, rtol) = limiter

    ρq_data = Fields.field_values(ρq)
    ρ_data = Fields.field_values(ρ)
    WJ_data = Spaces.local_geometry_data(axes(ρq)).WJ

    converged = true
    max_rel_err = zero(rtol)
    min_tracer_mass = Inf
    (Nv, _, _, Nh) = size(ρq_data)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρ = slab(ρ_data, v, h)
            slab_ρq = slab(ρq_data, v, h)
            slab_WJ = slab(WJ_data, v, h)
            slab_q_bounds = slab(q_bounds_nbr, v, h)
            (_converged, slab_max_rel_err, slab_min_tracer_mass) =
                apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)
            converged &= _converged
            max_rel_err = max(slab_max_rel_err, max_rel_err)
            min_tracer_mass = max(min_tracer_mass, slab_min_tracer_mass)
        end
    end
    if !converged
        lcs = limiter.convergence_stats
        update_convergence_stats!(lcs, max_rel_err, min_tracer_mass)
    end
    if warn
        lcs = limiter.convergence_stats
        msg = convergence_stats_str(lcs)
        msg *= "Use `warn = false` in `Limiters.apply_limiter!` to suppress this message.\n"
        @warn msg
    end

    call_post_op_callback() && post_op_callback(ρq, ρq, ρ, limiter, dev)
    return ρq
end

# One scalar view per component of a layout's element type, or the layout
# itself when its element type has no fields (a single scalar component). The
# views are wrapped in Vals because property_view requires a type-domain field
# index, so a runtime index would trigger dynamic dispatch in GPU kernels.
@inline component_views(data) =
    iszero(fieldcount(eltype(data))) ? (data,) :
    unrolled_map(
        i -> DataLayouts.property_view(data, i),
        ntuple(Val, Val(fieldcount(eltype(data)))),
    )

# Compute ∫ρ in its own function, so that the total is not reassigned in
# apply_limit_slab!, where its capture by the unrolled_map closure would
# require it to be wrapped in a Core.Box
@inline function slab_total_mass(slab_ρ, slab_WJ, Ni, Nj)
    total_mass = zero(eltype(parent(slab_ρ)))
    for j in 1:Nj, i in 1:Ni
        total_mass += slab_ρ[1, i, j, 1] * slab_WJ[1, i, j, 1]
    end
    return total_mass
end

"""
    apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)

Limit the nodal values of one element's tracer density `slab_ρq` in place so that
`q = ρq / ρ` lies within `slab_q_bounds`, the min and max for the element, given the
density `slab_ρ` and the quadrature weights times Jacobians `slab_WJ`.

The bounds are first widened to include the element mean of `q`, so that a solution exists.
Values outside the bounds are clipped, and the resulting tracer mass change is
redistributed over the nodes not at a bound, in proportion to `slab_ρ * slab_WJ`, until the
relative mass change is at most `rtol` or `Nq^2` iterations have been done. Each component
of a struct-valued `slab_ρq` is limited independently against the matching component of
`slab_q_bounds`.

# Returns

A tuple `(converged, max_rel_err, min_tracer_mass)`: whether all components met the
tolerance, the largest relative mass error, and the smallest absolute tracer mass over the
components.
"""
function apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)
    (_, Ni, Nj, _) = size(slab_ρq)

    total_mass = slab_total_mass(slab_ρ, slab_WJ, Ni, Nj)
    @assert total_mass > 0

    field_results = unrolled_map(
        (field_ρq, field_q_bounds) -> apply_limit_slab_field!(
            field_ρq,
            slab_ρ,
            slab_WJ,
            field_q_bounds,
            total_mass,
            rtol,
        ),
        component_views(slab_ρq),
        component_views(slab_q_bounds),
    )
    return unrolled_reduce(field_results) do result1, result2
        (
            result1[1] && result2[1],
            max(result1[2], result2[2]),
            min(result1[3], result2[3]),
        )
    end
end

# Apply the limit for one component of ρq, given views of that component in
# slab_ρq and slab_q_bounds. Return whether the tolerance condition could be
# satisfied, along with the maximum relative error and the minimum tracer mass.
function apply_limit_slab_field!(
    field_ρq,
    slab_ρ,
    slab_WJ,
    field_q_bounds,
    total_mass,
    rtol,
)
    FT = eltype(parent(field_ρq))
    (_, Ni, Nj, _) = size(field_ρq)
    maxiter = Ni * Nj

    q_min = field_q_bounds[1, 1, 1, 1]
    q_max = field_q_bounds[1, 2, 1, 1]

    converged = true
    max_rel_err = zero(rtol)
    min_tracer_mass = FT(Inf)
    rel_err = zero(FT)

    # 2) compute ∫ρq
    tracer_mass = zero(FT)
    for j in 1:Nj, i in 1:Ni
        tracer_mass += field_ρq[1, i, j, 1] * slab_WJ[1, i, j, 1]
    end

    # TODO: Should this condition be enforced? (It isn't in HOMME.)
    # @assert tracer_mass >= 0

    # 3) set bounds
    q_avg = tracer_mass / total_mass
    q_min = min(q_min, q_avg)
    q_max = max(q_max, q_avg)

    # 3) modify ρq
    for iter in 1:maxiter
        Δtracer_mass = zero(FT)
        for j in 1:Nj, i in 1:Ni
            ρ = slab_ρ[1, i, j, 1]
            ρq = field_ρq[1, i, j, 1]
            ρq_max = ρ * q_max
            ρq_min = ρ * q_min
            w = slab_WJ[1, i, j, 1]
            if ρq > ρq_max
                Δtracer_mass += (ρq - ρq_max) * w
                field_ρq[1, i, j, 1] = ρq_max
            elseif ρq < ρq_min
                Δtracer_mass += (ρq - ρq_min) * w
                field_ρq[1, i, j, 1] = ρq_min
            end
        end

        rel_err = abs(Δtracer_mass) / abs(tracer_mass)
        max_rel_err = max(max_rel_err, rel_err)
        min_tracer_mass = min(min_tracer_mass, abs(tracer_mass))
        if rel_err <= rtol
            break
        end

        if Δtracer_mass > 0 # add mass
            total_mass_at_Δ_points = zero(FT)
            for j in 1:Nj, i in 1:Ni
                ρ = slab_ρ[1, i, j, 1]
                ρq = field_ρq[1, i, j, 1]
                w = slab_WJ[1, i, j, 1]
                if ρq < ρ * q_max
                    total_mass_at_Δ_points += ρ * w
                end
            end
            Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
            for j in 1:Nj, i in 1:Ni
                ρ = slab_ρ[1, i, j, 1]
                ρq = field_ρq[1, i, j, 1]
                if ρq < ρ * q_max
                    field_ρq[1, i, j, 1] += ρ * Δq_at_Δ_points
                end
            end
        else # remove mass
            total_mass_at_Δ_points = zero(FT)
            for j in 1:Nj, i in 1:Ni
                ρ = slab_ρ[1, i, j, 1]
                ρq = field_ρq[1, i, j, 1]
                w = slab_WJ[1, i, j, 1]
                if ρq > ρ * q_min
                    total_mass_at_Δ_points += ρ * w
                end
            end
            Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
            for j in 1:Nj, i in 1:Ni
                ρ = slab_ρ[1, i, j, 1]
                ρq = field_ρq[1, i, j, 1]
                if ρq > ρ * q_min
                    field_ρq[1, i, j, 1] += ρ * Δq_at_Δ_points
                end
            end
        end

        if iter == maxiter
            converged = false
        end
    end
    return (converged, max_rel_err, min_tracer_mass)
end
