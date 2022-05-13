"""
    QuasiMonotoneLimiter

This limiter is inspired by the one presented in Guba et al
[GubaOpt2014](@cite). In the reference paper, it is denoted by OP1, and is
outlined in eqs. (37)-(40). Quasimonotone here is meant to be monotone with
respect to the spectral element nodal values. This limiter involves solving a
constrained optimization problem (a weighted least square problem up to a fixed
tolerance) that is completely local to each element.

As in HOMME, the implementation idea here is the following: we need to find a
grid field which is closest to the initial field (in terms of weighted sum), but
satisfies the min/max constraints. So, first we find values that do not satisfy
constraints and bring these values to a closest constraint. This way we
introduce some mass change (`mass_change`), which we then redistribute so that
the l2 error is smallest. This redistribution might violate constraints; thus,
we do a few iterations (typically a couple).

- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit
  mass. This can be a scalar field, or a struct-valued field.
- `ρ`: fluid density Field (scalar).

# Constructor

    limiter = QuasiMonotoneLimiter(ρq::Field, ρ::Field)

Creates a limiter instance for the field `ρq` and the field `ρ`.

# Usage

Call [`compute_bounds!`](@ref) on the input fields:

    compute_bounds!(limiter, ρq, ρ)

Then call [`apply_limiter!`](@ref) on the output fields:

    apply_limiter!(ρq, ρ, limiter)
"""
struct QuasiMonotoneLimiter{D, G}
    "contains the min and max of each element"
    q_bounds::D
    "contains the min and max of each element and its neighbors"
    q_bounds_nbr::D
    "communication buffer"
    ghost_buffer::G
end


function QuasiMonotoneLimiter(ρq::Fields.Field, ρ::Fields.Field)
    data = Fields.field_values(ρq)
    topology = Spaces.topology(axes(ρq))
    QuasiMonotoneLimiter(data, topology)
end

function QuasiMonotoneLimiter(
    data::Union{DataLayouts.IFH{S}, DataLayouts.IJFH{S}},
    topology::Topologies.AbstractTopology,
) where {S}
    Nf = DataLayouts.ncomponents(data)
    _, _, _, Nv, Nh = size(data)
    q_bounds = DataLayouts.IFH{S, 2}(similar(parent(data), (2, Nf, Nh)))
    q_bounds_nbr = similar(q_bounds)
    QuasiMonotoneLimiter(
        q_bounds,
        q_bounds_nbr,
        Spaces.create_ghost_buffer(q_bounds, topology),
    )
end
function QuasiMonotoneLimiter(
    data::Union{DataLayouts.VIFH{S}, DataLayouts.VIJFH{S}},
    topology::Topologies.AbstractTopology,
) where {S}
    Nf = DataLayouts.ncomponents(data)
    _, _, _, Nv, Nh = size(data)
    q_bounds = DataLayouts.VIFH{S, 2}(similar(parent(data), (Nv, 2, Nf, Nh)))
    q_bounds_nbr = similar(q_bounds)
    QuasiMonotoneLimiter(
        q_bounds,
        q_bounds_nbr,
        Spaces.create_ghost_buffer(q_bounds, topology),
    )
end


"""
    compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)

Given two fields `ρq` and `ρ`, computes the min and max of `q` in each element,
storing it in `limiter.q_bounds`.

Part of [`compute_bounds!`](@ref).
"""
function compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)
    q_bounds = limiter.q_bounds
    (Ni, Nj, _, Nv, Nh) = size(ρq)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρq = slab(ρq, v, h)
            slab_ρ = slab(ρ, v, h)
            local q_min, q_max
            for j in 1:Nj
                for i in 1:Ni
                    q = rdiv(slab_ρq[i, j], slab_ρ[i, j])
                    if i == 1 && j == 1
                        q_min = q
                        q_max = q
                    else
                        q_min = rmin(q_min, q)
                        q_max = rmax(q_max, q)
                    end
                end
            end
            slab_q_bounds = slab(q_bounds, v, h)
            slab_q_bounds[1] = q_min
            slab_q_bounds[2] = q_max
        end
    end
    return nothing
end

"""
    compute_neighbor_bounds_local!(limiter::QuasiMonotoneLimiter, topology)

Update the field `limiter.q_bounds_nbr` based on `limiter.q_bounds` in the local
neighbors.

Part of [`compute_bounds!`](@ref).
"""
function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    topology::Topologies.AbstractTopology,
)
    q_bounds = limiter.q_bounds
    q_bounds_nbr = limiter.q_bounds_nbr
    (_, _, _, Nv, Nh) = size(q_bounds_nbr)
    for h in 1:Nh
        for v in 1:Nv
            slab_q_bounds = slab(q_bounds, v, h)
            q_min = slab_q_bounds[1]
            q_max = slab_q_bounds[2]
            for h_nbr in Topologies.local_neighboring_elements(topology, h)
                slab_q_bounds = slab(q_bounds, v, h_nbr)
                q_min = rmin(q_min, slab_q_bounds[1])
                q_max = rmax(q_max, slab_q_bounds[2])
            end
            slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
            slab_q_bounds_nbr[1] = q_min
            slab_q_bounds_nbr[2] = q_max
        end
    end
    return nothing
end

"""
    compute_neighbor_bounds_ghost!(limiter::QuasiMonotoneLimiter, topology)

Update the field `limiter.q_bounds_nbr` based on `limiter.q_bounds` in the ghost
neighbors. This should be called after the ghost exchange has completed.

Part of [`compute_bounds!`](@ref).
"""
function compute_neighbor_bounds_ghost!(
    limiter::QuasiMonotoneLimiter,
    topology::Topologies.AbstractTopology,
)
    q_bounds_nbr = limiter.q_bounds_nbr
    (_, _, _, Nv, Nh) = size(q_bounds_nbr)
    if limiter.ghost_buffer isa Spaces.GhostBuffer
        q_bounds_ghost = limiter.ghost_buffer.recv_data

        for h in 1:Nh
            for v in 1:Nv
                slab_q_bounds = slab(q_bounds_nbr, v, h)
                q_min = slab_q_bounds[1]
                q_max = slab_q_bounds[2]
                for gidx in Topologies.ghost_neighboring_elements(topology, h)
                    ghost_slab_q_bounds = slab(q_bounds_ghost, v, gidx)
                    q_min = rmin(q_min, ghost_slab_q_bounds[1])
                    q_max = rmax(q_max, ghost_slab_q_bounds[2])
                end
                slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
                slab_q_bounds_nbr[1] = q_min
                slab_q_bounds_nbr[2] = q_max
            end
        end
    end
    return nothing
end

"""
    compute_bounds!(limiter::QuasiMonotoneLimiter, ρq::Field, ρ::Field)

Compute the desired bounds for the tracer concentration per unit mass `q`, based
on the tracer density, `ρq`, and density, `ρ`, fields.

This is computed by
 1. [`compute_element_bounds!`](@ref)
 2. starts the ghost exchange (if distributed)
 3. [`compute_neighbor_bounds_local!`](@ref)
 4. completes the ghost exchange (if distributed)
 5. [`compute_neighbor_bounds_ghost!`](@ref) (if distributed)
"""
function compute_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq::Fields.Field,
    ρ::Fields.Field,
)
    topology = Spaces.topology(axes(ρq))
    compute_bounds!(
        limiter,
        Fields.field_values(ρq),
        Fields.field_values(ρ),
        topology,
    )
end
function compute_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq,
    ρ,
    topology::Topologies.AbstractTopology,
)
    compute_element_bounds!(limiter, ρq, ρ)
    if limiter.ghost_buffer isa Spaces.GhostBuffer
        Spaces.fill_send_buffer!(
            topology,
            limiter.q_bounds,
            limiter.ghost_buffer,
        )
        ClimaComms.start(limiter.ghost_buffer.graph_context)
    end
    compute_neighbor_bounds_local!(limiter, topology)
    if limiter.ghost_buffer isa Spaces.GhostBuffer
        ClimaComms.finish(ghost_buffer.graph_context)
        compute_neighbor_bounds_ghost!(limiter, topology)
    end
end




"""
    apply_limiter!(ρq, ρ, limiter::QuasiMonotoneLimiter)

Apply the limiter on the tracer density  `ρq`, using the computed desired bounds
on the concentration `q` and density, `ρ`, as an optimal weight. This iterates
over each element, calling [`apply_limit_slab!`](@ref).
"""
function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
)

    space = axes(ρq)

    # Initialize temp variables

    ρq_data = Fields.field_values(ρq)
    ρ_data = Fields.field_values(ρ)
    WJ_data = Spaces.local_geometry_data(space).WJ

    (_, _, _, Nv, Nh) = size(ρq_data)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρ = slab(ρ_data, v, h)
            slab_ρq = slab(ρq_data, v, h)
            slab_wj = slab(WJ_data, v, h)
            slab_q_bounds = slab(limiter.q_bounds_nbr, v, h)
            apply_limit_slab!(slab_ρq, slab_ρ, slab_wj, slab_q_bounds)
        end # end of horz elem loop
    end # end of vert level loop
    return ρq
end

"""
    apply_limit_slab!(slab_ρq, slab_ρ, slab_wj, slab_q_bounds)

Apply the computed bounds of the tracer concentration (`slab_q_bounds`) in the
limiter to `slab_ρq`, given the total mass `slab_ρ` and weights `slab_wj`.
"""
function apply_limit_slab!(
    slab_ρq::DataLayouts.IJF{<:Any, Nij},
    slab_ρ::DataLayouts.IJF{<:Any, Nij},
    slab_WJ::DataLayouts.IJF{<:Any, Nij},
    slab_q_bounds::DataLayouts.IF{<:Any, 2},
) where {Nij}

    Nf = DataLayouts.ncomponents(slab_ρq)

    array_ρq = parent(slab_ρq)
    array_ρ = parent(slab_ρ)
    array_w = parent(slab_WJ)
    array_q_bounds = parent(slab_q_bounds)

    rtol = sqrt(eps(eltype(array_ρq)))
    maxiter = Nij * Nij

    # 1) compute ρ_tot
    ρ_tot = zero(eltype(array_ρ))
    for j in 1:Nij, i in 1:Nij
        ρ_tot += array_ρ[i, j, 1] * array_w[i, j, 1]
    end

    @assert ρ_tot > 0

    for f in 1:Nf
        q_min = array_q_bounds[1, f]
        q_max = array_q_bounds[2, f]

        # 2) compute ∫ρq
        ρq_tot = zero(eltype(array_ρq))
        for j in 1:Nij, i in 1:Nij
            ρq_tot += array_ρq[i, j, f] * array_w[i, j, 1]
        end

        # 3) set bounds
        q_avg = ρq_tot / ρ_tot
        q_min = min(q_min, q_avg)
        q_max = max(q_max, q_avg)

        # 3) compute total mass change
        for iter in 1:maxiter
            Δρq = zero(eltype(array_ρq))
            for j in 1:Nij, i in 1:Nij
                ρ = array_ρ[i, j, 1]
                ρq = array_ρq[i, j, f]
                ρq_max = ρ * q_max
                ρq_min = ρ * q_min
                w = array_w[i, j]
                if ρq > ρq_max
                    Δρq += (ρq - ρq_max) * w
                    array_ρq[i, j, f] = ρq_max
                elseif ρq < ρq_min
                    Δρq += (ρq - ρq_min) * w
                    array_ρq[i, j, f] = ρq_min
                end
            end

            if abs(Δρq) < rtol * ρq_tot
                break
            end

            if Δρq > 0 # add mass
                # compute total density
                Δρ = zero(eltype(array_ρ))
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq < ρ * q_max
                        Δρ += ρ * w
                    end
                end
                Δq = Δρq / Δρ # compute average ratio change
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq < ρ * q_max
                        array_ρq[i, j, f] += ρ * Δq
                    end
                end
            else # remove mass
                Δρ = zero(eltype(array_ρ))
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq > ρ * q_min
                        Δρ += ρ * w
                    end
                end
                Δq = Δρq / Δρ
                for j in 1:Nij, i in 1:Nij
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq > ρ * q_min
                        array_ρq[i, j, f] += ρ * Δq
                    end
                end
            end
        end
    end
end
