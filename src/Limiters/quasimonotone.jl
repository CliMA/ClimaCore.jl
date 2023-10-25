import ClimaComms
import CUDA

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
introduce some change in the tracer mass, which we then redistribute so that the
l2 error is smallest. This redistribution might violate constraints; thus, we do
a few iterations (until `abs(Δtracer_mass) <= rtol * tracer_mass`).

- `ρq`: tracer density Field, where `q` denotes tracer concentration per unit
  mass. This can be a scalar field, or a struct-valued field.
- `ρ`: fluid density Field (scalar).

# Constructor

    limiter = QuasiMonotoneLimiter(ρq::Field; rtol = eps(eltype(parent(ρq))))

Creates a limiter instance for the field `ρq` with relative tolerance `rtol`.

# Usage

Call [`compute_bounds!`](@ref) on the input fields:

    compute_bounds!(limiter, ρq, ρ)

Then call [`apply_limiter!`](@ref) on the output fields:

    apply_limiter!(ρq, ρ, limiter)
"""
struct QuasiMonotoneLimiter{D, G, FT}
    "contains the min and max of each element"
    q_bounds::D
    "contains the min and max of each element and its neighbors"
    q_bounds_nbr::D
    "communication buffer"
    ghost_buffer::G
    "relative tolerance for tracer mass change"
    rtol::FT
end


function QuasiMonotoneLimiter(ρq::Fields.Field; rtol = eps(eltype(parent(ρq))))
    q_bounds = make_q_bounds(Fields.field_values(ρq))
    ghost_buffer =
        Topologies.create_ghost_buffer(q_bounds, Spaces.topology(axes(ρq)))
    return QuasiMonotoneLimiter(q_bounds, similar(q_bounds), ghost_buffer, rtol)
end

Base.@deprecate(
    QuasiMonotoneLimiter(ρq::Fields.Field, ρ::Fields.Field),
    QuasiMonotoneLimiter(ρq::Fields.Field; rtol = eps(eltype(parent(ρq)))),
)

function make_q_bounds(
    ρq::Union{DataLayouts.IFH{S}, DataLayouts.IJFH{S}},
) where {S}
    Nf = DataLayouts.ncomponents(ρq)
    _, _, _, _, Nh = size(ρq)
    return DataLayouts.IFH{S, 2}(similar(parent(ρq), (2, Nf, Nh)))
end
function make_q_bounds(
    ρq::Union{DataLayouts.VIFH{S}, DataLayouts.VIJFH{S}},
) where {S}
    Nf = DataLayouts.ncomponents(ρq)
    _, _, _, Nv, Nh = size(ρq)
    return DataLayouts.VIFH{S, 2}(similar(parent(ρq), (Nv, 2, Nf, Nh)))
end


"""
    compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)

Given two fields `ρq` and `ρ`, computes the min and max of `q` in each element,
storing it in `limiter.q_bounds`.

Part of [`compute_bounds!`](@ref).
"""
function compute_element_bounds!(limiter::QuasiMonotoneLimiter, ρq, ρ)
    compute_element_bounds!(limiter, ρq, ρ, ClimaComms.device(ρ))
end

function compute_element_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq,
    ρ,
    ::ClimaComms.CUDADevice,
) end

function compute_element_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq,
    ρ,
    ::ClimaComms.AbstractCPUDevice,
)
    ρ_data = Fields.field_values(ρ)
    ρq_data = Fields.field_values(ρq)
    q_bounds = limiter.q_bounds
    (Ni, Nj, _, Nv, Nh) = size(ρq_data)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρq = slab(ρq_data, v, h)
            slab_ρ = slab(ρ_data, v, h)
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
compute_neighbor_bounds_local!(limiter::QuasiMonotoneLimiter, ρ) =
    compute_neighbor_bounds_local!(limiter, ρ, ClimaComms.device(ρ))

function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    ::ClimaComms.CUDADevice,
) end

function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    ::ClimaComms.AbstractCPUDevice,
)
    topology = Spaces.topology(axes(ρ))
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
    compute_element_bounds!(limiter, ρq, ρ)
    if limiter.ghost_buffer isa Spaces.GhostBuffer
        Spaces.fill_send_buffer!(
            Spaces.topology(axes(ρq)),
            limiter.q_bounds,
            limiter.ghost_buffer,
        )
        ClimaComms.start(limiter.ghost_buffer.graph_context)
    end
    compute_neighbor_bounds_local!(limiter, ρ)
    if limiter.ghost_buffer isa Spaces.GhostBuffer
        ClimaComms.finish(limiter.ghost_buffer.graph_context)
        compute_neighbor_bounds_ghost!(limiter, Spaces.topology(axes(ρq)))
    end
end




"""
    apply_limiter!(ρq, ρ, limiter::QuasiMonotoneLimiter)

Apply the limiter on the tracer density  `ρq`, using the computed desired bounds
on the concentration `q` and density `ρ` as an optimal weight. This iterates
over each element, calling [`apply_limit_slab!`](@ref). If the limiter fails to
converge for any element, a warning is issued.
"""
apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
) = apply_limiter!(ρq, ρ, limiter, ClimaComms.device(ρ))

function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
    ::ClimaComms.CUDADevice,
) end

function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
    ::ClimaComms.AbstractCPUDevice,
)
    (; q_bounds_nbr, rtol) = limiter

    ρq_data = Fields.field_values(ρq)
    ρ_data = Fields.field_values(ρ)
    WJ_data = Spaces.local_geometry_data(axes(ρq)).WJ

    converged = true
    (_, _, _, Nv, Nh) = size(ρq_data)
    for h in 1:Nh
        for v in 1:Nv
            slab_ρ = slab(ρ_data, v, h)
            slab_ρq = slab(ρq_data, v, h)
            slab_WJ = slab(WJ_data, v, h)
            slab_q_bounds = slab(q_bounds_nbr, v, h)
            converged &=
                apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)
        end
    end
    converged || @warn "Limiter failed to converge with rtol = $rtol"

    return ρq
end

"""
    apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)

Apply the computed bounds of the tracer concentration (`slab_q_bounds`) in the
limiter to `slab_ρq`, given the total mass `slab_ρ`, metric terms `slab_WJ`,
and relative tolerance `rtol`. Return whether the tolerance condition could be
satisfied.
"""
function apply_limit_slab!(slab_ρq, slab_ρ, slab_WJ, slab_q_bounds, rtol)
    Nf = DataLayouts.ncomponents(slab_ρq)
    (Ni, Nj, _, _, _) = size(slab_ρq)
    maxiter = Ni * Nj

    array_ρq = parent(slab_ρq)
    array_ρ = parent(slab_ρ)
    array_w = parent(slab_WJ)
    array_q_bounds = parent(slab_q_bounds)

    # 1) compute ∫ρ
    total_mass = zero(eltype(array_ρ))
    for j in 1:Nj, i in 1:Ni
        total_mass += array_ρ[i, j, 1] * array_w[i, j, 1]
    end

    @assert total_mass > 0

    converged = true
    for f in 1:Nf
        q_min = array_q_bounds[1, f]
        q_max = array_q_bounds[2, f]

        # 2) compute ∫ρq
        tracer_mass = zero(eltype(array_ρq))
        for j in 1:Nj, i in 1:Ni
            tracer_mass += array_ρq[i, j, f] * array_w[i, j, 1]
        end

        # TODO: Should this condition be enforced? (It isn't in HOMME.)
        # @assert tracer_mass >= 0

        # 3) set bounds
        q_avg = tracer_mass / total_mass
        q_min = min(q_min, q_avg)
        q_max = max(q_max, q_avg)

        # 3) modify ρq
        for iter in 1:maxiter
            Δtracer_mass = zero(eltype(array_ρq))
            for j in 1:Nj, i in 1:Ni
                ρ = array_ρ[i, j, 1]
                ρq = array_ρq[i, j, f]
                ρq_max = ρ * q_max
                ρq_min = ρ * q_min
                w = array_w[i, j]
                if ρq > ρq_max
                    Δtracer_mass += (ρq - ρq_max) * w
                    array_ρq[i, j, f] = ρq_max
                elseif ρq < ρq_min
                    Δtracer_mass += (ρq - ρq_min) * w
                    array_ρq[i, j, f] = ρq_min
                end
            end

            if abs(Δtracer_mass) <= rtol * abs(tracer_mass)
                break
            end

            if Δtracer_mass > 0 # add mass
                total_mass_at_Δ_points = zero(eltype(array_ρ))
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq < ρ * q_max
                        total_mass_at_Δ_points += ρ * w
                    end
                end
                Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq < ρ * q_max
                        array_ρq[i, j, f] += ρ * Δq_at_Δ_points
                    end
                end
            else # remove mass
                total_mass_at_Δ_points = zero(eltype(array_ρ))
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    w = array_w[i, j]
                    if ρq > ρ * q_min
                        total_mass_at_Δ_points += ρ * w
                    end
                end
                Δq_at_Δ_points = Δtracer_mass / total_mass_at_Δ_points
                for j in 1:Nj, i in 1:Ni
                    ρ = array_ρ[i, j, 1]
                    ρq = array_ρq[i, j, f]
                    if ρq > ρ * q_min
                        array_ρq[i, j, f] += ρ * Δq_at_Δ_points
                    end
                end
            end

            if iter == maxiter
                converged = false
            end
        end
    end
    return converged
end
