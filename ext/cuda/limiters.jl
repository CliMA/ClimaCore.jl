import ClimaCore.Limiters:
    QuasiMonotoneLimiter,
    compute_element_bounds!,
    compute_neighbor_bounds_local!,
    apply_limiter!
import ClimaCore.Fields
import ClimaCore: DataLayouts, Spaces, Topologies, Fields
using CUDA

function config_threadblock(Nv, Nh)
    nitems = Nv * Nh
    nthreads = min(256, nitems)
    nblocks = cld(nitems, nthreads)
    return (nthreads, nblocks)
end

function compute_element_bounds!(
    limiter::QuasiMonotoneLimiter,
    ρq,
    ρ,
    ::ClimaComms.CUDADevice,
)
    S = size(Fields.field_values(ρ))
    (Ni, Nj, _, Nv, Nh) = S
    nthreads, nblocks = config_threadblock(Nv, Nh)

    args = (
        limiter,
        Fields.field_values(Operators.strip_space(ρq, axes(ρq))),
        Fields.field_values(Operators.strip_space(ρ, axes(ρ))),
        Nv,
        Nh,
        Val(Ni),
        Val(Nj),
    )
    auto_launch!(
        compute_element_bounds_kernel!,
        args,
        ρ;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    return nothing
end


function compute_element_bounds_kernel!(
    limiter,
    ρq,
    ρ,
    Nv,
    Nh,
    ::Val{Ni},
    ::Val{Nj},
) where {Ni, Nj}
    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I
        (; q_bounds) = limiter
        local q_min, q_max
        slab_ρq = slab(ρq, v, h)
        slab_ρ = slab(ρ, v, h)
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
    return nothing
end


function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    ::ClimaComms.CUDADevice,
)
    topology = Spaces.topology(axes(ρ))
    Ni, Nj, _, Nv, Nh = size(Fields.field_values(ρ))
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (
        limiter,
        topology.local_neighbor_elem,
        topology.local_neighbor_elem_offset,
        Nv,
        Nh,
        Val(Ni),
        Val(Nj),
    )
    auto_launch!(
        compute_neighbor_bounds_local_kernel!,
        args,
        ρ;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
end

function compute_neighbor_bounds_local_kernel!(
    limiter,
    local_neighbor_elem,
    local_neighbor_elem_offset,
    Nv,
    Nh,
    ::Val{Ni},
    ::Val{Nj},
) where {Ni, Nj}

    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I
        (; q_bounds, q_bounds_nbr, ghost_buffer, rtol) = limiter
        slab_q_bounds = slab(q_bounds, v, h)
        q_min = slab_q_bounds[1]
        q_max = slab_q_bounds[2]
        for lne in
            local_neighbor_elem_offset[h]:(local_neighbor_elem_offset[h + 1] - 1)
            h_nbr = local_neighbor_elem[lne]
            slab_q_bounds = slab(q_bounds, v, h_nbr)
            q_min = rmin(q_min, slab_q_bounds[1])
            q_max = rmax(q_max, slab_q_bounds[2])
        end
        slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
        slab_q_bounds_nbr[1] = q_min
        slab_q_bounds_nbr[2] = q_max
    end
    return nothing
end

function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
    ::ClimaComms.CUDADevice,
)
    ρq_data = Fields.field_values(ρq)
    (Ni, Nj, _, Nv, Nh) = size(ρq_data)
    Nf = DataLayouts.ncomponents(ρq_data)
    maxiter = Ni * Nj
    WJ = Spaces.local_geometry_data(axes(ρq)).WJ
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (
        limiter,
        Fields.field_values(Operators.strip_space(ρq, axes(ρq))),
        Fields.field_values(Operators.strip_space(ρ, axes(ρ))),
        WJ,
        Nv,
        Nh,
        Val(Nf),
        Val(Ni),
        Val(Nj),
        Val(maxiter),
    )
    auto_launch!(
        apply_limiter_kernel!,
        args,
        ρ;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    return nothing
end

function apply_limiter_kernel!(
    limiter::QuasiMonotoneLimiter,
    ρq_data,
    ρ_data,
    WJ_data,
    Nv,
    Nh,
    ::Val{Nf},
    ::Val{Ni},
    ::Val{Nj},
    ::Val{maxiter},
) where {Nf, Ni, Nj, maxiter}
    (; q_bounds_nbr, rtol) = limiter
    converged = true
    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I

        slab_ρ = slab(ρ_data, v, h)
        slab_ρq = slab(ρq_data, v, h)
        slab_WJ = slab(WJ_data, v, h)
        slab_q_bounds = slab(q_bounds_nbr, v, h)

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

    end
    # converged || @warn "Limiter failed to converge with rtol = $rtol"

    return nothing
end
