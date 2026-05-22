import ClimaCore.Limiters:
    QuasiMonotoneLimiter,
    compute_element_bounds!,
    compute_neighbor_bounds_local!,
    apply_limiter!,
    VerticalMassBorrowingLimiter,
    column_massborrow!
import ClimaCore.Fields
import ClimaCore: DataLayouts, Spaces, Topologies, Fields
import ClimaCore.DataLayouts: slab_index, getindex_field, setindex_field!, column
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
    dev::ClimaComms.CUDADevice,
)
    ρ_values = Base.broadcastable(
        Fields.field_values(Operators.strip_space(ρ, axes(ρ))),
    )
    ρq_values = Base.broadcastable(
        Fields.field_values(Operators.strip_space(ρq, axes(ρq))),
    )
    (_, _, _, Nv, Nh) = DataLayouts.universal_size(ρ_values)
    nthreads, nblocks = config_threadblock(Nv, Nh)

    args = (limiter, ρq_values, ρ_values)
    auto_launch!(
        compute_element_bounds_kernel!,
        args;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds, limiter, ρq, ρ, dev)
    return nothing
end


function compute_element_bounds_kernel!(limiter, ρq, ρ)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(ρ)
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
                q = slab_ρq[slab_index(i, j)] / slab_ρ[slab_index(i, j)]
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
        slab_q_bounds[slab_index(1)] = q_min
        slab_q_bounds[slab_index(2)] = q_max
    end
    return nothing
end


function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    dev::ClimaComms.CUDADevice,
)
    topology = Spaces.topology(axes(ρ))
    us = DataLayouts.UniversalSize(Fields.field_values(ρ))
    (_, _, _, Nv, Nh) = DataLayouts.universal_size(us)
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (
        limiter,
        topology.local_neighbor_elem,
        topology.local_neighbor_elem_offset,
        us,
    )
    auto_launch!(
        compute_neighbor_bounds_local_kernel!,
        args;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    call_post_op_callback() &&
        post_op_callback(limiter.q_bounds, limiter, ρ, dev)
end

function compute_neighbor_bounds_local_kernel!(
    limiter,
    local_neighbor_elem,
    local_neighbor_elem_offset,
    us::DataLayouts.UniversalSize,
)
    (_, _, _, Nv, Nh) = DataLayouts.universal_size(us)
    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I
        (; q_bounds_nbr, ghost_buffer, rtol) = limiter
        q_bounds = Base.broadcastable(limiter.q_bounds)
        slab_q_bounds = slab(q_bounds, v, h)
        q_min = slab_q_bounds[slab_index(1)]
        q_max = slab_q_bounds[slab_index(2)]
        for lne in
            local_neighbor_elem_offset[h]:(local_neighbor_elem_offset[h + 1] - 1)
            h_nbr = local_neighbor_elem[lne]
            slab_q_bounds = slab(q_bounds, v, h_nbr)
            q_min = min(q_min, slab_q_bounds[slab_index(1)])
            q_max = max(q_max, slab_q_bounds[slab_index(2)])
        end
        slab_q_bounds_nbr = slab(q_bounds_nbr, v, h)
        slab_q_bounds_nbr[slab_index(1)] = q_min
        slab_q_bounds_nbr[slab_index(2)] = q_max
    end
    return nothing
end

function apply_limiter!(
    ρq::Fields.Field,
    ρ::Fields.Field,
    limiter::QuasiMonotoneLimiter,
    dev::ClimaComms.CUDADevice;
    warn::Bool = true,
)
    ρq_data = Fields.field_values(ρq)
    us = DataLayouts.UniversalSize(ρq_data)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    maxiter = Ni * Nj
    Nf = DataLayouts.ncomponents(ρq_data)
    WJ = Spaces.local_geometry_data(axes(ρq)).WJ
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (
        limiter,
        Fields.field_values(Operators.strip_space(ρq, axes(ρq))),
        Fields.field_values(Operators.strip_space(ρ, axes(ρ))),
        WJ,
        us,
        Val(Nf),
        Val(maxiter),
    )
    auto_launch!(
        apply_limiter_kernel!,
        args;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    call_post_op_callback() && post_op_callback(ρq, ρq, ρ, limiter, dev)
    return nothing
end

function apply_limiter_kernel!(
    limiter::QuasiMonotoneLimiter,
    ρq_data,
    ρ_data,
    WJ_data,
    us::DataLayouts.UniversalSize,
    ::Val{Nf},
    ::Val{maxiter},
) where {Nf, maxiter}
    (; q_bounds_nbr, rtol) = limiter
    converged = true
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
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

"""
    apply_limiter!(
        q::Fields.Field,
        ρ::Fields.Field,
        space,
        limiter::VerticalMassBorrowingLimiter,
        dev::ClimaComms.CUDADevice;
        warn::Bool = true,
    )

Apply the VerticalMassBorrowingLimiter to the field `q` with density field `ρ`.
"""
function apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    space,
    limiter::VerticalMassBorrowingLimiter,
    dev::ClimaComms.CUDADevice;
    warn::Bool = true,
)
    q_data = Fields.field_values(q)
    Nf = DataLayouts.ncomponents(q_data)
    us = DataLayouts.UniversalSize(q_data)
    q_min = limiter.q_min
    (; J) = Fields.local_geometry_field(ρ)
    # J is the local Jacobian magnitude (determinant), which already represents
    # the volume element per unit horizontal area for column fields.
    # For shallow atmospheres: J ≈ Δz (units: m)
    # For deep atmospheres: J accounts for spherical geometry (units: m)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    ncols = Ni * Nj * Nh
    nthread_x = Ni * Nj
    nthread_y = Nf
    nthread_z = cld(64, nthread_x * nthread_y) # ensure block is at least 64 threads
    # threads x dim represents nodes within an element
    # threads y represents different tracers
    # threads z may represent different horizontal elements if needed to reach 64 threads per block
    # blocks x dim represents different horizontal elements
    nthreads = (nthread_x, nthread_y, nthread_z)
    nblocks = cld(ncols * Nf, prod(nthreads))

    args = (
        typeof(limiter),
        Fields.field_values(q),
        Fields.field_values(ρ),
        Fields.field_values(J),
        q_min,
        us,
    )
    auto_launch!(
        apply_limiter_kernel!,
        args;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    call_post_op_callback() && post_op_callback(q, ρ, limiter, dev)
    return nothing
end

function apply_limiter_kernel!(
    ::Type{LM},
    q_data,
    ρ_data,
    ΔV_data,
    q_min_tuple,
    us::DataLayouts.UniversalSize) where {LM <: VerticalMassBorrowingLimiter}
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    j_idx, i_idx = divrem(CUDA.threadIdx().x - Int32(1), Ni)
    j_idx += Int32(1)
    i_idx += Int32(1)
    f_idx = CUDA.threadIdx().y
    # each z in a block is a different element
    h_idx = CUDA.blockDim().z * (CUDA.blockIdx().x - Int32(1)) + CUDA.threadIdx().z
    @inbounds if h_idx <= Nh
        q_min = q_min_tuple[f_idx]
        q_column_data = column(q_data, i_idx, j_idx, h_idx)
        ρ_column_data = column(ρ_data, i_idx, j_idx, h_idx)
        ΔV_column_data = column(ΔV_data, i_idx, j_idx, h_idx)
        column_massborrow!(
            (@view parent(q_column_data)[:, f_idx]),
            (@view parent(ρ_column_data)[:, 1]),
            (@view parent(ΔV_column_data)[:, 1]),
            q_min,
        )
    end
    return
end
