import ClimaCore.Limiters:
    QuasiMonotoneLimiter,
    compute_element_bounds!,
    compute_neighbor_bounds_local!,
    apply_limiter!,
    apply_limit_slab!,
    VerticalMassBorrowingLimiter,
    column_massborrow!
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
    dev::ClimaComms.CUDADevice,
)
    ρ_values = Base.broadcastable(Fields.field_values(ρ))
    ρq_values = Base.broadcastable(Fields.field_values(ρq))
    (Nv, _, _, Nh) = size(ρ_values)
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
    (Nv, Ni, Nj, Nh) = size(ρ)
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
    return nothing
end


function compute_neighbor_bounds_local!(
    limiter::QuasiMonotoneLimiter,
    ρ,
    dev::ClimaComms.CUDADevice,
)
    topology = Spaces.topology(axes(ρ))
    (Nv, _, _, Nh) = size(Fields.field_values(ρ))
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (
        limiter,
        topology.local_neighbor_elem,
        topology.local_neighbor_elem_offset,
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
)
    (; q_bounds_nbr, ghost_buffer, rtol) = limiter
    (Nv, _, _, Nh) = size(q_bounds_nbr)
    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I
        q_bounds = Base.broadcastable(limiter.q_bounds)
        slab_q_bounds = slab(q_bounds, v, h)
        q_min = slab_q_bounds[1]
        q_max = slab_q_bounds[2]
        for lne in
            local_neighbor_elem_offset[h]:(local_neighbor_elem_offset[h + 1] - 1)
            h_nbr = local_neighbor_elem[lne]
            slab_q_bounds = slab(q_bounds, v, h_nbr)
            q_min = min(q_min, slab_q_bounds[1])
            q_max = max(q_max, slab_q_bounds[2])
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
    dev::ClimaComms.CUDADevice;
    warn::Bool = true,
)
    ρq_data = Fields.field_values(ρq)
    (Nv, _, _, Nh) = size(ρq_data)
    WJ = Spaces.local_geometry_data(axes(ρq)).WJ
    nthreads, nblocks = config_threadblock(Nv, Nh)
    args = (limiter, ρq_data, Fields.field_values(ρ), WJ)
    auto_launch!(
        apply_limiter_kernel!,
        args;
        threads_s = nthreads,
        blocks_s = nblocks,
    )
    call_post_op_callback() && post_op_callback(ρq, ρq, ρ, limiter, dev)
    return nothing
end

function apply_limiter_kernel!(limiter::QuasiMonotoneLimiter, ρq_data, ρ_data, WJ_data)
    (; q_bounds_nbr, rtol) = limiter
    (Nv, _, _, Nh) = size(ρq_data)
    n = (Nv, Nh)
    tidx = thread_index()
    @inbounds if valid_range(tidx, prod(n))
        (v, h) = kernel_indexes(tidx, n).I
        # Convergence statistics are discarded on GPUs (no warning on failure).
        apply_limit_slab!(
            slab(ρq_data, v, h),
            slab(ρ_data, v, h),
            slab(WJ_data, v, h),
            slab(q_bounds_nbr, v, h),
            rtol,
        )
    end
    return nothing
end

"""
    apply_limiter!(
        q::Fields.Field,
        ρ::Fields.Field,
        space,
        limiter::VerticalMassBorrowingLimiter,
        dev::ClimaComms.CUDADevice,
    )

Apply the VerticalMassBorrowingLimiter to the field `q` with density field `ρ`.
"""
function apply_limiter!(
    q::Fields.Field,
    ρ::Fields.Field,
    space,
    limiter::VerticalMassBorrowingLimiter,
    dev::ClimaComms.CUDADevice,
)
    q_data = Fields.field_values(q)
    Nf = DataLayouts.ncomponents(q_data)
    q_min = limiter.q_min
    (; J) = Fields.local_geometry_field(ρ)
    # J is the local Jacobian magnitude (determinant), which already represents
    # the volume element per unit horizontal area for column fields.
    # For shallow atmospheres: J ≈ Δz (units: m)
    # For deep atmospheres: J accounts for spherical geometry (units: m)
    (_, Ni, Nj, Nh) = size(q_data)
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
) where {LM <: VerticalMassBorrowingLimiter}
    (_, Ni, _, Nh) = size(q_data)
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
        # Use full-rank indices into the 5-D column parents; views at partial
        # indices reshape their parents, which cannot be compiled in kernels.
        column_massborrow!(
            (@view parent(q_column_data)[:, 1, 1, f_idx, 1]),
            (@view parent(ρ_column_data)[:, 1, 1, 1, 1]),
            (@view parent(ΔV_column_data)[:, 1, 1, 1, 1]),
            q_min,
        )
    end
    return
end
