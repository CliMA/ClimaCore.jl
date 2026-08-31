# Wrappers for sizes and indices that are passed to kernels as type parameters.
@inline unval(x) = x
@inline unval(::Val{x}) where {x} = x

"""
    linear_thread_idx()

Returns the linear index of the current thread across all blocks, computed from
CUDA's `threadIdx`, `blockIdx` and `blockDim`.
"""
@inline linear_thread_idx() =
    (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x

##### Masked
# mask.N holds the active column count in a one-element device array; reading it
# to build a launch configuration on the host requires an explicit scalar access.
@inline cartesian_indices_mask(data, mask::IJHMask) =
    cartesian_indices_mask(data, typeof(mask), CUDA.@allowscalar(mask.N[1]))

@inline function cartesian_indices_mask(
    data,
    ::Type{<:IJHMask},
    n_active_columns::Integer,
)
    Nv = size(data, 1)
    return CartesianIndices((1:Nv, 1:n_active_columns))
end
@inline masked_partition(mask::IJHMask, n_max_threads, data) =
    masked_partition(typeof(mask), CUDA.@allowscalar(mask.N[1]), n_max_threads, data)

@inline function masked_partition(
    ::Type{<:IJHMask},
    n_active_columns,
    n_max_threads,
    data,
)
    Nv = size(data, 1)
    nitems = n_active_columns * Nv
    return linear_partition(nitems, n_max_threads)
end

@inline function masked_universal_index(mask::IJHMask, cart_inds)
    tidx = linear_thread_idx()
    (v, ijh) = unval(cart_inds)[tidx].I
    (; i_map, j_map, h_map) = mask
    @inbounds i = i_map[ijh]
    @inbounds j = j_map[ijh]
    @inbounds h = h_map[ijh]
    return CartesianIndex((v, i, j, h))
end

#####
##### Custom partitions
#####

##### linear partition
@inline function linear_partition(nitems::Integer, n_max_threads::Integer)
    @assert nitems > 0
    threads = min(nitems, n_max_threads)
    blocks = cld(nitems, threads)
    return (; threads, blocks)
end
@inline cartesian_indices(data) =
    CartesianIndices(map(Base.OneTo, size(data)))
@inline linear_is_valid_index(i::Integer, data) = 1 ≤ i ≤ length(data)

##### Column-wise
@inline function cartesian_indices_columnwise(data)
    (_, Ni, Nj, Nh) = size(data)
    return CartesianIndices(map(Base.OneTo, (Ni, Nj, Nh)))
end

##### Element-wise (e.g., limiters)
# TODO

##### Multiple-field solve partition
@inline function cartesian_indices_multiple_field_solve(data; Nnames)
    (_, Ni, Nj, Nh) = size(data)
    return CartesianIndices(map(Base.OneTo, (Ni, Nj, Nh, Nnames)))
end

##### shmem fd kernel partition
@inline function fd_shmem_stencil_partition(
    data,
    n_face_levels::Integer,
    n_max_threads::Integer = 256;
)
    (Nv, Ni, Nj, Nh) = size(data)
    Nvthreads = n_face_levels
    Nvlimit = device_attributes().max_block_dim_x
    @assert Nvthreads <= Nvlimit "Number of vertical face levels cannot exceed $Nvlimit"
    Nvblocks = cld(Nv, Nvthreads) # +1 may be needed to guarantee that shared memory is populated at the last cell face
    return (;
        threads = (Nvthreads,),
        blocks = (Nh, Nvblocks, Ni * Nj),
        Nvthreads,
    )
end
@inline function fd_shmem_stencil_universal_index(space::Spaces.AbstractSpace, data)
    (tv,) = CUDA.threadIdx()
    (h, bv, ij) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    (_, Ni, Nj, _) = size(data)
    if Ni * Nj < ij
        return CartesianIndex((-1, -1, -1, -1))
    end
    @inbounds (i, j) = CartesianIndices((Ni, Nj))[ij].I
    return CartesianIndex((v, i, j, h))
end
@inline fd_shmem_stencil_is_valid_index(I, data) = 1 ≤ I[4] ≤ size(data, 4)
