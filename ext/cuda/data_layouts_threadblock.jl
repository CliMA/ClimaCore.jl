const CI5 = CartesianIndex{5}
# using ClimaCartesianIndices: FastCartesianIndices
FastCartesianIndices(x) = CartesianIndices(x)

maximum_allowable_threads() = (
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),
)

"""
    universal_index(::AbstractData)

Returns a universal cartesian index,
computed from CUDA's `threadIdx`,
`blockIdx` and `blockDim`.
"""
function universal_index end

##### Masked
@inline cartesian_indicies_mask(us::DataLayouts.UniversalSize, mask::IJHMask) =
    cartesian_indicies_mask(us, typeof(mask), mask.N[1])

@inline function cartesian_indicies_mask(
    us::DataLayouts.UniversalSize,
    ::Type{<:IJHMask},
    n_active_columns::Integer,
)
    (Ni, _, _, Nv, Nh) = DataLayouts.universal_size(us)
    return FastCartesianIndices((1:Nv, 1:n_active_columns))
end
@inline masked_partition(mask::IJHMask, n_max_threads, us) =
    masked_partition(typeof(mask), mask.N[1], n_max_threads, us)

@inline function masked_partition(
    ::Type{<:IJHMask},
    n_active_columns,
    n_max_threads,
    us,
)
    (_, _, _, Nv, _) = DataLayouts.universal_size(us)
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
    return CartesianIndex((i, j, 1, v, h))
end

#####
##### Custom partitions
#####

##### linear partition
@inline function linear_partition(nitems::Integer, n_max_threads::Integer)
    threads = min(nitems, n_max_threads)
    blocks = cld(nitems, threads)
    return (; threads, blocks)
end
@inline function cartesian_indices(us::UniversalSize)
    inds = DataLayouts.universal_size(us)
    return FastCartesianIndices(map(Base.OneTo, inds))
end
@inline linear_is_valid_index(i::Integer, us::UniversalSize) =
    1 ≤ i ≤ DataLayouts.get_N(us)

##### Column-wise
@inline function cartesian_indices_columnwise(us::DataLayouts.UniversalSize)
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    inds = (Ni, Nj, Nh)
    return FastCartesianIndices(map(Base.OneTo, inds))
end

##### Element-wise (e.g., limiters)
# TODO

##### Multiple-field solve partition
@inline function cartesian_indices_multiple_field_solve(
    us::DataLayouts.UniversalSize;
    Nnames,
)
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    inds = (Ni, Nj, Nh, Nnames)
    return FastCartesianIndices(map(Base.OneTo, inds))
end
@inline function multiple_field_solve_universal_index(us::UniversalSize)
    (i, j, iname) = CUDA.threadIdx()
    (h,) = CUDA.blockIdx()
    return (CartesianIndex((i, j, 1, 1, h)), iname)
end
@inline multiple_field_solve_is_valid_index(I::CI5, us::UniversalSize) =
    1 ≤ I[5] ≤ DataLayouts.get_Nh(us)

##### spectral kernel partition
@inline function spectral_partition(
    us::DataLayouts.UniversalSize,
    n_max_threads::Integer = 256;
)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    Nvthreads = min(fld(n_max_threads, Ni * Nj), maximum_allowable_threads()[3])
    Nvblocks = cld(Nv, Nvthreads)
    @assert prod((Ni, Nj, Nvthreads)) ≤ n_max_threads "threads,n_max_threads=($(prod((Ni, Nj, Nvthreads))),$n_max_threads)"
    @assert Ni * Nj ≤ n_max_threads
    return (; threads = (Ni, Nj, Nvthreads), blocks = (Nh, Nvblocks), Nvthreads)
end
@inline function spectral_universal_index(space::Spaces.AbstractSpace)
    i = threadIdx().x
    j = threadIdx().y
    k = threadIdx().z
    h = blockIdx().x
    vid = k + (blockIdx().y - 1) * blockDim().z
    if space isa Spaces.AbstractSpectralElementSpace
        v = nothing
    elseif space isa Spaces.FaceExtrudedFiniteDifferenceSpace
        v = vid - half
    elseif space isa Spaces.CenterExtrudedFiniteDifferenceSpace
        v = vid
    else
        error("Invalid space")
    end
    ij = CartesianIndex((i, j))
    slabidx = Fields.SlabIndex(v, h)
    return (ij, slabidx)
end
@inline spectral_is_valid_index(
    space::Spaces.AbstractSpectralElementSpace,
    ij,
    slabidx,
) = Operators.is_valid_index(space, ij, slabidx)

##### shmem fd kernel partition
@inline function fd_shmem_stencil_partition(
    us::DataLayouts.UniversalSize,
    n_face_levels::Integer,
    n_max_threads::Integer = 256;
)
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    Nvthreads = n_face_levels
    @assert Nvthreads <= maximum_allowable_threads()[1] "Number of vertical face levels cannot exceed $(maximum_allowable_threads()[1])"
    Nvblocks = cld(Nv, Nvthreads) # +1 may be needed to guarantee that shared memory is populated at the last cell face
    return (;
        threads = (Nvthreads,),
        blocks = (Nh, Nvblocks, Ni * Nj),
        Nvthreads,
    )
end
@inline function fd_shmem_stencil_universal_index(
    space::Spaces.AbstractSpace,
    us,
)
    (tv,) = CUDA.threadIdx()
    (h, bv, ij) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    (Ni, Nj, _, _, _) = DataLayouts.universal_size(us)
    if Ni * Nj < ij
        return CartesianIndex((-1, -1, 1, -1, -1))
    end
    @inbounds (i, j) = CartesianIndices((Ni, Nj))[ij].I
    return CartesianIndex((i, j, 1, v, h))
end
@inline fd_shmem_stencil_is_valid_index(I::CI5, us::UniversalSize) =
    1 ≤ I[5] ≤ DataLayouts.get_Nh(us)
