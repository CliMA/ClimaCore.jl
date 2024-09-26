const CI5 = CartesianIndex{5}

maximum_allowable_threads() = (
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),
)

"""
    partition(::AbstractData, n_max_threads)

Given `n_max_threads`, which should be determined
from CUDA's occupancy API, `partition` returns
a NamedTuple containing:

 - `threads` size of threads to pass to CUDA
 - `blocks` size of blocks to pass to CUDA

The general pattern followed here, which seems
to produce good results, is to satisfy a few
criteria:

 - Maximize the number of (allowable) threads
   in the thread partition
 - The order of the thread partition should
   follow the fastest changing index in the
   datalayout (e.g., VIJ in VIJFH)
"""
function partition end

"""
    universal_index(::AbstractData)

Returns a universal cartesian index,
computed from CUDA's `threadIdx`,
`blockIdx` and `blockDim`.
"""
function universal_index end

"""
    is_valid_index(::AbstractData, I::CartesianIndex, us::UniversalSize)

Check the minimal number of index
bounds to ensure that the result of
`universal_index` is valid.
"""
function is_valid_index end

##### VIJFH
@inline function partition(data::DataLayouts.VIJFH, n_max_threads::Integer)
    (Nij, _, _, Nv, Nh) = DataLayouts.universal_size(data)
    Nv_thread = min(Int(fld(n_max_threads, Nij * Nij)), Nv)
    Nv_blocks = cld(Nv, Nv_thread)
    @assert prod((Nv_thread, Nij, Nij)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nv_thread, Nij, Nij))),$n_max_threads)"
    return (; threads = (Nv_thread, Nij, Nij), blocks = (Nv_blocks, Nh))
end
@inline function universal_index(::DataLayouts.VIJFH)
    (tv, i, j) = CUDA.threadIdx()
    (bv, h) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    return CartesianIndex((i, j, 1, v, h))
end
@inline is_valid_index(::DataLayouts.VIJFH, I::CI5, us::UniversalSize) =
    1 ≤ I[4] ≤ DataLayouts.get_Nv(us)

##### IJFH
@inline function partition(data::DataLayouts.IJFH, n_max_threads::Integer)
    (Nij, _, _, _, Nh) = DataLayouts.universal_size(data)
    Nh_thread = min(
        Int(fld(n_max_threads, Nij * Nij)),
        Nh,
        maximum_allowable_threads()[3],
    )
    Nh_blocks = cld(Nh, Nh_thread)
    @assert prod((Nij, Nij)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nij, Nij))),$n_max_threads)"
    return (; threads = (Nij, Nij, Nh_thread), blocks = (Nh_blocks,))
end
@inline function universal_index(::DataLayouts.IJFH)
    (i, j, th) = CUDA.threadIdx()
    (bh,) = CUDA.blockIdx()
    h = th + (bh - 1) * CUDA.blockDim().z
    return CartesianIndex((i, j, 1, 1, h))
end
@inline is_valid_index(::DataLayouts.IJFH, I::CI5, us::UniversalSize) =
    1 ≤ I[5] ≤ DataLayouts.get_Nh(us)

##### IFH
@inline function partition(data::DataLayouts.IFH, n_max_threads::Integer)
    (Ni, _, _, _, Nh) = DataLayouts.universal_size(data)
    Nh_thread = min(Int(fld(n_max_threads, Ni)), Nh)
    Nh_blocks = cld(Nh, Nh_thread)
    @assert prod((Ni, Nh_thread)) ≤ n_max_threads "threads,n_max_threads=($(prod((Ni, Nh_thread))),$n_max_threads)"
    return (; threads = (Ni, Nh_thread), blocks = (Nh_blocks,))
end
@inline function universal_index(::DataLayouts.IFH)
    (i, th) = CUDA.threadIdx()
    (bh,) = CUDA.blockIdx()
    h = th + (bh - 1) * CUDA.blockDim().y
    return CartesianIndex((i, 1, 1, 1, h))
end
@inline is_valid_index(::DataLayouts.IFH, I::CI5, us::UniversalSize) =
    1 ≤ I[5] ≤ DataLayouts.get_Nh(us)

##### IJF
@inline function partition(data::DataLayouts.IJF, n_max_threads::Integer)
    (Nij, _, _, _, _) = DataLayouts.universal_size(data)
    @assert prod((Nij, Nij)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nij, Nij))),$n_max_threads)"
    return (; threads = (Nij, Nij), blocks = (1,))
end
@inline function universal_index(::DataLayouts.IJF)
    (i, j) = CUDA.threadIdx()
    return CartesianIndex((i, j, 1, 1, 1))
end
@inline is_valid_index(::DataLayouts.IJF, I::CI5, us::UniversalSize) = true

##### IF
@inline function partition(data::DataLayouts.IF, n_max_threads::Integer)
    (Ni, _, _, _, _) = DataLayouts.universal_size(data)
    @assert Ni ≤ n_max_threads "threads,n_max_threads=($(Ni),$n_max_threads)"
    return (; threads = (Ni,), blocks = (1,))
end
@inline function universal_index(::DataLayouts.IF)
    (i,) = CUDA.threadIdx()
    return CartesianIndex((i, 1, 1, 1, 1))
end
@inline is_valid_index(::DataLayouts.IF, I::CI5, us::UniversalSize) = true

##### VIFH
@inline function partition(data::DataLayouts.VIFH, n_max_threads::Integer)
    (Ni, _, _, Nv, Nh) = DataLayouts.universal_size(data)
    Nv_thread = min(Int(fld(n_max_threads, Ni)), Nv)
    Nv_blocks = cld(Nv, Nv_thread)
    @assert prod((Nv_thread, Ni)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nv_thread, Ni))),$n_max_threads)"
    return (; threads = (Nv_thread, Ni), blocks = (Nv_blocks, Nh))
end
@inline function universal_index(::DataLayouts.VIFH)
    (tv, i) = CUDA.threadIdx()
    (bv, h) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    return CartesianIndex((i, 1, 1, v, h))
end
@inline is_valid_index(::DataLayouts.VIFH, I::CI5, us::UniversalSize) =
    1 ≤ I[4] ≤ DataLayouts.get_Nv(us)

##### VF
@inline function partition(data::DataLayouts.VF, n_max_threads::Integer)
    (_, _, _, Nv, _) = DataLayouts.universal_size(data)
    Nvt = fld(n_max_threads, Nv)
    Nv_thread = Nvt == 0 ? n_max_threads : min(Int(Nvt), Nv)
    Nv_blocks = cld(Nv, Nv_thread)
    @assert Nv_thread ≤ n_max_threads "threads,n_max_threads=($(Nv_thread),$n_max_threads)"
    (; threads = (Nv_thread,), blocks = (Nv_blocks,))
end
@inline function universal_index(::DataLayouts.VF)
    (tv,) = CUDA.threadIdx()
    (bv,) = CUDA.blockIdx()
    v = tv + (bv - 1) * CUDA.blockDim().x
    return CartesianIndex((1, 1, 1, v, 1))
end
@inline is_valid_index(::DataLayouts.VF, I::CI5, us::UniversalSize) =
    1 ≤ I[4] ≤ DataLayouts.get_Nv(us)

##### DataF
@inline partition(data::DataLayouts.DataF, n_max_threads::Integer) =
    (; threads = 1, blocks = 1)
@inline universal_index(::DataLayouts.DataF) = CartesianIndex((1, 1, 1, 1, 1))
@inline is_valid_index(::DataLayouts.DataF, I::CI5, us::UniversalSize) = true

#####
##### Custom partitions
#####

##### Column-wise
@inline function columnwise_partition(
    us::DataLayouts.UniversalSize,
    n_max_threads::Integer,
)
    (Nij, _, _, _, Nh) = DataLayouts.universal_size(us)
    Nh_thread = min(
        Int(fld(n_max_threads, Nij * Nij)),
        maximum_allowable_threads()[3],
        Nh,
    )
    Nh_blocks = cld(Nh, Nh_thread)
    @assert prod((Nij, Nij, Nh_thread)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nij, Nij, Nh_thread))),$n_max_threads)"
    return (; threads = (Nij, Nij, Nh_thread), blocks = (Nh_blocks,))
end
@inline function columnwise_universal_index(us::UniversalSize)
    (i, j, th) = CUDA.threadIdx()
    (bh,) = CUDA.blockIdx()
    h = th + (bh - 1) * CUDA.blockDim().z
    return CartesianIndex((i, j, 1, 1, h))
end
@inline columnwise_is_valid_index(I::CI5, us::UniversalSize) =
    1 ≤ I[5] ≤ DataLayouts.get_Nh(us)

##### Element-wise (e.g., limiters)
# TODO

##### Multiple-field solve partition
@inline function multiple_field_solve_partition(
    us::DataLayouts.UniversalSize,
    n_max_threads::Integer;
    Nnames,
)
    (Nij, _, _, _, Nh) = DataLayouts.universal_size(us)
    @assert prod((Nij, Nij, Nnames)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nij, Nij, Nnames))),$n_max_threads)"
    return (; threads = (Nij, Nij, Nnames), blocks = (Nh,))
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
    (Nq, _, _, Nv, Nh) = DataLayouts.universal_size(us)
    Nvthreads = min(fld(n_max_threads, Nq * Nq), maximum_allowable_threads()[3])
    Nvblocks = cld(Nv, Nvthreads)
    @assert prod((Nq, Nq, Nvthreads)) ≤ n_max_threads "threads,n_max_threads=($(prod((Nq, Nq, Nvthreads))),$n_max_threads)"
    @assert Nq * Nq ≤ n_max_threads
    return (; threads = (Nq, Nq, Nvthreads), blocks = (Nh, Nvblocks), Nvthreads)
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
