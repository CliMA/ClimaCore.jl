# To implement a single flexible mapreduce, let's define
# a `OnesArray` that has nothing, and always returns 1:
struct OnesArray{T, N} <: AbstractArray{T, N} end
OnesArray(x::AbstractArray) = OnesArray{eltype(x), ndims(x)}()
Base.@propagate_inbounds Base.getindex(::OnesArray, inds...) = 1
Base.parent(x::OnesArray) = x

function mapreduce_cuda(
    f,
    op,
    data::DataLayouts.DataF;
    weighted_jacobian = OnesArray(parent(data)),
    opargs...,
)
    pdata = parent(data)
    S = eltype(data)
    return DataLayouts.DataF{S}(Array(Array(f(pdata))[1, :]))
end

function mapreduce_cuda(
    f,
    op,
    data::Union{DataLayouts.VF, DataLayouts.IJFH, DataLayouts.VIJFH};
    weighted_jacobian = OnesArray(parent(data)),
    opargs...,
)
    S = eltype(data)
    pdata = parent(data)
    T = eltype(pdata)
    (Ni, Nj, Nk, Nv, Nh) = size(data)
    Nf = div(length(pdata), prod(size(data))) # length of field dimension
    pwt = parent(weighted_jacobian)

    nitems = Nv * Ni * Nj * Nk * Nh
    max_threads = 256# 512 1024
    nthreads = min(max_threads, nitems)
    # perform n ops during loading to shmem (this is a tunable parameter)
    n_ops_on_load = cld(nitems, nthreads) == 1 ? 0 : 7
    effective_blksize = nthreads * (n_ops_on_load + 1)
    nblocks = cld(nitems, effective_blksize)

    reduce_cuda = CuArray{T}(undef, nblocks, Nf)
    shmemsize = nthreads
    # place each field on a different block
    @cuda always_inline = true threads = (nthreads) blocks = (nblocks, Nf) mapreduce_cuda_kernel!(
        reduce_cuda,
        f,
        op,
        pdata,
        pwt,
        n_ops_on_load,
        Val(shmemsize),
    )
    # reduce block data
    if nblocks > 1
        nthreads = min(32, nblocks)
        shmemsize = nthreads
        @cuda always_inline = true threads = (nthreads) blocks = (Nf) reduce_cuda_blocks_kernel!(
            reduce_cuda,
            op,
            Val(shmemsize),
        )
    end
    return DataLayouts.DataF{S}(Array(Array(reduce_cuda)[1, :]))
end

function mapreduce_cuda_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    f,
    op,
    pdata::AbstractArray{T, N},
    pwt::AbstractArray{T, N},
    n_ops_on_load::Int,
    ::Val{shmemsize},
) where {T, N, shmemsize}
    blksize = blockDim().x
    nblk = gridDim().x
    tidx = threadIdx().x
    bidx = blockIdx().x
    fidx = blockIdx().y
    dataview = _dataview(pdata, fidx)
    effective_blksize = blksize * (n_ops_on_load + 1)
    gidx = _get_gidx(tidx, bidx, effective_blksize)
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)
    reduction[tidx] = 0
    (Nv, Nij, Nf, Nh) = _get_dims(dataview)
    nitems = Nv * Nij * Nij * Nf * Nh

    # load shmem
    if gidx ≤ nitems
        reduction[tidx] = f(dataview[gidx]) * pwt[gidx]
        for n_ops in 1:n_ops_on_load
            gidx2 = _get_gidx(tidx + blksize * n_ops, bidx, effective_blksize)
            if gidx2 ≤ nitems
                reduction[tidx] =
                    op(reduction[tidx], f(dataview[gidx2]) * pwt[gidx2])
            end
        end
    end
    sync_threads()
    _cuda_intrablock_reduce!(op, reduction, tidx, blksize)

    tidx == 1 && (reduce_cuda[bidx, fidx] = reduction[1])
    return nothing
end

@inline function _get_gidx(tidx, bidx, effective_blksize)
    return tidx + (bidx - 1) * effective_blksize
end
# for VF DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 2}) where {FT}
    (Nv, Nf) = size(pdata)
    return (Nv, 1, Nf, 1)
end
@inline _dataview(pdata::AbstractArray{FT, 2}, fidx) where {FT} =
    view(pdata, :, fidx:fidx)

# for IJFH DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 4}) where {FT}
    (Nij, _, Nf, Nh) = size(pdata)
    return (1, Nij, Nf, Nh)
end
@inline _dataview(pdata::AbstractArray{FT, 4}, fidx) where {FT} =
    view(pdata, :, :, fidx:fidx, :)

# for VIJFH DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 5}) where {FT}
    (Nv, Nij, _, Nf, Nh) = size(pdata)
    return (Nv, Nij, Nf, Nh)
end
@inline _dataview(pdata::AbstractArray{FT, 5}, fidx) where {FT} =
    view(pdata, :, :, :, fidx:fidx, :)

@inline function _cuda_reduce!(op, reduction, tidx, reduction_size, N)
    if reduction_size > N
        if tidx ≤ reduction_size - N
            @inbounds reduction[tidx] = op(reduction[tidx], reduction[tidx + N])
        end
        N > 32 && sync_threads()
        return N
    end
    return reduction_size
end

function reduce_cuda_blocks_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    op,
    ::Val{shmemsize},
) where {T, shmemsize}
    blksize = blockDim().x
    fidx = blockIdx().x
    tidx = threadIdx().x
    nitems = size(reduce_cuda, 1)
    nloads = cld(nitems, blksize) - 1
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)

    reduction[tidx] = reduce_cuda[tidx, fidx]

    for i in 1:nloads
        idx = tidx + blksize * i
        if idx ≤ nitems
            reduction[tidx] = op(reduction[tidx], reduce_cuda[idx, fidx])
        end
    end

    blksize > 32 && sync_threads()
    _cuda_intrablock_reduce!(op, reduction, tidx, blksize)

    tidx == 1 && (reduce_cuda[1, fidx] = reduction[1])
    return nothing
end

@inline function _cuda_intrablock_reduce!(op, reduction, tidx, blksize)
    # assumes max_threads ≤ 1024 which is the current max on any CUDA device
    newsize = _cuda_reduce!(op, reduction, tidx, blksize, 512)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 256)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 128)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 64)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 32)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 16)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 8)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 4)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 2)
    newsize = _cuda_reduce!(op, reduction, tidx, newsize, 1)
    return nothing
end
