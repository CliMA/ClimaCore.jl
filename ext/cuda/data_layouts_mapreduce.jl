import ClimaCore.DataLayouts: AbstractDataSingleton
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
    data_out = DataLayouts.DataF{S}(Array(Array(f(pdata))[1, :]))
    call_post_op_callback() &&
        post_op_callback(data_out, f, op, data; weighted_jacobian, opargs...)
    return data_out
end

function mapreduce_cuda(
    f,
    op,
    data::DataLayouts.AbstractData;
    weighted_jacobian = OnesArray(parent(data)),
    opargs...,
)
    # This function implements the following parallel reduction algorithm:
    #
    # Each thread in each blocks processes multiple data points at the same time
    # (n_ops_on_load) each and we perform a block-wise reduction, with each
    # block writing to an array of (block-)shared memory. This array has the
    # same size as the block, ie, it is as long as many threads are available.
    # Processing multiple points means that we apply the reduction to the point
    # with index reduction[thread_index] = f(thread_index, thread_index +
    # OFFSET), with various OFFSETS that depend on `n_ops_on_load` and block
    # size.
    #
    # For the purpose of indexing, this is equivalent to having larger blocks
    # with size effective_blksize = blksize * (n_ops_on_load + 1).
    #
    #
    # After this operation, we have reduced all the data by a factor of
    # 1/n_ops_on_load and have results in various arrays `reduction` (one per
    # block)
    #
    # Once we have all the blocks reduced, we perform a tree reduction within
    # the block and "move" the reduced value to the first element of the array.
    # In this, one of the things to watch out for is that the last block might
    # not necessarily have all threads doing work, so we have to be careful to
    # not include data in `reduction` that did not have corresponding work.
    # Threads of index 1 will write that array into an output array.
    #
    # The output array has size nblocks, so we do another round of reduction,
    # but this time we put each Field in a different block.

    S = eltype(data)
    pdata = parent(data)
    T = eltype(pdata)
    (Ni, Nj, Nk, Nv, Nh) = size(data)
    Nf = DataLayouts.ncomponents(data) # length of field dimension
    pwt = parent(weighted_jacobian)

    nitems = Nv * Ni * Nj * Nk * Nh
    max_threads = 256# 512 1024
    nthreads = min(max_threads, nitems)
    # perform n ops during loading to shmem (this is a tunable parameter)
    n_ops_on_load = cld(nitems, nthreads) == 1 ? 0 : 7
    effective_blksize = nthreads * (n_ops_on_load + 1)
    nblocks = cld(nitems, effective_blksize)
    s = DataLayouts.singleton(data)
    us = DataLayouts.UniversalSize(data)

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
        s,
        us,
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
    data_out = DataLayouts.DataF{S}(Array(Array(reduce_cuda)[1, :]))

    call_post_op_callback() &&
        post_op_callback(data_out, f, op, data; weighted_jacobian, opargs...)
    return data_out
end

function mapreduce_cuda_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    f,
    op,
    pdata::AbstractArray{T, N},
    pwt::AbstractArray{T, N},
    n_ops_on_load::Int,
    s::AbstractDataSingleton,
    us::DataLayouts.UniversalSize,
    ::Val{shmemsize},
) where {T, N, shmemsize}
    blksize = blockDim().x
    nblk = gridDim().x
    tidx = threadIdx().x
    bidx = blockIdx().x
    fidx = blockIdx().y
    dataview = _dataview(pdata, s, fidx)
    effective_blksize = blksize * (n_ops_on_load + 1)
    gidx = _get_gidx(tidx, bidx, effective_blksize)
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)
    reduction[tidx] = 0
    (Ni, Nj, _, Nv, Nh) = DataLayouts.universal_size(us)
    Nf = 1 # a view into `fidx` always gives a size of Nf = 1
    nitems = Nv * Ni * Nj * Nf * Nh

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

    # The last block might not have enough threads to fill `reduction`, so some
    # of its elements might still have the value at initialization.
    blksize_for_reduction =
        min(blksize, nitems - effective_blksize * (bidx - 1))

    _cuda_intrablock_reduce!(op, reduction, tidx, blksize_for_reduction)

    tidx == 1 && (reduce_cuda[bidx, fidx] = reduction[1])
    return nothing
end

@inline function _get_gidx(tidx, bidx, effective_blksize)
    return tidx + (bidx - 1) * effective_blksize
end

@inline function _dataview(pdata::AbstractArray, s::AbstractDataSingleton, fidx)
    fdim = DataLayouts.field_dim(s)
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ndims(pdata) - fdim))
    return @inbounds view(pdata, Ipre..., fidx:fidx, Ipost...)
end

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
