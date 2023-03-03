
Base.sum(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.CUDA,
) = mapreduce_cuda(identity, +, field, weighting = true) #TODO: distributed support to be added

Base.sum(fn, field::Field, ::ClimaComms.CUDA) =
    mapreduce_cuda(fn, +, field, weighting = true) #TODO: distributed support to be added

Base.maximum(fn, field::Field, ::ClimaComms.CUDA) =
    mapreduce_cuda(fn, max, field) #TODO: distributed support to be added

Base.maximum(field::Field, ::ClimaComms.CUDA) =
    mapreduce_cuda(identity, max, field) #TODO: distributed support to be added

Base.minimum(fn, field::Field, ::ClimaComms.CUDA) =
    mapreduce_cuda(fn, min, field) #TODO: distributed support to be added

Base.minimum(field::Field, ::ClimaComms.CUDA) =
    mapreduce_cuda(identity, min, field) #TODO: distributed support to be added

Statistics.mean(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.CUDA,
) = Base.sum(field) ./ Spaces.local_area(axes(field)) #TODO: distributed support to be added

Statistics.mean(fn, field::Field, ::ClimaComms.CUDA) =
    Base.sum(fn, field) ./ Spaces.local_area(axes(field)) #TODO: distributed support to be added

function mapreduce_cuda(
    f,
    op,
    field::Field{V};
    weighting = false,
    opargs...,
) where {
    Nij,
    A <: AbstractArray,
    V <:
    Union{DataLayouts.IJFH{<:Any, Nij, A}, DataLayouts.VIJFH{<:Any, Nij, A}},
}
    data = Fields.field_values(field)
    pdata = parent(data)
    T = eltype(pdata)
    (_, _, _, Nv, Nh) = size(data)
    Nf = div(length(pdata), prod(size(data))) # length of field dimension
    wt = Spaces.weighted_jacobian(axes(field)) # wt always IJFH layout
    pwt = parent(wt)

    nitems = Nv * Nij * Nij * Nh
    max_threads = 256# 512 1024
    nthreads = min(max_threads, nitems)
    # perform n ops during loading to shmem (this is a tunable parameter)
    n_ops_on_load = cld(nitems, nthreads) == 1 ? 0 : 7
    effective_blksize = nthreads * (n_ops_on_load + 1)
    nblocks = cld(nitems, effective_blksize)

    reduce_cuda = CuArray{T}(undef, nblocks, Nf)
    shmemsize = nthreads
    # place each field on a different block
    @cuda threads = (nthreads) blocks = (nblocks, Nf) mapreduce_cuda_kernel!(
        reduce_cuda,
        f,
        op,
        pdata,
        pwt,
        weighting,
        n_ops_on_load,
        Val(shmemsize),
    )
    # reduce block data
    if nblocks > 1
        nthreads = min(32, nblocks)
        shmemsize = nthreads
        @cuda threads = (nthreads) blocks = (Nf) reduce_cuda_blocks_kernel!(
            reduce_cuda,
            op,
            Val(shmemsize),
        )
    end
    return Array(Array(reduce_cuda)[1, :])
end

function mapreduce_cuda_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    f,
    op,
    pdata::AbstractArray{T, N},
    pwt::AbstractArray{T, 4},
    weighting::Bool,
    n_ops_on_load::Int,
    ::Val{shmemsize},
) where {T, N, shmemsize}
    blksize = blockDim().x
    nblk = gridDim().x
    tidx = threadIdx().x
    bidx = blockIdx().x
    fidx = blockIdx().y
    effective_blksize = blksize * (n_ops_on_load + 1)
    #gidx = tidx + (bidx - 1) * effective_blksize + (fidx - 1) * effective_blksize * nblk
    gidx = _get_gidx(tidx, bidx, fidx, effective_blksize, nblk)
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)
    reduction[tidx] = 0
    (Nv, Nij, Nf, Nh) = _get_dims(pdata)
    nitems = Nv * Nij * Nij * Nf * Nh
    iidx, jidx, hidx = _get_idxs(Nv, Nij, Nf, Nh, fidx, gidx)

    # load shmem
    if gidx ≤ nitems
        if weighting
            reduction[tidx] = f(pdata[gidx]) * pwt[iidx, jidx, 1, hidx]
            for n_ops in 1:n_ops_on_load
                gidx2 = _get_gidx(
                    tidx + blksize * n_ops,
                    bidx,
                    fidx,
                    effective_blksize,
                    nblk,
                )
                if gidx2 ≤ nitems
                    iidx2, jidx2, hidx2 =
                        _get_idxs(Nv, Nij, Nf, Nh, fidx, gidx2)
                    reduction[tidx] = op(
                        reduction[tidx],
                        f(pdata[gidx2]) * pwt[iidx2, jidx2, 1, hidx2],
                    )
                end
            end
        else
            reduction[tidx] = f(pdata[gidx])
            for n_ops in 1:n_ops_on_load
                gidx2 = _get_gidx(
                    tidx + blksize * n_ops,
                    bidx,
                    fidx,
                    effective_blksize,
                    nblk,
                )
                if gidx2 ≤ nitems
                    iidx2, jidx2, hidx2 =
                        _get_idxs(Nv, Nij, Nf, Nh, fidx, gidx2)
                    reduction[tidx] = op(reduction[tidx], f(pdata[gidx2]))
                end
            end
        end
    end
    sync_threads()
    _cuda_intrablock_reduce!(op, reduction, tidx, blksize)

    tidx == 1 && (reduce_cuda[bidx, fidx] = reduction[1])
    return nothing
end

@inline function _get_gidx(tidx, bidx, fidx, effective_blksize, nblk)
    return tidx +
           (bidx - 1) * effective_blksize +
           (fidx - 1) * effective_blksize * nblk
end

@inline function _get_dims(pdata::AbstractArray{FT, 4}) where {FT}
    (Nij, _, Nf, Nh) = size(pdata)
    return (1, Nij, Nf, Nh)
end

@inline function _get_dims(pdata::AbstractArray{FT, 5}) where {FT}
    (Nv, Nij, _, Nf, Nh) = size(pdata)
    return (Nv, Nij, Nf, Nh)
end

@inline function _get_idxs(Nv, Nij, Nf, Nh, fidx, gidx)
    hidx = cld(gidx, Nv * Nij * Nij * Nf)
    offset = ((hidx - 1) * Nf + (fidx - 1)) * Nv * Nij * Nij
    jidx = cld(gidx - offset, Nv * Nij)
    offset += (jidx - 1) * Nv * Nij
    iidx = cld(gidx - offset, Nv)
    return (iidx, jidx, hidx)
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
