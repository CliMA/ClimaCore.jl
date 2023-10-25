
function Base.sum(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.CUDADevice,
)
    context = ClimaComms.context(axes(field))
    localsum = mapreduce_cuda(identity, +, field, weighting = true)
    ClimaComms.allreduce!(context, parent(localsum), +)
    return localsum[]
end

function Base.sum(fn, field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localsum = mapreduce_cuda(fn, +, field, weighting = true)
    ClimaComms.allreduce!(context, parent(localsum), +)
    return localsum[]
end

function Base.maximum(fn, field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmax = mapreduce_cuda(fn, max, field)
    ClimaComms.allreduce!(context, parent(localmax), max)
    return localmax[]
end

function Base.maximum(field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmax = mapreduce_cuda(identity, max, field)
    ClimaComms.allreduce!(context, parent(localmax), max)
    return localmax[]
end

function Base.minimum(fn, field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmin = mapreduce_cuda(fn, min, field)
    ClimaComms.allreduce!(context, parent(localmin), min)
    return localmin[]
end

function Base.minimum(field::Field, ::ClimaComms.CUDADevice)
    context = ClimaComms.context(axes(field))
    localmin = mapreduce_cuda(identity, min, field)
    ClimaComms.allreduce!(context, parent(localmin), min)
    return localmin[]
end

Statistics.mean(
    field::Union{Field, Base.Broadcast.Broadcasted{<:FieldStyle}},
    ::ClimaComms.CUDADevice,
) = Base.sum(field) ./ Base.sum(ones(axes(field)))

Statistics.mean(fn, field::Field, ::ClimaComms.CUDADevice) =
    Base.sum(fn, field) ./ Base.sum(ones(axes(field)))

function LinearAlgebra.norm(
    field::Field,
    ::ClimaComms.CUDADevice,
    p::Real = 2;
    normalize = true,
)
    if p == 2
        # currently only one which supports structured types
        # TODO: perform map without allocation new field
        if normalize
            sqrt.(Statistics.mean(LinearAlgebra.norm_sqr.(field)))
        else
            sqrt.(sum(LinearAlgebra.norm_sqr.(field)))
        end
    elseif p == 1
        if normalize
            Statistics.mean(abs, field)
        else
            mapreduce_cuda(abs, +, field)
        end
    elseif p == Inf
        Base.maximum(abs, field)
    else
        if normalize
            Statistics.mean(x -> x^p, field) .^ (1 / p)
        else
            mapreduce_cuda(x -> x^p, +, field) .^ (1 / p)
        end
    end
end

function mapreduce_cuda(
    f,
    op,
    field::Field{V};
    weighting = false,
    opargs...,
) where {
    S,
    V <: Union{DataLayouts.VF{S}, DataLayouts.IJFH{S}, DataLayouts.VIJFH{S}},
}
    data = Fields.field_values(field)
    pdata = parent(data)
    T = eltype(pdata)
    (Ni, Nj, Nk, Nv, Nh) = size(data)
    Nf = div(length(pdata), prod(size(data))) # length of field dimension
    wt = Spaces.weighted_jacobian(axes(field))
    pwt = parent(wt)

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
    return DataLayouts.DataF{S}(Array(Array(reduce_cuda)[1, :]))
end

function mapreduce_cuda_kernel!(
    reduce_cuda::AbstractArray{T, 2},
    f,
    op,
    pdata::AbstractArray{T, N},
    pwt::AbstractArray{T, N},
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
    gidx = _get_gidx(tidx, bidx, fidx, effective_blksize, nblk)
    reduction = CUDA.CuStaticSharedArray(T, shmemsize)
    reduction[tidx] = 0
    (Nv, Nij, Nf, Nh) = _get_dims(pdata)
    nitems = Nv * Nij * Nij * Nf * Nh

    # load shmem
    if gidx ≤ nitems
        if weighting
            reduction[tidx] = f(pdata[gidx]) * pwt[gidx]
            for n_ops in 1:n_ops_on_load
                gidx2 = _get_gidx(
                    tidx + blksize * n_ops,
                    bidx,
                    fidx,
                    effective_blksize,
                    nblk,
                )
                if gidx2 ≤ nitems
                    reduction[tidx] =
                        op(reduction[tidx], f(pdata[gidx2]) * pwt[gidx2])
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
# for VF DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 2}) where {FT}
    (Nv, Nf) = size(pdata)
    return (Nv, 1, Nf, 1)
end

# for IJFH DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 4}) where {FT}
    (Nij, _, Nf, Nh) = size(pdata)
    return (1, Nij, Nf, Nh)
end

# for VIJFH DataLayout
@inline function _get_dims(pdata::AbstractArray{FT, 5}) where {FT}
    (Nv, Nij, _, Nf, Nh) = size(pdata)
    return (Nv, Nij, Nf, Nh)
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
