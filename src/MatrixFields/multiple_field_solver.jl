# TODO: Can different A's be different matrix styles?
#       if so, how can we handle fuse/parallelize?

# First, dispatch based on the first x and the device:
function multiple_field_solve!(cache, x, A, b)
    name1 = first(matrix_row_keys(keys(A)))
    x1 = x[name1]
    multiple_field_solve!(ClimaComms.device(axes(x1)), cache, x, A, b, x1)
end

# TODO: fuse/parallelize
function multiple_field_solve!(
    ::ClimaComms.AbstractCPUDevice,
    cache,
    x,
    A,
    b,
    x1,
)
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end
end

function multiple_field_solve!(::ClimaComms.CUDADevice, cache, x, A, b, x1)
    Ni, Nj, _, _, Nh = size(Fields.field_values(x1))
    names = matrix_row_keys(keys(A))
    Nnames = length(names)
    nthreads, nblocks = Topologies._configure_threadblock(Ni * Nj * Nh * Nnames)
    sscache = Operators.strip_space(cache)
    ssx = Operators.strip_space(x)
    ssA = Operators.strip_space(A)
    ssb = Operators.strip_space(b)
    cache_tup = map(name -> sscache[name], names)
    x_tup = map(name -> ssx[name], names)
    A_tup = map(name -> ssA[name, name], names)
    b_tup = map(name -> ssb[name], names)
    x1 = first(x_tup)

    tups = (cache_tup, x_tup, A_tup, b_tup)

    device = ClimaComms.device(x[first(names)])
    CUDA.@cuda threads = nthreads blocks = nblocks multiple_field_solve_kernel!(
        device,
        tups,
        x1,
        Val(Nnames),
    )
end

function get_ijhn(Ni, Nj, Nh, Nnames, blockIdx, threadIdx, blockDim, gridDim)
    tidx = (blockIdx.x - 1) * blockDim.x + threadIdx.x
    (i, j, h, n) = if 1 ≤ tidx ≤ prod((Ni, Nj, Nh, Nnames))
        CartesianIndices((1:Ni, 1:Nj, 1:Nh, 1:Nnames))[tidx].I
    else
        (-1, -1, -1, -1)
    end
    return (i, j, h, n)
end

column_A(A::UniformScaling, i, j, h) = A
column_A(A, i, j, h) = Spaces.column(A, i, j, h)

@inline function _recurse(js::Tuple, tups::Tuple, transform, device, i::Int)
    if first(js) == i
        tup_args = map(x -> transform(first(x)), tups)
        _single_field_solve!(tup_args..., device)
    end
    _recurse(Base.tail(js), map(x -> Base.tail(x), tups), transform, device, i)
end

@inline _recurse(js::Tuple{}, tups::Tuple, transform, device, i::Int) = nothing

@inline function _recurse(
    js::Tuple{Int},
    tups::Tuple,
    transform,
    device,
    i::Int,
)
    if first(js) == i
        tup_args = map(x -> transform(first(x)), tups)
        _single_field_solve!(tup_args..., device)
    end
    return nothing
end

function multiple_field_solve_kernel!(
    device::ClimaComms.CUDADevice,
    tups,
    x1,
    ::Val{Nnames},
) where {Nnames}
    @inbounds begin
        Ni, Nj, _, _, Nh = size(Fields.field_values(x1))
        (i, j, h, iname) = get_ijhn(
            Ni,
            Nj,
            Nh,
            Nnames,
            CUDA.blockIdx(),
            CUDA.threadIdx(),
            CUDA.blockDim(),
            CUDA.gridDim(),
        )
        if 1 ≤ i <= Ni && 1 ≤ j ≤ Nj && 1 ≤ h ≤ Nh && 1 ≤ iname ≤ Nnames

            nt = ntuple(ξ -> ξ, Val(Nnames))
            _recurse(nt, tups, ξ -> column_A(ξ, i, j, h), device, iname)
            # _recurse effectively calls
            #    _single_field_solve!(
            #        Spaces.column(caches[iname], i, j, h),
            #        Spaces.column(xs[iname], i, j, h),
            #        column_A(As[iname], i, j, h),
            #        Spaces.column(bs[iname], i, j, h),
            #        device,
            #    )
        end
    end
    return nothing
end
