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
    # ssx1 = first(map(name -> ssx[name], names))
    # @show typeof(Spaces.column(ssx1, 1, 1, 1))
    # @show Spaces.column(ssx1, 1, 1, 1) isa Fields.ColumnField
    CUDA.@cuda always_inline = true threads = nthreads blocks = nblocks multiple_field_solve_kernel!(
        map(name -> sscache[name], names),
        map(name -> ssx[name], names),
        map(name -> ssA[name, name], names),
        map(name -> ssb[name], names),
        Val(Nnames),
    )
end

function get_ijhn(Ni, Nj, Nh, Nnames, blockIdx, threadIdx, blockDim, gridDim)
    tidx = (blockIdx.x - 1) * blockDim.x + threadIdx.x
    (i, j, h, n) = CartesianIndices((1:Ni, 1:Nj, 1:Nh, 1:Nnames))[tidx].I
    return (i, j, h, n)
end

function multiple_field_solve_kernel!(
    caches,
    xs,
    As,
    bs,
    ::Val{Nnames},
) where {Nnames}
    @inbounds begin
        x1 = first(xs)
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
            cache = caches[iname]
            x = xs[iname]
            A = As[iname]
            b = bs[iname]
            _single_field_solve!(
                Spaces.column(cache, i, j, h),
                Spaces.column(x, i, j, h),
                Spaces.column(A, i, j, h),
                Spaces.column(b, i, j, h),
            )
        end
    end
    return nothing
end
