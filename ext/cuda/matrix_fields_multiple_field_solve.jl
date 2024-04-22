import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: multiple_field_solve!
import ClimaCore.MatrixFields: is_CuArray_type
import ClimaCore.MatrixFields: allow_scalar_func

allow_scalar_func(::ClimaComms.CUDADevice, f, args) =
    CUDA.@allowscalar f(args...)

is_CuArray_type(::Type{T}) where {T <: CUDA.CuArray} = true

function multiple_field_solve!(::ClimaComms.CUDADevice, cache, x, A, b, x1)
    Ni, Nj, _, _, Nh = size(Fields.field_values(x1))
    names = MatrixFields.matrix_row_keys(keys(A))
    Nnames = length(names)
    nthreads, nblocks = _configure_threadblock(Ni * Nj * Nh * Nnames)
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

column_A(A::UniformScaling, i, j, h) = A
column_A(A, i, j, h) = Spaces.column(A, i, j, h)

function get_ijhn(Ni, Nj, Nh, Nnames, blockIdx, threadIdx, blockDim, gridDim)
    tidx = (blockIdx.x - 1) * blockDim.x + threadIdx.x
    (i, j, h, n) = if 1 ≤ tidx ≤ prod((Ni, Nj, Nh, Nnames))
        CartesianIndices((1:Ni, 1:Nj, 1:Nh, 1:Nnames))[tidx].I
    else
        (-1, -1, -1, -1)
    end
    return (i, j, h, n)
end

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
