import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: multiple_field_solve!
import ClimaCore.MatrixFields: is_CuArray_type
import ClimaCore.MatrixFields: allow_scalar_func
import ClimaCore.Utilities.UnrolledFunctions: unrolled_map

allow_scalar_func(::ClimaComms.CUDADevice, f, args) =
    CUDA.@allowscalar f(args...)

is_CuArray_type(::Type{T}) where {T <: CUDA.CuArray} = true

function multiple_field_solve!(::ClimaComms.CUDADevice, cache, x, A, b, x1)
    names = MatrixFields.matrix_row_keys(keys(A))
    Nnames = length(names)
    sscache = Operators.strip_space(cache)
    ssx = Operators.strip_space(x)
    ssA = Operators.strip_space(A)
    ssb = Operators.strip_space(b)
    caches = map(name -> sscache[name], names)
    xs = map(name -> ssx[name], names)
    As = map(name -> ssA[name, name], names)
    bs = map(name -> ssb[name], names)
    x1 = first(xs)

    device = ClimaComms.device(x[first(names)])

    args = (device, caches, xs, As, bs, x1, Val(Nnames))

    auto_launch!(
        multiple_field_solve_kernel!,
        args,
        x1;
        auto = true,
        always_inline = true,
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

@generated function generated_single_field_solve!(
    caches,
    xs,
    As,
    bs,
    device,
    i,
    j,
    h,
    iname,
    ::Val{Nnames},
) where {Nnames}
    return quote
        Base.Cartesian.@nif $Nnames ξ -> (iname == ξ) ξ -> begin
            _single_field_solve!(
                column_A(caches[ξ], i, j, h),
                column_A(xs[ξ], i, j, h),
                column_A(As[ξ], i, j, h),
                column_A(bs[ξ], i, j, h),
                device,
            )
        end
    end
end

function multiple_field_solve_kernel!(
    device::ClimaComms.CUDADevice,
    caches,
    xs,
    As,
    bs,
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
            generated_single_field_solve!(
                caches,
                xs,
                As,
                bs,
                device,
                i,
                j,
                h,
                iname,
                Val(Nnames),
            )
        end
    end
    return nothing
end
