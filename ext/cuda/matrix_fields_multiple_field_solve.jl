import CUDA
import ClimaComms
import LinearAlgebra: UniformScaling
import ClimaCore.Operators
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: _single_field_solve!
import ClimaCore.MatrixFields: multiple_field_solve!
import ClimaCore.MatrixFields: is_CuArray_type
import ClimaCore.Utilities.UnrolledFunctions: unrolled_map

is_CuArray_type(::Type{T}) where {T <: CUDA.CuArray} = true

NVTX.@annotate function multiple_field_solve!(
    ::ClimaComms.CUDADevice,
    cache,
    x,
    A,
    b,
    x1,
)
    names = MatrixFields.matrix_row_keys(keys(A))
    Nnames = length(names)
    Ni, Nj, _, _, Nh = size(Fields.field_values(x1))
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

    us = UniversalSize(Fields.field_values(x1))
    args = (device, caches, xs, As, bs, x1, us, Val(Nnames))

    nitems = Ni * Nj * Nh * Nnames
    threads = threads_via_occupancy(multiple_field_solve_kernel!, args)
    n_max_threads = min(threads, nitems)
    # p = multiple_field_solve_partition(us, n_max_threads; Nnames)
    p = linear_partition(nitems, n_max_threads)

    auto_launch!(
        multiple_field_solve_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
        always_inline = true,
    )
end

Base.@propagate_inbounds column_A(A::UniformScaling, i, j, h) = A
Base.@propagate_inbounds column_A(A, i, j, h) = Spaces.column(A, i, j, h)

@generated function generated_single_field_solve!(
    device,
    caches,
    xs,
    As,
    bs,
    i,
    j,
    h,
    iname,
    ::Val{Nnames},
) where {Nnames}
    return quote
        Base.Cartesian.@nif $Nnames ξ -> (iname == ξ) ξ -> begin
            _single_field_solve!(
                device,
                column_A(caches[ξ], i, j, h),
                column_A(xs[ξ], i, j, h),
                column_A(As[ξ], i, j, h),
                column_A(bs[ξ], i, j, h),
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
    us::UniversalSize,
    ::Val{Nnames},
) where {Nnames}
    @inbounds begin
        (Ni, Nj, _, _, Nh) = size(Fields.field_values(x1))
        tidx = thread_index()
        n = (Ni, Nj, Nh, Nnames)
        if valid_range(tidx, prod(n))
            (i, j, h, iname) = kernel_indexes(tidx, n).I
            generated_single_field_solve!(
                device,
                caches,
                xs,
                As,
                bs,
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
