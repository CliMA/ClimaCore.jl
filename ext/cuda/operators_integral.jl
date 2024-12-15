import ClimaCore: Spaces, Fields, level, column
import ClimaCore.Operators:
    left_idx,
    strip_space,
    column_reduce_device!,
    single_column_reduce!,
    column_accumulate_device!,
    single_column_accumulate!
import ClimaComms
using CUDA: @cuda

function column_reduce_device!(
    ::ClimaComms.CUDADevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
) where {F, T}
    Ni, Nj, _, _, Nh = size(Fields.field_values(output))
    us = UniversalSize(Fields.field_values(output))
    args = (
        single_column_reduce!,
        f,
        transform,
        strip_space(output, axes(output)), # The output space is irrelevant here
        strip_space(input, space),
        init,
        space,
        us,
    )
    nitems = Ni * Nj * Nh
    threads = threads_via_occupancy(bycolumn_kernel!, args)
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        bycolumn_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
end

function column_accumulate_device!(
    ::ClimaComms.CUDADevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
) where {F, T}
    out_fv = Fields.field_values(output)
    us = UniversalSize(out_fv)
    args = (
        single_column_accumulate!,
        f,
        transform,
        strip_space(output, space),
        strip_space(input, space),
        init,
        space,
        us,
    )
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * Nh
    threads = threads_via_occupancy(bycolumn_kernel!, args)
    n_max_threads = min(threads, nitems)
    p = linear_partition(nitems, n_max_threads)
    auto_launch!(
        bycolumn_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
end

bycolumn_kernel!(
    single_column_function!::S,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
    us::DataLayouts.UniversalSize,
) where {S, F, T} =
    if space isa Spaces.FiniteDifferenceSpace
        single_column_function!(f, transform, output, input, init, space)
    else
        idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        Ni, Nj, _, _, Nh = size(Fields.field_values(output))
        if idx <= Ni * Nj * Nh
            i, j, h = cart_ind((Ni, Nj, Nh), idx).I
            single_column_function!(
                f,
                transform,
                column(output, i, j, h),
                column(input, i, j, h),
                init,
                column(space, i, j, h),
            )
        end
    end
