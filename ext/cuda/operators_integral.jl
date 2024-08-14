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
    threads_s, blocks_s = _configure_threadblock(Ni * Nj * Nh)
    args = (
        single_column_reduce!,
        f,
        transform,
        strip_space(output, axes(output)), # The output space is irrelevant here
        strip_space(input, space),
        init,
        space,
    )
    auto_launch!(bycolumn_kernel!, args; threads_s, blocks_s)
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
    Ni, Nj, _, _, Nh = size(Fields.field_values(output))
    threads_s, blocks_s = _configure_threadblock(Ni * Nj * Nh)
    args = (
        single_column_accumulate!,
        f,
        transform,
        strip_space(output, space),
        strip_space(input, space),
        init,
        space,
    )
    auto_launch!(bycolumn_kernel!, args; threads_s, blocks_s)
end

bycolumn_kernel!(
    single_column_function!::S,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
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
