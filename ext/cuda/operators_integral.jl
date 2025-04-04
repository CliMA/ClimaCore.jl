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
    dev::ClimaComms.CUDADevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
) where {F, T}
    Ni, Nj, _, _, Nh = size(Fields.field_values(output))
    us = UniversalSize(Fields.field_values(output))
    mask = Spaces.get_mask(space)
    if !(mask isa DataLayouts.NoMask) && space isa Spaces.FiniteDifferenceSpace
        error("Masks not supported for FiniteDifferenceSpace")
    end
    args = (
        single_column_reduce!,
        f,
        transform,
        strip_space(output, axes(output)), # The output space is irrelevant here
        strip_space(input, space),
        init,
        space,
        us,
        mask,
    )
    nitems = Ni * Nj * Nh
    threads = threads_via_occupancy(bycolumn_kernel!, args)
    n_max_threads = min(threads, nitems)
    p = columnwise_partition(us, n_max_threads)
    auto_launch!(
        bycolumn_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
    call_post_op_callback() && post_op_callback(
        output,
        (dev, f, transform, output, input, init, space),
        (;),
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
    mask = Spaces.get_mask(space)
    if !(mask isa DataLayouts.NoMask) && space isa Spaces.FiniteDifferenceSpace
        error("Masks not supported for FiniteDifferenceSpace")
    end
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
        mask,
    )
    (Ni, Nj, _, _, Nh) = DataLayouts.universal_size(us)
    nitems = Ni * Nj * Nh
    threads = threads_via_occupancy(bycolumn_kernel!, args)
    n_max_threads = min(threads, nitems)
    p = columnwise_partition(us, n_max_threads)
    auto_launch!(
        bycolumn_kernel!,
        args;
        threads_s = p.threads,
        blocks_s = p.blocks,
    )
end

function bycolumn_kernel!(
    single_column_function!::S,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
    us::DataLayouts.UniversalSize,
    mask,
) where {S, F, T}
    if space isa Spaces.FiniteDifferenceSpace
        single_column_function!(f, transform, output, input, init, space)
    else
        I = columnwise_universal_index(us)
        DataLayouts.should_compute(mask, I) || return nothing
        if columnwise_is_valid_index(I, us)
            (i, j, _, _, h) = I.I
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
end
