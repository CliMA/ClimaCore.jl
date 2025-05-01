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
    cart_inds = cartesian_indices_columnwise(us)
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
        cart_inds,
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
    cart_inds = cartesian_indices_columnwise(us)
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
        cart_inds,
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
    cart_inds,
) where {S, F, T}
    if space isa Spaces.FiniteDifferenceSpace
        single_column_function!(f, transform, output, input, init, space)
    else
        tidx = linear_thread_idx()
        if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
            I = unval(cart_inds)[tidx]
            (i, j, h) = I.I
            ui = CartesianIndex((i, j, 1, 1, h))
            DataLayouts.should_compute(mask, ui) || return nothing
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
