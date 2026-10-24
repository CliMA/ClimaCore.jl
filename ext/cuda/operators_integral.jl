import ClimaCore: DataLayouts, Spaces, Fields, level, column
import ClimaCore.Grids: toggle_placeholder_grid
import ClimaCore.Operators:
    column_reduce_device!,
    single_column_reduce!,
    column_accumulate_device!,
    single_column_accumulate!
import ClimaComms

# The output of `column_reduce!` on a `FiniteDifferenceSpace` is a 0-dimensional
# `DataF`, so use `size(data, d)` (which is 1 for `d > ndims(data)`) instead of
# destructuring `size(data)` or calling `cartesian_indices_columnwise(data)`.
@inline function columnwise_cartesian_indices(data)
    (Ni, Nj, Nh) = (size(data, 2), size(data, 3), size(data, 4))
    return CartesianIndices(map(Base.OneTo, (Ni, Nj, Nh)))
end

function column_reduce_device!(
    dev::ClimaComms.CUDADevice,
    f::F,
    transform::T,
    output,
    input,
    init,
    space,
    reverse,
) where {F, T}
    out_fv = Fields.field_values(output)
    mask = Spaces.get_mask(space)
    if !(mask isa DataLayouts.NoMask) && space isa Spaces.FiniteDifferenceSpace
        error("Masks not supported for FiniteDifferenceSpace")
    end
    cart_inds = columnwise_cartesian_indices(out_fv)
    args = (
        single_column_reduce!,
        f,
        transform,
        toggle_placeholder_grid(output, space),
        toggle_placeholder_grid(input, space),
        toggle_placeholder_grid(init, space),
        space,
        mask,
        cart_inds,
        reverse,
    )
    auto_launch!(bycolumn_kernel!, args; nitems = length(cart_inds))
    call_post_op_callback() && post_op_callback(
        output,
        (dev, f, transform, output, input, init, space, reverse),
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
    reverse,
) where {F, T}
    out_fv = Fields.field_values(output)
    mask = Spaces.get_mask(space)
    if !(mask isa DataLayouts.NoMask) && space isa Spaces.FiniteDifferenceSpace
        error("Masks not supported for FiniteDifferenceSpace")
    end
    cart_inds = columnwise_cartesian_indices(out_fv)
    args = (
        single_column_accumulate!,
        f,
        transform,
        toggle_placeholder_grid(output, space),
        toggle_placeholder_grid(input, space),
        toggle_placeholder_grid(init, space),
        space,
        mask,
        cart_inds,
        reverse,
    )
    auto_launch!(bycolumn_kernel!, args; nitems = length(cart_inds))
end

function bycolumn_kernel!(
    single_column_function!::S,
    f::F,
    transform::T,
    output′,
    input′,
    init′,
    space,
    mask,
    cart_inds,
    reverse,
) where {S, F, T}
    output = toggle_placeholder_grid(output′, space)
    input = toggle_placeholder_grid(input′, space)
    init = toggle_placeholder_grid(init′, space)
    if space isa Spaces.FiniteDifferenceSpace
        single_column_function!(f, transform, output, input, init, space, reverse)
    else
        tidx = thread_index()
        if linear_is_valid_index(tidx, unval(cart_inds))
            I = unval(cart_inds)[tidx]
            (i, j, h) = I.I
            ui = CartesianIndex(1, i, j, h)
            DataLayouts.should_compute(mask, ui) || return nothing
            single_column_function!(
                f,
                transform,
                column(output, i, j, h),
                column(input, i, j, h),
                Operators.column_init(init, i, j, h),
                column(space, i, j, h),
                reverse,
            )
        end
    end
end
