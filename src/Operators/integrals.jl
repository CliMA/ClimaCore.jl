import RootSolvers
import ClimaComms

broadcast_zero(field) = zero(eltype(Base.broadcastable(field)))

"""
    column_integral_definite!(ϕ_top, ᶜ∂ϕ∂z, [ϕ_bot])

Set `````ϕ_top```{}= \\frac{1}{ΔA(z_{bot})}\\int_{z_{bot}}^{z_{top}}\\, ```ᶜ∂ϕ∂z```(z)\\,ΔA(z)\\,dz +{}```ϕ_bot`````, where ``z_{bot}`` and ``z_{top}`` are
the values of `z` at the bottom and top of the domain, and where `ΔA` is the
area differential `J/Δz`, with `J` denoting the metric Jacobian. The input
`ᶜ∂ϕ∂z` must be a cell-center `Field` or a lazy broadcast of `Field`s, and the
output `ϕ_top` must be a horizontal `Field`. The default value of `ϕ_bot` is 0.
"""
function column_integral_definite!(ϕ_top, ᶜ∂ϕ∂z, ϕ_bot = broadcast_zero(ϕ_top))
    ᶜJ = Fields.local_geometry_field(axes(ᶜ∂ϕ∂z)).J
    f_space = Spaces.face_space(axes(ᶜ∂ϕ∂z))
    J_bot = Fields.level(Fields.local_geometry_field(f_space).J, half)
    Δz_bot = Fields.level(Fields.Δz_field(f_space), half)
    ΔA_bot = Base.broadcasted(/, J_bot, Δz_bot)
    ᶜΔϕ = Base.broadcasted(*, ᶜ∂ϕ∂z, Base.broadcasted(/, ᶜJ, ΔA_bot))
    column_reduce!(+, ϕ_top, ᶜΔϕ; init = ϕ_bot)
end

"""
    column_integral_indefinite!(ᶠϕ, ᶜ∂ϕ∂z, [ϕ_bot])

Set `````ᶠϕ```(z) = \\frac{1}{ΔA(z_{bot})}\\int_{z_{bot}}^z\\,```ᶜ∂ϕ∂z```(z')\\, ΔA(z')\\,dz' +{}```ϕ_bot`````, where ``z_{bot}`` is the value of `z` at the bottom
of the domain, and where `ΔA` is the area differential `J/Δz`, with `J` denoting
the metric Jacobian. The input `ᶜ∂ϕ∂z` must be a cell-center `Field` or a lazy
broadcast of `Field`s, and the output `ᶠϕ` must be a cell-face `Field`. The default value
of `ϕ_bot` is 0.

    column_integral_indefinite!(∂ϕ∂z, ᶠϕ, [ϕ_bot], [rtol])

Set `````ᶠϕ```(z) = \\frac{1}{ΔA(z_{bot})}\\int_{z_{bot}}^z\\, ```∂ϕ∂z```(```ᶠϕ```(z'), z')\\,ΔA(z')\\,dz' +{}```ϕ_bot`````, where `∂ϕ∂z` can be
any scalar-valued two-argument function. When a shallow atmosphere approximation
is used, `ΔA = ΔA_{bot}` at all values of `z`, and the output `ᶠϕ` satisfies
`ᶜgradᵥ.(ᶠϕ) ≈ ∂ϕ∂z.(ᶜint.(ᶠϕ), ᶜz)` with a relative tolerance of `rtol`, where
`ᶜgradᵥ = GradientF2C()` and `ᶜint = InterpolateF2C()`. When a deep atmosphere
is used, the vertical gradient is replaced with an area-weighted gradient. The
default value of `ϕ_bot` is 0, and the default value of `rtol` is 0.001.
"""
function column_integral_indefinite!(ᶠϕ, ᶜ∂ϕ∂z, ϕ_bot = broadcast_zero(ᶠϕ))
    ᶜJ = Fields.local_geometry_field(axes(ᶜ∂ϕ∂z)).J
    J_bot = Fields.level(Fields.local_geometry_field(ᶠϕ).J, half)
    Δz_bot = Fields.level(Fields.Δz_field(ᶠϕ), half)
    ΔA_bot = Base.broadcasted(/, J_bot, Δz_bot)
    ᶜΔϕ = Base.broadcasted(*, ᶜ∂ϕ∂z, Base.broadcasted(/, ᶜJ, ΔA_bot))
    column_accumulate!(+, ᶠϕ, ᶜΔϕ; init = ϕ_bot)
end
function column_integral_indefinite!(
    ∂ϕ∂z::F,
    ᶠϕ,
    ϕ_bot = broadcast_zero(ᶠϕ),
    rtol = eltype(parent(ᶠϕ))(0.001),
) where {F <: Function}
    device = ClimaComms.device(ᶠϕ)
    c_space = Spaces.center_space(axes(ᶠϕ))
    ᶜz = Fields.coordinate_field(c_space).z
    ᶜJ = Fields.local_geometry_field(c_space).J
    J_bot = Fields.level(Fields.local_geometry_field(ᶠϕ).J, half)
    Δz_bot = Fields.level(Fields.Δz_field(ᶠϕ), half)
    ΔA_bot = Base.broadcasted(/, J_bot, Δz_bot)
    ᶜz_and_Δz = Base.broadcasted(tuple, ᶜz, Base.broadcasted(/, ᶜJ, ΔA_bot))
    column_accumulate!(ᶠϕ, ᶜz_and_Δz; init = ϕ_bot) do ϕ_prev, (z, weighted_Δz)
        residual(ϕ_new) =
            (ϕ_new - ϕ_prev) / weighted_Δz - ∂ϕ∂z((ϕ_prev + ϕ_new) / 2, z)
        (; converged, root) = RootSolvers.find_zero(
            residual,
            RootSolvers.NewtonsMethodAD(ϕ_prev),
            RootSolvers.CompactSolution(),
            RootSolvers.RelativeSolutionTolerance(rtol),
        )
        ClimaComms.@assert device converged "unable to integrate through \
                                             z = $z with rtol set to $rtol"
        return root
    end
end

################################################################################

# A column field (e.g., Spaces.column(field, i, j, h)) keeps the space of its own column in
# every column of a loop over other fields, so an operator expression with one is
# instantiated, and each of its broadcasts is sliced with the space it has outside of the
# loop (see Fields.shared_space). Stored spaces make other inputs slower in every column.
loop_input(input) =
    input isa StencilBroadcasted && !(axes(input) isa Spaces.FiniteDifferenceSpace) &&
    has_column_fields(input) ? Broadcast.instantiate(input) : input

has_column_fields(arg) = false
has_column_fields(field::Fields.Field) = axes(field) isa Spaces.FiniteDifferenceSpace
has_column_fields(bc::Broadcasted) = unrolled_any(has_column_fields, bc.args)
@drop_recursion_limits has_column_fields

# Operators that are not associative are applied in order by one thread of each column (see
# DataLayouts.scan_points), so a GPU kernel gives that thread a column of its own. So are the
# operators of reductions that are not commutative (see DataLayouts.fold_points).
serial_column_loop(f) = Val(!DataLayouts.is_associative(f))
serial_column_loop(f, input) = Val(!DataLayouts.is_commutative(f, eltype(input)))

# The function of a loop that folds or scans every input column with op. Its buffers are
# known from the types of the columns, so a GPU loop is only compiled at the level of shared
# memory that holds them (see DataLayouts.foreach_slice): those of the input's operators,
# and those in which the threads of a column combine values. Constant propagation from the
# slices of a loop would infer f a second time, and constant keyword arguments (e.g., the
# default reverse) would infer the loops of column_reduce! and column_accumulate! a second
# time, so none of these are constant-propagated.
struct ColumnScan{O, F}
    op::O
    f::F
end
@inline_types_only (scan::ColumnScan)(output_column, input_column) =
    @inline scan.f(scan.op, output_column, input_column)
Adapt.@adapt_structure ColumnScan
DataLayouts.required_shared_memory(scan::ColumnScan, output_column, input_column) = max(
    largest_application_bytes(input_column),
    combined_values_bytes(scan.op, output_column, input_column),
)
DataLayouts.uses_shared_memory(scan::ColumnScan, output_column, input_column) =
    uses_shared_buffers(input_column) ||
    combined_values_bytes(scan.op, output_column, input_column) > 0
combined_values_bytes(op::O, output_column, input_column) where {O} =
    DataLayouts.combined_values_bytes(
        DataLayouts.DataScope(output_column),
        Utilities.return_type(op, Tuple{eltype(input_column), eltype(input_column)}),
    )

# A thread with private buffers applies the operators of a column where its values are
# folded or scanned, instead of in a separately compiled function that returns them. A scan
# inlines every read of the values, so it can read them from mutable registers without
# copying them into immutable ones (see constant_field).
@inline column_operators(arg) = apply_operators(arg)
@inline column_operators(bc::StencilBroadcasted) =
    has_private_buffers(bc) ? constant_field(scanned_column(bc)) : apply_operators(bc)

@inline scanned_column(arg) = apply_operators(arg)
@inline function scanned_column(bc::StencilBroadcasted)
    has_private_buffers(bc) || return apply_operators(bc)
    stencil_bc = pointwise_stencil(bc, caching_fusion(bc))
    dest = register_similar(destination_template(stencil_bc), eltype(stencil_bc))
    return apply_stencil!(dest, stencil_bc, Val(true))
end

"""
    column_reduce!(f, output, input; [init], [transform], [reverse])

Apply `reduce` to `input` along the vertical direction, storing the result in
`output`. The `input` can be either a `Field`, or a lazy broadcast that involves
pointwise or columnwise operations on `Field`s. Each reduced value is computed
by iteratively applying `f` to the values in `input`, starting from the bottom
of each column and moving upward, and the result of the final iteration is
passed to the `transform` function before being stored in `output`. If `init` is
specified, it is used as the initial value of the iteration; otherwise, the
value at the starting boundary of each column in `input` is used as the initial
value. By default, reduction starts at the bottom boundary and proceeds upward.
When `reverse = true`, it starts at the top boundary and proceeds downward. If `f`
is an associative operator like `+`, `*`, or `max`, all threads that process a
column share the work of reducing it, and if `f` is also commutative, CPUs reduce
the values in any order, either of which can change the result by roundoff; `init`
is then combined with the reduced value, in the type of the values.

With `first_level` and `last_level` denoting the indices of the boundary levels
of `input`, the default reduction in each column can be summarized as follows:

  - If `init` is unspecified,
    ```
    reduced_value = input[first_level]
    for level in (first_level + 1):last_level
        reduced_value = f(reduced_value, input[level])
    end
    output[] = transform(reduced_value)
    ```
  - If `init` is specified,
    ```
    reduced_value = init
    for level in first_level:last_level
        reduced_value = f(reduced_value, input[level])
    end
    output[] = transform(reduced_value)
    ```
"""
@inline_types_only function column_reduce!(
    f::F,
    output::Fields.Field,
    input::Union{Fields.Field, Fields.PointwiseBroadcasted, StencilBroadcasted};
    init = DataLayouts.NoInit(),
    transform::T = identity,
    reverse::Bool = false,
) where {F, T}
    # The output point is set by the first thread of each column (see owns_point), from which
    # the other threads copy a single point in registers (see share_single_point!). The
    # function f is passed by the loop's ColumnScan, since a closure that also captured it
    # would take up twice its bytes in the parameters of a GPU kernel.
    function _column_reduce!(f, output_point, input_column)
        values = Broadcast.instantiate(Fields.field_values(column_operators(input_column)))
        result = DataLayouts.fold_points(f, values, reverse, init)
        DataLayouts.owns_point(DataLayouts.DataScope(output_point), 1) &&
            (@inbounds output_point[] = transform(result))
        return nothing
    end
    input′ = loop_input(input)
    mask = Spaces.get_mask(axes(input))
    DataLayouts._foreach_slice(
        column,
        ColumnScan(f, _column_reduce!),
        mask,
        Val(false),
        serial_column_loop(f, input),
        output,
        input′,
    )
    DataLayouts.share_single_point!(Fields.field_values(output))
end

"""
    column_accumulate!(f, output, input; [init], [transform], [reverse])

Apply `accumulate` to `input` along the vertical direction, storing the result
in `output`. The `input` can be either a `Field`, or a lazy broadcast that involves
pointwise or columnwise operations on `Field`s. By default, each accumulated
value is computed by iteratively applying `f` to the values in `input`, starting
from the bottom of each column and moving upward, and the result of each
iteration is passed to the `transform` function before being stored in `output`.
The `init` value is optional for center-to-center, face-to-face, and
face-to-center accumulation, but it is required for center-to-face accumulation.
When `reverse = true`, accumulation starts at the top boundary and proceeds
downward, with the corresponding staggered boundary offsets reversed. As in
[`column_reduce!`](@ref), the threads that process a column share the work of
accumulating it when `f` is an associative operator.

With `first_level` and `last_level` denoting the indices of the boundary levels
of `input`, the default accumulation in each column can be summarized as follows:

  - For center-to-center and face-to-face accumulation with `init` unspecified,
    ```
    accumulated_value = input[first_level]
    output[first_level] = transform(accumulated_value)
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level] = transform(accumulated_value)
    end
    ```
  - For center-to-center and face-to-face accumulation with `init` specified,
    ```
    accumulated_value = init
    for level in first_level:last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level] = transform(accumulated_value)
    end
    ```
  - For face-to-center accumulation with `init` unspecified,
    ```
    accumulated_value = input[first_level]
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level - half] = transform(accumulated_value)
    end
    ```
  - For face-to-center accumulation with `init` specified,
    ```
    accumulated_value = f(init, input[first_level])
    for level in (first_level + 1):last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level - half] = transform(accumulated_value)
    end
    ```
  - For center-to-face accumulation,
    ```
    accumulated_value = init
    output[first_level - half] = transform(accumulated_value)
    for level in first_level:last_level
        accumulated_value = f(accumulated_value, input[level])
        output[level + half] = transform(accumulated_value)
    end
    ```
"""
@inline_types_only function column_accumulate!(
    f::F,
    output::Fields.Field,
    input::Union{Fields.Field, Fields.PointwiseBroadcasted, StencilBroadcasted};
    init = DataLayouts.NoInit(),
    transform::T = identity,
    reverse::Bool = false,
) where {F, T}
    is_c2f =
        Spaces.staggering(axes(input)) == Spaces.CellCenter() &&
        Spaces.staggering(axes(output)) == Spaces.CellFace()
    init == DataLayouts.NoInit() && is_c2f &&
        error("init must be specified for center-to-face accumulation")
    function _column_accumulate!(f, output_column, input_column)
        scope = DataLayouts.DataScope(output_column, input_column)
        init_index = reverse ? length(output_column) : 1
        init isa DataLayouts.NoInit || !is_c2f ||
            !DataLayouts.owns_point(scope, init_index) ||
            (@inbounds output_column[init_index] = transform(init))
        # The result for each input level is stored at the same output level, except in
        # upward accumulations across staggered levels, which store it one level higher
        # (center-to-face) or one level lower (face-to-center): the difference between the
        # numbers of output and input levels, which is known from the types of the columns,
        # so that the range check below is compiled away.
        offset = reverse ? 0 : length(output_column) - length(input_column)
        values = Broadcast.instantiate(Fields.field_values(scanned_column(input_column)))
        DataLayouts.scan_points(f, values, reverse, init) do index, value
            @inline
            1 <= index + offset <= length(output_column) &&
                (@inbounds output_column[index + offset] = transform(value))
            nothing
        end
    end
    input′ = loop_input(input)
    mask = Spaces.get_mask(axes(input))
    DataLayouts._foreach_slice(
        column,
        ColumnScan(f, _column_accumulate!),
        mask,
        Val(false),
        serial_column_loop(f),
        output,
        input′,
    )
end
