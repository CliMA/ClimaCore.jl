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
column share the work of reducing it, which can change the result by roundoff.

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
function column_reduce!(
    f::F,
    output::Fields.Field,
    input::Union{Fields.Field, Fields.PointwiseBroadcasted, StencilBroadcasted};
    init = DataLayouts.NoInit(),
    transform::T = identity,
    reverse::Bool = false,
) where {F, T}
    # The output point is set by the first thread of each column (see owns_point).
    function _column_reduce!(output_point, input_column)
        values = Broadcast.instantiate(Fields.field_values(apply_operators(input_column)))
        result = DataLayouts.fold_points(f, values, reverse, init)
        DataLayouts.owns_point(DataLayouts.DataScope(output_point), 1) &&
            (@inbounds output_point[] = transform(result))
        return nothing
    end
    input′ = loop_input(input)
    mask = Spaces.get_mask(axes(input))
    DataLayouts._foreach_slice(column, _column_reduce!, mask, Val(false), output, input′)
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
function column_accumulate!(
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
    is_f2c =
        Spaces.staggering(axes(input)) == Spaces.CellFace() &&
        Spaces.staggering(axes(output)) == Spaces.CellCenter()
    init == DataLayouts.NoInit() && is_c2f &&
        error("init must be specified for center-to-face accumulation")
    # The result for each input level is stored at the same output level, except
    # in upward accumulations across staggered levels, which store it one level
    # higher (center-to-face) or one level lower (face-to-center).
    offset = reverse ? 0 : (is_c2f ? 1 : (is_f2c ? -1 : 0))
    function _column_accumulate!(output_column, input_column)
        scope = DataLayouts.DataScope(output_column, input_column)
        init_index = reverse ? length(output_column) : 1
        init isa DataLayouts.NoInit || !is_c2f ||
            !DataLayouts.owns_point(scope, init_index) ||
            (@inbounds output_column[init_index] = transform(init))
        values = Broadcast.instantiate(Fields.field_values(apply_operators(input_column)))
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
        _column_accumulate!,
        mask,
        Val(false),
        output,
        input′,
    )
end
