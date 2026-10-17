import ..Utilities: fieldtype_vals

const SingleValue = Union{Number, AbstractTensor}

"""
    project_for_mul(axes, y, lg)

Project `y` so that multiplying it by a value of type `X` does not raise a
`DimensionMismatch` error, where `axes` is `_dual_axes_for_projection(X)`: the first axis of
every tensor in `y` is projected onto the dual of the last axis of the corresponding
component of `X`, using `lg` (see [`projection_metric`](@ref)). Other values are unchanged.
"""
@inline project_for_mul(::Nothing, y, _) = y
@inline project_for_mul(::Components, y, _) = y
@inline project_for_mul(axis::Components, y::AbstractTensor, lg) = project(axis, y, lg)

"""
    projection_metric(axes, Y, lg)

Return the part of `lg` (a `LocalGeometry` or a `Field` of them) that
[`project_for_mul`](@ref)`(axes, y, lg)` reads for values `y` of type `Y`: the metric its
projected tensors need, `nothing` if none, or all of `lg` if they need different ones.
"""
@inline function projection_metric(axes, ::Type{Y}, lg) where {Y}
    metric = projected_metric(axes, Val(Y), lg)
    return isnothing(metric) ? nothing : something(metric)
end

# Like projection_metric, but in a Some; metrics are compared by type (unique per lg field).
@inline projected_metric(::Nothing, _, _) = nothing
@inline projected_metric(::Components, _, _) = nothing
@inline projected_metric(axis::Components, ::Val{Y}, lg) where {Y <: AbstractTensor} =
    Some(metric_for_components_type(components_type(axis), Y, lg))
@inline combine_projected_metrics(metric1, metric2, lg) =
    isnothing(metric1) ? metric2 :
    isnothing(metric2) || typeof(metric1) == typeof(metric2) ? metric1 : Some(lg)

"""
    _dual_axes_for_projection(X)

Return the axes that the second operand of a multiplication must be projected onto for
entries of type `X` in the first operand, or `nothing` if no projection is
needed. For entries with multiple components that do not all share one axis, the
result is a `Tuple` (or `NamedTuple`) of axes that pairs componentwise with the entry, with
`nothing` for the components that need no projection.

The result must reduce to a compile-time constant (a type parameter of the projection in
`MatrixFields.projected_operand`). The recursion stays
foldable because components are traversed as `Val`s of their field types
(`fieldtype_vals`), which inference specializes unconditionally instead of
hitting its recursion limiter; new methods for new entry types (see
`auto_broadcaster_methods.jl`) must preserve this, i.e. only branch on type
information.
"""
@inline _dual_axes_for_projection(::Val{X}) where {X} =
    _dual_axes_for_projection(X)
@inline _dual_axes_for_projection(::Type{X}) where {X <: Tensor{2}} =
    dual(tensor_axes(X)[2])
# Entries with multiple components (Tuples or NamedTuples, and AutoBroadcasters
# of them; see auto_broadcaster_methods.jl) pair componentwise in a
# multiplication, so the dual axes form a matching Tuple or NamedTuple.
@inline function _dual_axes_for_projection(::Type{X}) where {X <: Union{Tuple, NamedTuple}}
    axes = unrolled_map(_dual_axes_for_projection, fieldtype_vals(X))
    unrolled_all(isnothing, axes) && return nothing
    # When every component projects onto the same axis, collapse the Tuple into
    # that single axis. Projecting every tensor leaf onto it is equivalent to
    # pairing componentwise, and avoids copying a second operand that is not multi-component
    # (e.g. a NamedTuple of covectors times a vector, as in `(ᶜρχ, ᶠu₃)` blocks).
    unrolled_allequal(axes) && return first(axes)
    return X <: NamedTuple ? NamedTuple{fieldnames(X)}(axes) : axes
end
@inline function _dual_axes_for_projection(::Type{X}) where {X}
    Y = eltype(X)
    Y === X && return nothing
    return _dual_axes_for_projection(Y)
end
