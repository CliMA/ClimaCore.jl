import ..Utilities: fieldtype_vals

const SingleValue = Union{Number, AbstractTensor}

"""
    project_for_mul(axes, y, lg)

Project `y` so that multiplying it by a value of type `X` does not raise a
`DimensionMismatch` error, where `axes` is `_dual_axes_for_projection(X)`. For example,
if `X` is a covector along the `Covariant3Axis` (e.g., the type of
`Covariant3Vector(1)'`), then `y` is projected onto the `Contravariant3Axis`. In general,
the first axis of every tensor in `y` is projected onto the dual of the last axis of the
corresponding component of `X`, using `lg`, which is a `LocalGeometry` or the part of one
returned by [`projection_metric`](@ref). Values that are not tensors are left unchanged,
as is all of `y` when `axes` is `nothing`.
"""
@inline project_for_mul(::Nothing, y, _) = y
@inline project_for_mul(::Components, y, _) = y
@inline project_for_mul(axis::Components, y::AbstractTensor, lg) = project(axis, y, lg)

"""
    projection_metric(axes, Y, lg)

Return the part of `lg` (a `LocalGeometry`, or a `Field` of them) that
[`project_for_mul`](@ref)`(axes, y, lg)` reads for values `y` of type `Y`. This is the
metric that every projected tensor in `y` needs (see `metric_for_components_type`), or
`nothing` when none of them needs a metric, or all of `lg` when they need different
metrics.
"""
@inline function projection_metric(axes, ::Type{Y}, lg) where {Y}
    metric = projected_metric(axes, Val(Y), lg)
    return isnothing(metric) ? nothing : something(metric)
end

# The metric that project_for_mul reads for every tensor in a value of type Y,
# wrapped in a Some, or nothing when no tensor is projected. Metrics are only
# compared by their types, which differ for the different fields of lg.
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
result is a `Tuple` (or a `NamedTuple`, for `NamedTuple` entries) of axes that pairs
componentwise with the entry, with `nothing` for the components that need no
projection. See [`project_for_mul`](@ref) for the projection itself.

The result must reduce to a compile-time constant, since it is a type parameter of the
broadcast expression that projects the second operand of a band matrix product (see
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
# multiplication, so the dual axes form a matching Tuple or NamedTuple, with
# `nothing` for components that need no projection. When no component needs
# projection, the result is `nothing`.
@inline function _dual_axes_for_projection(
    ::Type{X},
) where {X <: Union{Tuple, NamedTuple}}
    axes = unrolled_map(_dual_axes_for_projection, fieldtype_vals(X))
    unrolled_all(isnothing, axes) && return nothing
    # When every component projects onto the same axis, collapse the Tuple into
    # that single axis. Projecting every tensor leaf onto it is equivalent to
    # pairing componentwise, and it also handles a second operand that is not
    # itself multi-component without copying it for every component (e.g. a
    # NamedTuple of covectors multiplying a single vector, as in `(ᶜρχ, ᶠu₃)`
    # blocks).
    unrolled_allequal(axes) && return first(axes)
    return X <: NamedTuple ? NamedTuple{fieldnames(X)}(axes) : axes
end
@inline function _dual_axes_for_projection(::Type{X}) where {X}
    Y = eltype(X)
    Y === X && return nothing
    return _dual_axes_for_projection(Y)
end
