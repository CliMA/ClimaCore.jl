import ..Utilities:
    AutoBroadcaster,
    nested_broadcast,
    nested_broadcast_result_type,
    unwrap,
    add_auto_broadcasters

# TODO: Avoid defining these methods by refactoring the Geometry module so that
# all relevant functionality is expressed in terms of standard math operations

(::Type{T})(x::AutoBroadcaster) where {T <: AbstractTensor} = nested_broadcast(T, x)

for f in (:covariant, :contravariant), n in (1, 2, 3)
    @eval $(Symbol(f, n))(x::AutoBroadcaster, lg) =
        nested_broadcast(Base.Fix2($(Symbol(f, n)), lg), x)
end
Jcontravariant3(x::AutoBroadcaster, lg) =
    nested_broadcast(Base.Fix2(Jcontravariant3, lg), x)

# An AutoBroadcaster entry pairs componentwise like its wrapped collection
# (see `project_for_mul` below and `_dual_axes_for_projection`).
@inline _dual_axes_for_projection(::Type{X}) where {X <: AutoBroadcaster} =
    _dual_axes_for_projection(unwrap(X))

# A single axis projects every tensor in an AutoBroadcaster, while the axes of
# an entry with multiple components pair componentwise with the components of an
# AutoBroadcaster, or with copies of a single tensor, like the multiplication of
# the entry by the projected value. Nested axes are wrapped in AutoBroadcasters,
# so that nested_broadcast pairs every level and only projects single tensors.
@inline project_for_mul(axis::Components, y::AutoBroadcaster, lg) =
    nested_broadcast(y -> project_for_mul(axis, y, lg), y)
@inline project_for_mul(::Union{Tuple, NamedTuple}, y, _) = y
@inline project_for_mul(
    axes::Union{Tuple, NamedTuple},
    y::Union{AbstractTensor, AutoBroadcaster},
    lg,
) = nested_broadcast(
    (y, axis) -> project_for_mul(axis, y, lg),
    y,
    add_auto_broadcasters(axes),
)

@inline projected_metric(axis::Components, ::Val{Y}, lg) where {Y <: AutoBroadcaster} =
    unrolled_mapreduce(
        val -> projected_metric(axis, val, lg),
        (metric1, metric2) -> combine_projected_metrics(metric1, metric2, lg),
        fieldtype_vals(unwrap(Y));
        init = nothing,
    )
@inline projected_metric(::Union{Tuple, NamedTuple}, _, _) = nothing
@inline projected_metric(
    axes::Union{Tuple, NamedTuple},
    val::Val{<:AbstractTensor},
    lg,
) = unrolled_mapreduce(
    component_axes -> projected_metric(component_axes, val, lg),
    (metric1, metric2) -> combine_projected_metrics(metric1, metric2, lg),
    values(axes);
    init = nothing,
)
@inline projected_metric(
    axes::Union{Tuple, NamedTuple},
    ::Val{Y},
    lg,
) where {Y <: AutoBroadcaster} = unrolled_mapreduce(
    (component_axes, val) -> projected_metric(component_axes, val, lg),
    (metric1, metric2) -> combine_projected_metrics(metric1, metric2, lg),
    values(axes),
    fieldtype_vals(unwrap(Y));
    init = nothing,
)

divergence_result_type(::Type{X}) where {X <: AutoBroadcaster} =
    nested_broadcast_result_type(divergence_result_type, X)
# The Union{} methods terminate the recursion above when inference reaches a
# bottom type, which happens while it is still widening the element types of a
# nested broadcast. Without them, nested_broadcast_result_type would recurse on
# Union{} and inference would give up, so these are needed for inference rather
# than for any call that can actually happen at run time.
divergence_result_type(::Type{Union{}}) = Union{}
gradient_result_type(val, ::Type{X}) where {X <: AutoBroadcaster} =
    nested_broadcast_result_type(Base.Fix1(gradient_result_type, val), X)
gradient_result_type(val, ::Type{Union{}}) = Union{}
curl_result_type(val, ::Type{X}) where {X <: AutoBroadcaster} =
    nested_broadcast_result_type(Base.Fix1(curl_result_type, val), X)
curl_result_type(val, ::Type{Union{}}) = Union{}
