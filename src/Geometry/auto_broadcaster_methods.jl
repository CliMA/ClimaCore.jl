import ..Utilities:
    AutoBroadcaster, nested_auto_broadcast, nested_auto_broadcast_result_type

# TODO: Avoid defining these methods by refactoring the Geometry module so that
# all relevant functionality is expressed in terms of standard math operations

for f in (:covariant, :contravariant), n in (1, 2, 3)
    @eval $(Symbol(f, n))(x::AutoBroadcaster, lg) =
        nested_auto_broadcast(Base.Fix2($(Symbol(f, n)), lg), x)
end
Jcontravariant3(x::AutoBroadcaster, lg) =
    nested_auto_broadcast(Base.Fix2(Jcontravariant3, lg), x)

mul_with_projection(x::AutoBroadcaster, y::AutoBroadcaster, lg) =
    nested_auto_broadcast((x, y) -> mul_with_projection(x, y, lg), x, y)
mul_with_projection(x::AutoBroadcaster, y, lg) =
    nested_auto_broadcast(x -> mul_with_projection(x, y, lg), x)
mul_with_projection(x, y::AutoBroadcaster, lg) =
    nested_auto_broadcast(y -> mul_with_projection(x, y, lg), y)

needs_projection(
    ::Type{X},
    ::Type{Y},
) where {X <: AutoBroadcaster, Y <: AutoBroadcaster} =
    needs_projection(eltype(X), eltype(Y))
needs_projection(::Type{X}, ::Type{Y}) where {X <: AutoBroadcaster, Y} =
    needs_projection(eltype(X), Y)
needs_projection(::Type{X}, ::Type{Y}) where {X, Y <: AutoBroadcaster} =
    needs_projection(X, eltype(Y))

mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {X <: AutoBroadcaster, Y <: AutoBroadcaster} =
    nested_auto_broadcast_result_type(mul_return_type, X, Y)
mul_return_type(::Type{X}, ::Type{Y}) where {X <: AutoBroadcaster, Y} =
    nested_auto_broadcast_result_type(Base.Fix2(mul_return_type, Y), X)
mul_return_type(::Type{X}, ::Type{Y}) where {X, Y <: AutoBroadcaster} =
    nested_auto_broadcast_result_type(Base.Fix1(mul_return_type, X), Y)

divergence_result_type(::Type{X}) where {X <: AutoBroadcaster} =
    nested_auto_broadcast_result_type(divergence_result_type, X)
gradient_result_type(val, ::Type{X}) where {X <: AutoBroadcaster} =
    nested_auto_broadcast_result_type(Base.Fix1(gradient_result_type, val), X)
curl_result_type(val, ::Type{X}) where {X <: AutoBroadcaster} =
    nested_auto_broadcast_result_type(Base.Fix1(curl_result_type, val), X)
