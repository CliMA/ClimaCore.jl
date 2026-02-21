import ..Utilities: AutoBroadcaster, broadcast_over_element_types

# TODO: Refactor the Geometry module to avoid defining the methods in this file

(::Type{T})(u::AutoBroadcaster) where {T <: AxisTensor} = broadcast(T, u)
(::Type{T})(u::AutoBroadcaster, local_geometry) where {T <: AxisTensor} =
    broadcast(Base.Fix2(T, local_geometry), u)

covariant1(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(covariant1, local_geometry), u)
covariant2(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(covariant2, local_geometry), u)
covariant3(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(covariant3, local_geometry), u)
contravariant1(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(contravariant1, local_geometry), u)
contravariant2(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(contravariant2, local_geometry), u)
contravariant3(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(contravariant3, local_geometry), u)
Jcontravariant3(u::AutoBroadcaster, local_geometry) =
    broadcast(Base.Fix2(Jcontravariant3, local_geometry), u)

divergence_result_type(::Type{T}) where {T <: AutoBroadcaster} =
    typeof(broadcast_over_element_types(zero ∘ divergence_result_type, T))
gradient_result_type(val, ::Type{T}) where {T <: AutoBroadcaster} =
    typeof(broadcast_over_element_types(zero ∘ Base.Fix1(gradient_result_type, val), T))
curl_result_type(val, ::Type{T}) where {T <: AutoBroadcaster} =
    typeof(broadcast_over_element_types(zero ∘ Base.Fix1(curl_result_type, val), T))

mul_with_projection(x::AutoBroadcaster, y::AutoBroadcaster, lg) =
    broadcast((x, y) -> mul_with_projection(x, y, lg), x, y)
mul_with_projection(x::AutoBroadcaster, y, lg) =
    broadcast(x -> mul_with_projection(x, y, lg), x)
mul_with_projection(x, y::AutoBroadcaster, lg) =
    broadcast(y -> mul_with_projection(x, y, lg), y)

mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {X <: AutoBroadcaster, Y <: AutoBroadcaster} =
    typeof(broadcast_over_element_types(zero ∘ mul_return_type, X, Y))
mul_return_type(::Type{X}, ::Type{Y}) where {X <: AutoBroadcaster, Y} =
    typeof(broadcast_over_element_types(zero ∘ Base.Fix2(mul_return_type, Y), X))
mul_return_type(::Type{X}, ::Type{Y}) where {X, Y <: AutoBroadcaster} =
    typeof(broadcast_over_element_types(zero ∘ Base.Fix1(mul_return_type, X), Y))
