import ..Utilities: MathMapper, math_mapper_broadcast, math_mapper_type_broadcast

(::Type{T})(u::MathMapper, local_geometry) where {T <: AxisTensor} =
    math_mapper_broadcast(Base.Fix2(T, local_geometry), u)

covariant1(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(covariant1, local_geometry), u)
covariant2(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(covariant2, local_geometry), u)
covariant3(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(covariant3, local_geometry), u)
contravariant1(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(contravariant1, local_geometry), u)
contravariant2(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(contravariant2, local_geometry), u)
contravariant3(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(contravariant3, local_geometry), u)
Jcontravariant3(u::MathMapper, local_geometry) =
    math_mapper_broadcast(Base.Fix2(Jcontravariant3, local_geometry), u)

@inline divergence_result_type(::Type{T}) where {T <: MathMapper} =
    typeof(math_mapper_type_broadcast(zero ∘ divergence_result_type, T))
@inline gradient_result_type(val, ::Type{T}) where {T <: MathMapper} = typeof(
    math_mapper_type_broadcast(zero ∘ Base.Fix1(gradient_result_type, val), T),
)
@inline curl_result_type(val, ::Type{T}) where {T <: MathMapper} = typeof(
    math_mapper_type_broadcast(zero ∘ Base.Fix1(curl_result_type, val), T),
)

mul_with_projection(x::MathMapper, y::MathMapper, lg) =
    math_mapper_broadcast((x, y) -> mul_with_projection(x, y, lg), x, y)
mul_with_projection(x::MathMapper, y, lg) =
    math_mapper_broadcast(x -> mul_with_projection(x, y, lg), x)
mul_with_projection(x, y::MathMapper, lg) =
    math_mapper_broadcast(y -> mul_with_projection(x, y, lg), y)

mul_return_type(::Type{X}, ::Type{Y}) where {X <: MathMapper, Y <: MathMapper} =
    typeof(math_mapper_type_broadcast(zero ∘ mul_return_type, X, Y))
mul_return_type(::Type{X}, ::Type{Y}) where {X <: MathMapper, Y} =
    typeof(math_mapper_type_broadcast(zero ∘ Base.Fix2(mul_return_type, Y), X))
mul_return_type(::Type{X}, ::Type{Y}) where {X, Y <: MathMapper} =
    typeof(math_mapper_type_broadcast(zero ∘ Base.Fix1(mul_return_type, X), Y))
