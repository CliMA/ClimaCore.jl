"""
    mul_with_projection(x, y, lg)

Similar to `x * y`, except that this version automatically projects `y` to avoid
`DimensionMismatch` errors for `AxisTensor`s. For example, if `x` is a covector
along the `Covariant3Axis` (e.g., `Covariant3Vector(1)'`), then `y` will be
projected onto the `Contravariant3Axis`. In general, the first axis of `y` will
be projected onto the dual of the last axis of `x`.
"""
mul_with_projection(x, y, _) = x * y
mul_with_projection(
    x::Union{Geometry.AdjointAxisVector, Geometry.Axis2TensorOrAdj},
    y::Geometry.AxisTensor,
    lg,
) = x * Geometry.project(Geometry.dual(axes(x)[2]), y, lg)

"""
    rmul_with_projection(x, y, lg)

Similar to `rmul(x, y)`, except that this version calls `mul_with_projection`
instead of `*`.
"""
rmul_with_projection(x, y, lg) =
    rmap((x′, y′) -> mul_with_projection(x′, y′, lg), x, y)
rmul_with_projection(x::SingleValue, y, lg) =
    rmap(y′ -> mul_with_projection(x, y′, lg), y)
rmul_with_projection(x, y::SingleValue, lg) =
    rmap(x′ -> mul_with_projection(x′, y, lg), x)
rmul_with_projection(x::SingleValue, y::SingleValue, lg) =
    mul_with_projection(x, y, lg)

axis_tensor_type(::Type{T}, ::Type{Tuple{A1}}) where {T, A1} =
    Geometry.AxisVector{T, A1, SVector{Geometry._length(A1), T}}
function axis_tensor_type(::Type{T}, ::Type{Tuple{A1, A2}}) where {T, A1, A2}
    N1 = Geometry._length(A1)
    N2 = Geometry._length(A2)
    return Geometry.Axis2Tensor{T, Tuple{A1, A2}, SMatrix{N1, N2, T, N1 * N2}}
end

adjoint_type(::Type{A}) where {A} = Adjoint{eltype(A), A}
adjoint_type(::Type{A}) where {T, S, A <: Adjoint{T, S}} = S

axis1(::Type{<:Geometry.Axis2Tensor{<:Any, <:Tuple{A, Any}}}) where {A} = A
axis1(::Type{<:Geometry.AdjointAxis2Tensor{<:Any, <:Tuple{Any, A}}}) where {A} =
    A

axis2(::Type{<:Geometry.Axis2Tensor{<:Any, <:Tuple{Any, A}}}) where {A} = A
axis2(::Type{<:Geometry.AdjointAxis2Tensor{<:Any, <:Tuple{A, Any}}}) where {A} =
    A
