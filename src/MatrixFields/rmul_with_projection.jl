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

"""
    mul_return_type(X, Y)

Computes the return type of `mul_with_projection(x, y, lg)`, where `x isa X`
and `y isa Y`. This can also be used to obtain the return type of `x * y`,
although `x * y` will throw an error when projection is necessary.

Note that this is equivalent to calling the internal function `_return_type`:
`Base._return_type(mul_with_projection, Tuple{X, Y, LG})`, where `lg isa LG`.
"""
mul_return_type(::Type{X}, ::Type{Y}) where {X, Y} = error(
    "Unable to infer return type: Multiplying an object of type $X with an \
     object of type $Y will result in a method error",
)
# Note: If the behavior of * changes for any relevant types, the corresponding
# methods below should be updated.

# Methods from Base:
mul_return_type(::Type{X}, ::Type{Y}) where {X <: Number, Y <: Number} =
    promote_type(X, Y)
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {X <: AdjointAbsVec, Y <: AbstractMatrix} =
    adjoint_type(mul_return_type(adjoint_type(Y), adjoint_type(X)))

# Methods from ClimaCore: 
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {T, N, A, X <: Number, Y <: Geometry.AxisTensor{T, N, A}} =
    axis_tensor_type(promote_type(X, T), A)
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {T, N, A, X <: Geometry.AxisTensor{T, N, A}, Y <: Number} =
    axis_tensor_type(promote_type(T, Y), A)
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {T, N, A, X <: Number, Y <: Geometry.AdjointAxisTensor{T, N, A}} =
    adjoint_type(axis_tensor_type(promote_type(X, T), A))
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {T, N, A, X <: Geometry.AdjointAxisTensor{T, N, A}, Y <: Number} =
    adjoint_type(axis_tensor_type(promote_type(T, Y), A))
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {
    T1,
    T2,
    X <: Geometry.AdjointAxisVector{T1},
    Y <: Geometry.AxisVector{T2},
} = promote_type(T1, T2) # This comes from the definition of dot.
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {
    T1,
    T2,
    A1,
    A2,
    X <: Geometry.AxisVector{T1, A1},
    Y <: Geometry.AdjointAxisVector{T2, A2},
} = axis_tensor_type(promote_type(T1, T2), Tuple{A1, A2})
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {
    T1,
    T2,
    X <: Geometry.Axis2TensorOrAdj{T1},
    Y <: Geometry.AxisVector{T2},
} = axis_tensor_type(promote_type(T1, T2), Tuple{axis1(X)})
mul_return_type(
    ::Type{X},
    ::Type{Y},
) where {
    T1,
    T2,
    X <: Geometry.Axis2TensorOrAdj{T1},
    Y <: Geometry.Axis2TensorOrAdj{T2},
} = axis_tensor_type(promote_type(T1, T2), Tuple{axis1(X), axis2(Y)})

"""
    rmul_return_type(X, Y)

Computes the return type of `rmul_with_projection(x, y, lg)`, where `x isa X`
and `y isa Y`. This can also be used to obtain the return type of `rmul(x, y)`,
although `rmul(x, y)` will throw an error when projection is necessary.

Note that this is similar to calling the internal function `Base.promote_op`:
`Base.promote_op(rmul_with_projection, Tuple{X, Y, LG})`, where `lg isa LG`.
"""
rmul_return_type(::Type{X}, ::Type{Y}) where {X, Y} =
    rmaptype((X′, Y′) -> mul_return_type(X′, Y′), X, Y)
rmul_return_type(::Type{X}, ::Type{Y}) where {X <: SingleValue, Y} =
    rmaptype(Y′ -> mul_return_type(X, Y′), Y)
rmul_return_type(::Type{X}, ::Type{Y}) where {X, Y <: SingleValue} =
    rmaptype(X′ -> mul_return_type(X′, Y), X)
rmul_return_type(
    ::Type{X},
    ::Type{Y},
) where {X <: SingleValue, Y <: SingleValue} = mul_return_type(X, Y)

#####
##### Old
#####

rmul_return_type_old(::Type{X}, ::Type{Y}) where {X, Y} =
    rmaptype((X′, Y′) -> mul_return_type(X′, Y′), X, Y)
rmul_return_type_old(::Type{X}, ::Type{Y}) where {X <: SingleValue, Y} =
    rmaptype(Y′ -> mul_return_type(X, Y′), Y)
rmul_return_type_old(::Type{X}, ::Type{Y}) where {X, Y <: SingleValue} =
    rmaptype(X′ -> mul_return_type(X′, Y), X)
rmul_return_type_old(
    ::Type{X},
    ::Type{Y},
) where {X <: SingleValue, Y <: SingleValue} = mul_return_type(X, Y)
