import LinearAlgebra: Adjoint

const SingleValue = Union{Number, AbstractTensor}

"""
    mul_with_projection(x, y, lg)

Similar to `x * y`, except that this version automatically projects `y` to avoid
`DimensionMismatch` errors for `Tensor`s. For example, if `x` is a covector
along the `Covariant3Axis` (e.g., `Covariant3Vector(1)'`), then `y` will be
projected onto the `Contravariant3Axis`. In general, the first axis of `y` will
be projected onto the dual of the last axis of `x`.
"""
mul_with_projection(x, y, _) = x * y
mul_with_projection(x::Tensor{2}, y::AbstractTensor, lg) =
    x * project(dual(axes(x, 2)), y, lg)

# Construct a Tensor type from element type and bases tuple type
tensor_type(::Type{T}, ::Type{Tuple{B1}}) where {T, B1 <: Components} =
    Tensor{1, T, Tuple{B1}, SVector{length(B1.instance), T}}
function tensor_type(::Type{T}, ::Type{Tuple{B1, B2}}) where {T, B1 <: Components, B2 <: Components}
    N1 = length(B1.instance)
    N2 = length(B2.instance)
    return Tensor{2, T, Tuple{B1, B2}, SMatrix{N1, N2, T, N1 * N2}}
end
# Covector storage uses Adjoint{T, SVector} rather than SMatrix{1, N}
function tensor_type(
    ::Type{T}, ::Type{Tuple{ScalarComponents, B2}},
) where {T, B2 <: Components}
    N2 = length(B2.instance)
    return Tensor{2, T, Tuple{ScalarComponents, B2}, Adjoint{T, SVector{N2, T}}}
end

basis1(::Type{<:AbstractTensor{2, <:Any, <:Tuple{B, Any}}}) where {B} = B
basis2(::Type{<:AbstractTensor{2, <:Any, <:Tuple{Any, B}}}) where {B} = B

recursively_find_dual_axes_for_projection(
    ::Type{X},
) where {X <: Tensor{2}} = dual(Geometry.tensor_axes(X)[2])
@inline function recursively_find_dual_axes_for_projection(::Type{X}) where {X}
    Y = eltype(X)
    Y === X && return nothing
    return recursively_find_dual_axes_for_projection(Y)
end


"""
    mul_return_type(X, Y)

Return type of `mul_with_projection(x, y, lg)`. Equivalent to
`Base._return_type(mul_with_projection, Tuple{X, Y, LG})` but explicit so
that MatrixFields' eltype inference always sees a concrete type — internal
`_return_type` can widen to `Union`/`Any` and is unstable across Julia versions.

The methods cover six distinct result shapes (scalar×scalar, scalar×tensor,
covector×vector, vector×covector, 2-tensor×vector, 2-tensor×2-tensor) where
the output `ndims` goes up, down, or stays the same — no single formula
fits all of them.

Future cleanup: try replacing with `Base._return_type` and keep only methods
that fail; collapse the two `Tensor{2}*Tensor{N}` cases via
`Base.tail(axes(Y))`; move `tensor_type`/`basis1`/`basis2` to `tensors.jl`.
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

# Number * Tensor = Tensor (same bases, promoted element type)
# For covectors (component storage is Adjoint), preserve the exact type rather
# than reconstructing via tensor_type (which would use SMatrix instead of Adjoint).
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {T, B, C, X <: Number, Y <: Tensor{<:Any, T, B, C}} =
    Tensor{ndims(Y), promote_type(X, T), B, C}
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {T, B, C, X <: Tensor{<:Any, T, B, C}, Y <: Number} =
    Tensor{ndims(X), promote_type(T, Y), B, C}

# Covector * Vector = scalar (dot product)
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {T1, T2, X <: Covector{T1}, Y <: Tensor{1, T2}} =
    promote_type(T1, T2)

# Vector * Covector = 2-tensor (outer product)
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {
    T1, T2, B1, B2,
    X <: Tensor{1, T1, Tuple{B1}},
    Y <: Covector{T2, <:Tuple{<:Any, B2}},
} =
    tensor_type(promote_type(T1, T2), Tuple{B1, B2})

# 2-Tensor * Vector = Vector
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {T1, T2, X <: Tensor{2, T1}, Y <: Tensor{1, T2}} =
    tensor_type(promote_type(T1, T2), Tuple{basis1(X)})

# 2-Tensor * 2-Tensor = 2-Tensor
mul_return_type(
    ::Type{X}, ::Type{Y},
) where {T1, T2, X <: Tensor{2, T1}, Y <: Tensor{2, T2}} =
    tensor_type(promote_type(T1, T2), Tuple{basis1(X), basis2(Y)})
