import LinearAlgebra: Adjoint, AdjointAbsVec
import .RecursiveApply: rmap, rmaptype

# Types that are treated as single values when using matrix fields.
# AbstractCovector (Tensor{2} with ScalarBasis) is already covered by AbstractTensor.
# Adjoint{T, <:AbstractTensor} covers the case where adjoint() returns a Julia Adjoint
# wrapper rather than our Covector type (e.g., from composition or old codepaths).
const SingleValue = Union{Number, AbstractTensor, Adjoint{<:Any, <:AbstractTensor}}

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

"""
    rmul_with_projection(x, y, lg)

Similar to `rmul(x, y)`, except that this version calls `mul_with_projection`
instead of `*`.
"""
rmul_with_projection(x, y, lg) = rmap((x′, y′) -> mul_with_projection(x′, y′, lg), x, y)
rmul_with_projection(x::SingleValue, y, lg) = rmap(y′ -> mul_with_projection(x, y′, lg), y)
rmul_with_projection(x, y::SingleValue, lg) = rmap(x′ -> mul_with_projection(x′, y, lg), x)
rmul_with_projection(x::SingleValue, y::SingleValue, lg) = mul_with_projection(x, y, lg)

# Construct a Tensor type from element type and bases tuple type
tensor_type(::Type{T}, ::Type{Tuple{B1}}) where {T, B1 <: Basis} =
    Tensor{1, T, Tuple{B1}, SVector{length(B1.instance), T}}
function tensor_type(::Type{T}, ::Type{Tuple{B1, B2}}) where {T, B1 <: Basis, B2 <: Basis}
    N1 = length(B1.instance)
    N2 = length(B2.instance)
    return Tensor{2, T, Tuple{B1, B2}, SMatrix{N1, N2, T, N1 * N2}}
end
# Covector storage uses Adjoint{T, SVector} rather than SMatrix{1, N}
function tensor_type(
    ::Type{T}, ::Type{Tuple{ScalarBasis, B2}},
) where {T, B2 <: Basis}
    N2 = length(B2.instance)
    return Tensor{2, T, Tuple{ScalarBasis, B2}, Adjoint{T, SVector{N2, T}}}
end

basis1(::Type{<:Tensor{2, <:Any, <:Tuple{B, Any}}}) where {B} = B
basis2(::Type{<:Tensor{2, <:Any, <:Tuple{Any, B}}}) where {B} = B

"""
    needs_projection(::Type{X}, ::Type{Y})

Returns `true` if multiplying an object of type `X` with an object of type `Y` would require
projection. This always returns false if `X` or `Y` are a `Tuple` or `NamedTuple` with
eltype any.
"""
needs_projection(::Type{X}, ::Type{Y}) where {X <: Number, Y <: SingleValue} = false
needs_projection(::Type{X}, ::Type{Y}) where {X <: SingleValue, Y <: SingleValue} = false
function needs_projection(::Type{X}, ::Type{Y}) where {X, Y}
    (eltype(X) === Any || eltype(Y) === Any) && return false
    needs_projection(eltype(X), eltype(Y))
end
needs_projection(
    ::Type{X},
    ::Type{Y},
) where {X <: Tensor{2}, Y <: AbstractTensor} =
    axes(X.instance)[2] != Geometry.dual(axes(Y.instance)[1])
function needs_projection(
    ::Type{X},
    ::Type{Y},
) where {X <: SingleValue, Y <: Union{Tuple, NamedTuple}}
    X <: Number && return false
    eltype(Y) === Any && return false
    needs_projection(X, eltype(Y))
end
function needs_projection(
    ::Type{X},
    ::Type{Y},
) where {X <: Union{Tuple, NamedTuple}, Y <: SingleValue}
    Y <: Number && return false
    eltype(X) === Any && return false
    needs_projection(eltype(X), Y)
end

recursively_find_dual_axes_for_projection(
    ::Type{X},
) where {X <: Tensor{2}} = dual(axes(X)[2])
recursively_find_dual_axes_for_projection(::Type{X}) where {X} =
    recursively_find_dual_axes_for_projection(eltype(X))


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

"""
    rmul_return_type(X, Y)

Computes the return type of `rmul_with_projection(x, y, lg)`, where `x isa X`
and `y isa Y`. This can also be used to obtain the return type of `rmul(x, y)`,
although `rmul(x, y)` will throw an error when projection is necessary.

Note that this is equivalent to calling the internal function `_return_type`:
`Base._return_type(rmul_with_projection, Tuple{X, Y, LG})`, where `lg isa LG`.
"""
rmul_return_type(::Type{X}, ::Type{Y}) where {X, Y} =
    rmaptype((X′, Y′) -> mul_return_type(X′, Y′), X, Y)
rmul_return_type(::Type{X}, ::Type{Y}) where {X <: SingleValue, Y} =
    rmaptype(Y′ -> mul_return_type(X, Y′), Y)
rmul_return_type(::Type{X}, ::Type{Y}) where {X, Y <: SingleValue} =
    rmaptype(X′ -> mul_return_type(X′, Y), X)
rmul_return_type(::Type{X}, ::Type{Y}) where {X <: SingleValue, Y <: SingleValue} =
    mul_return_type(X, Y)
