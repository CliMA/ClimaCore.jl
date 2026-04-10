import LinearAlgebra: issymmetric

isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, MT <: Metric}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` (reference space) to `x` (physical space)"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "Riemannian metric (wraps ∂x∂ξ as a Tensor with Orthonormal/Covariant bases)"
    metric::MT
end

@inline Base.getproperty(lg::LocalGeometry, name::Symbol) =
    _getproperty(lg, Val(name))

@inline _getproperty(lg::LocalGeometry, ::Val{:invJ}) = inv(getfield(lg, :J))
@inline _getproperty(lg::LocalGeometry, ::Val{:∂x∂ξ}) = getfield(lg, :metric).tensor
@inline _getproperty(lg::LocalGeometry, ::Val{:∂ξ∂x}) = inv(getfield(lg, :metric).tensor)
@inline _getproperty(lg::LocalGeometry, ::Val{:gⁱʲ}) = begin
    ∂ξ∂x = inv(getfield(lg, :metric).tensor)
    ∂ξ∂x * ∂ξ∂x'
end
@inline _getproperty(lg::LocalGeometry, ::Val{:gᵢⱼ}) = begin
    ∂x∂ξ = getfield(lg, :metric).tensor
    ∂x∂ξ' * ∂x∂ξ
end
@inline _getproperty(lg::LocalGeometry, ::Val{name}) where {name} =
    getfield(lg, name)

# Primary constructor: accepts a Tensor{2} with Orthonormal/Covariant bases
@inline function LocalGeometry(
    coordinates::C,
    J::FT,
    WJ::FT,
    ∂x∂ξ::Tensor{2},
) where {C, FT}
    names = basis_vector_names(axes(∂x∂ξ, 1))
    g = Metric(∂x∂ξ)
    gⁱʲ = change_of_basis_tensor(g, Contravariant(), Contravariant())
    gᵢⱼ = change_of_basis_tensor(g, Covariant(), Covariant())
    isapproxsymmetric(parent(gⁱʲ)) || error("gⁱʲ is not symmetric.")
    isapproxsymmetric(parent(gᵢⱼ)) || error("gᵢⱼ is not symmetric.")
    return LocalGeometry{names, C, FT, typeof(g)}(coordinates, J, WJ, g)
end

"""
    SurfaceGeometry

The necessary local metric information defined at each node on each surface.
"""
struct SurfaceGeometry{FT, N}
    "surface Jacobian determinant, multiplied by the surface quadrature weight"
    sWJ::FT
    "surface outward pointing normal vector"
    normal::N
end

"""
    CoordinateOnlyGeometry

The necessary coordinates information defined at each node.

This is currently used for constructing spaces with pressure as the vertical
coordinate.
"""
struct CoordinateOnlyGeometry{C <: AbstractPoint}
    "Coordinates of the current point"
    coordinates::C
end

undertype(::Type{<:LocalGeometry{I, C, FT}}) where {I, C, FT} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
undertype(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = eltype(C)

"""
    blockmat(a, b[, c])

Construct a block-diagonal (or block-lower-triangular) Tensor{2} from sub-blocks.
Uses `combine_bases` and `reshape` to zero-fill missing components.
"""
function blockmat(a::Tensor{2}, b::Tensor{2}, ::Nothing = nothing)
    new_bases = (
        combine_bases(axes(a, 1), axes(b, 1)),
        combine_bases(axes(a, 2), axes(b, 2)),
    )
    return reshape(a, new_bases) + reshape(b, new_bases)
end

function blockmat(a::Tensor{2}, b::Tensor{2}, c::Tensor{2})
    new_bases = (
        combine_bases(axes(a, 1), axes(b, 1)),
        combine_bases(axes(a, 2), axes(b, 2)),
    )
    return reshape(a, new_bases) + reshape(b, new_bases) + reshape(c, new_bases)
end
