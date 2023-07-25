
"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, S}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "inverse Jacobian"
    invJ::FT
    "Partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ"
    ∂x∂ξ::Axis2Tensor{FT, Tuple{LocalAxis{I}, CovariantAxis{I}}, S}
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, LocalAxis{I}}, S}
    "Contravariant metric tensor (inverse of gᵢⱼ), transforms covariant to contravariant vector components"
    gⁱʲ::Axis2Tensor{FT, Tuple{ContravariantAxis{I}, ContravariantAxis{I}}, S}
    "Covariant metric tensor (gᵢⱼ), transforms contravariant to covariant vector components"
    gᵢⱼ::Axis2Tensor{FT, Tuple{CovariantAxis{I}, CovariantAxis{I}}, S}
end

@inline function LocalGeometry(coordinates, J, WJ, ∂x∂ξ)
    ∂ξ∂x = inv(∂x∂ξ)
    return LocalGeometry(
        coordinates,
        J,
        WJ,
        inv(J),
        ∂x∂ξ,
        ∂ξ∂x,
        ∂ξ∂x * ∂ξ∂x',
        ∂x∂ξ' * ∂x∂ξ,
    )
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

undertype(::Type{LocalGeometry{I, C, FT, S}}) where {I, C, FT, S} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
