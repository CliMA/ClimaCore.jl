
"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{C, FT, Mxξ, Mξx}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "Partial derivatives of the map from `ξ` to `x`: `∂x∂ξ[i,j]` is ∂xⁱ/∂ξʲ"
    ∂x∂ξ::Mxξ
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::Mξx
end

const LocalGeometry1D =
    LocalGeometry{C, FT, M} where {C <: Abstract1DPoint{FT}} where {FT, M}
const LocalGeometry2D =
    LocalGeometry{C, FT, M} where {C <: Abstract2DPoint{FT}} where {FT, M}

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

undertype(::Type{LocalGeometry{C, FT, Mxξ, Mξx}}) where {C, FT, Mxξ, Mξx} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
