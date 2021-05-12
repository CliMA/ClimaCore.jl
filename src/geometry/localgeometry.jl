
"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{FT, M}
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "inverse mass matrix: inverse of the DSSed `WJ` terms"
    invM::FT
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::M
end
