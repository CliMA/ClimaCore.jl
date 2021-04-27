
"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{FT, M}
    "Jacobian determinant of the transformation `ξ` to `x`"
    J::FT
    "Metric: `J` multiplied by the quadrature weights"
    M::FT
    "inverse Metric terms: `1/M`"
    invM::FT
    "Partial derivatives of the map from `x` to `ξ`: `∂ξ∂x[i,j]` is ∂ξⁱ/∂xʲ"
    ∂ξ∂x::M
end
