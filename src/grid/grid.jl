"""
    Grids

- domain
- topology
- coordinates
- metric terms (inverse partial derivatives)
- quadrature rules and weights

https://ceed.exascaleproject.org/ceed-code/

QA: https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/sphere/sphere_helper_functions.jl

"""
module Grids



abstract type AbstractGrid end

abstract type HorizontalGrid end

# some notion of a local geometry at each node, we need
#  - coordinates
#  - mechanism to convert vectors to contravariant (for divergences) and from covariant (for gradients)
#  - Jacobian determinant + inverse (Riemannian volume form on reference element)



struct LocalGeometry{FT}
    x::SVector{3, FT}
    ∂ξ∂x::SMatrix{3, 3, FT, 9}
    J::FT
end




end # module
