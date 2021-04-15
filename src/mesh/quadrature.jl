
module Quadratures

import GaussQuadrature
import StaticArrays: SVector

export QuadratureStyle,
    GLL, GL, polynomial_degree, degrees_of_freedom, quadrature_points

"""
   QuadratureStyle

"""
abstract type QuadratureStyle end

"""
    polynomial_degree(QuadratureStyle) -> Int

Returns the polynomial degree of the `QuadratureStyle` concrete type
"""
function polynomial_degree end


"""
    degrees_of_freedom(QuadratureStyle) -> Int

Returns the degrees_of_freedom of the `QuadratureStyle` concrete type
"""
function degrees_of_freedom end

"""
    points, weights = quadrature_points(::Type{FT}, quadrature_style)

The points and weights of the quadrature rule in floating point type `FT`.
"""
function quadrature_points end


"""
    GLL{Nq}()

Gauss-Legendre-Lobatto quadrature using `Nq` quadrature points.
"""
struct GLL{Nq} <: QuadratureStyle end

polynomial_degree(::GLL{Nq}) where {Nq} = Nq - 1
degrees_of_freedom(::GLL{Nq}) where {Nq} = Nq


function quadrature_points(::Type{FT}, ::GLL{Nq}) where {FT, Nq}
    return GaussQuadrature.legendre(FT, Nq, GaussQuadrature.both)
end



"""
    GL{Nq}()

Gauss-Legendre quadrature using `Nq` quadrature points.
"""
struct GL{Nq} <: QuadratureStyle end

polynomial_degree(::GL{Nq}) where {Nq} = Nq - 1
degrees_of_freedom(::GL{Nq}) where {Nq} = Nq

function quadrature_points(::Type{FT}, ::GL{Nq}) where {FT, Nq}
    return GaussQuadrature.legendre(FT, Nq, GaussQuadrature.neither)
end


end # module
