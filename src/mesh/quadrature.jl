
module Quadratures

import GaussQuadrature
import StaticArrays: SVector, SMatrix

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

@generated function quadrature_points(::Type{FT}, ::GLL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.both)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    GL{Nq}()

Gauss-Legendre quadrature using `Nq` quadrature points.
"""
struct GL{Nq} <: QuadratureStyle end

polynomial_degree(::GL{Nq}) where {Nq} = Nq - 1
degrees_of_freedom(::GL{Nq}) where {Nq} = Nq


@generated function quadrature_points(::Type{FT}, ::GL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.neither)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    barycentric_weights(x::AbstractVector)

The barycentric weights associated with the array of point locations `x`:

```math
w[i] = \\frac{1}{\\prod_{j \\ne i} (x[i] - x[j])
```
Reference:
  - [Berrut2004](@cite) equation 3.2
"""
function barycentric_weights(r::SVector{Nq, T}) where {Nq, T}
    SVector{Nq}(ntuple(Nq) do i
        w = one(T)
        for j in 1:Nq
            if j != i
                w *= (r[j] - r[i])
            end
        end
        inv(w)
    end)
end
@generated function barycentric_weights(
    ::Type{FT},
    quadstyle::QuadratureStyle,
) where {FT}
    barycentric_weights(quadrature_points(FT, quadstyle())[1])
end


"""
    differentiation_matrix(r::SVector{Nq, T},
                           wb=barycentric_weights(r)::SVector{Nq,T}) where {Nq, T}

The spectral differentiation matrix for a polynomial of degree `Nq` - 1 defined on the
points `r` with associated barycentric weights `wb`.

```math
D_{i,j} = \\begin{cases}
    \\frac{w_j / w_i}{x_i - x_j} \\text{ if } i \\ne j \\\\
    -\\sum_{k \\ne j} D_{k,j} \\text{ if } i = j
\\end{cases}
```

Reference:
 - [Berrut2004](@cite),#https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
"""
function differentiation_matrix(
    r::SVector{Nq, T},
    wb = barycentric_weights(r)::SVector{Nq, T},
) where {Nq, T}
    SMatrix{Nq, Nq, T, Nq * Nq}(
        begin
            if i == j
                D = zero(T)
                for l in 1:Nq
                    if l != i
                        D += one(T) / (r[i] - r[l])
                    end
                end
                D
            else
                (wb[i] / wb[j]) / (r[j] - r[i])
            end
        end for j in 1:Nq, i in 1:Nq
    )
end
@generated function differentiation_matrix(
    ::Type{FT},
    quadstyle::QuadratureStyle,
) where {FT}
    differentiation_matrix(quadrature_points(FT, quadstyle())[1])
end




end # module
