
module Quadratures

import GaussQuadrature
import StaticArrays: SVector, SMatrix, MMatrix
import LinearAlgebra: Diagonal

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
    Uniform{Nq}()

Uniformly-spaced quadrature.
"""
struct Uniform{Nq} <: QuadratureStyle end

@generated function quadrature_points(::Type{FT}, ::Uniform{Nq}) where {FT, Nq}
    points = SVector{Nq}(range(-1 + 1 / Nq, step = 2 / Nq, length = Nq))
    weights = SVector{Nq}(ntuple(i -> 2 / Nq, Nq))
    :($points, $weights)
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

function interpolation_matrix(
    points_to::SVector{Nto},
    points_from::SVector{Nfrom},
) where {Nto, Nfrom}
    T = eltype(points_to)
    bw = barycentric_weights(points_from)
    M = zeros(MMatrix{Nto, Nfrom, T, Nto * Nfrom})
    for i in 1:Nto
        x_to = points_to[i]
        skip_row = false
        for j in 1:Nfrom
            if x_to == points_from[j]
                # assign to one to avoid singularity condition
                M[i, j] = one(T)
                # skip over the equal boundry condition
                skip_row = true
            end
            skip_row && break
        end
        skip_row && continue
        w = bw ./ (x_to .- points_from)
        M[i, :] .= w ./ sum(w)
    end
    return SMatrix(M)
end

@generated function interpolation_matrix(
    ::Type{FT},
    quadto::QuadratureStyle,
    quadfrom::QuadratureStyle,
) where {FT}
    interpolation_matrix(
        quadrature_points(FT, quadto())[1],
        quadrature_points(FT, quadfrom())[1],
    )
end

"""
    V = orthonormal_poly(points, quad)

`V[i,j]` contains the `j-1`th Legendre polynomial evaluated at `points[i]`.
i.e. it is the mapping from the modal to the nodal representation.
"""
function orthonormal_poly(
    points::SVector{Np, FT},
    quad::GLL{Nq},
) where {FT, Np, Nq}
    N = Nq - 1
    a, b = GaussQuadrature.legendre_coefs(FT, N)
    if N == 0
        return SMatrix{Np, 1}(ntuple(x -> b[1], Np))
    end
    return SMatrix{Np, Nq}(GaussQuadrature.orthonormal_poly(points, a, b))
end

function spectral_filter_matrix(
    quad::GLL{Nq},
    Σ::SVector{Nq, FT},
) where {Nq, FT}
    points, _ = quadrature_points(FT, quad)
    V = orthonormal_poly(points, quad)
    return V * Diagonal(Σ) / V
end

function cutoff_filter_matrix(
    ::Type{FT},
    quad::GLL{Nq},
    Nc::Integer,
) where {FT, Nq}
    Σ = SVector(ntuple(i -> i <= Nc ? FT(1) : FT(0), Nq))
    return spectral_filter_matrix(quad, Σ)
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
