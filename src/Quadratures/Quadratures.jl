
module Quadratures

import GaussQuadrature
import StaticArrays: SVector, SMatrix, MMatrix
import LinearAlgebra: Diagonal

export QuadratureStyle,
    GLL, GL, polynomial_degree, degrees_of_freedom, quadrature_points

"""
   QuadratureStyle

Quadrature style supertype. See sub-types:
 - [`GLL`](@ref)
 - [`GL`](@ref)
 - [`Uniform`](@ref)
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

Base.show(io::IO, ::GLL{Nq}) where {Nq} =
    print(io, Nq, "-point Gauss-Legendre-Lobatto quadrature")

@inline polynomial_degree(::GLL{Nq}) where {Nq} = Int(Nq - 1)
@inline degrees_of_freedom(::GLL{Nq}) where {Nq} = Int(Nq)
unique_degrees_of_freedom(::GLL{Nq}) where {Nq} = Int(Nq - 1)

@generated function quadrature_points(::Type{FT}, ::GLL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.both)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    GL{Nq}()

Gauss-Legendre quadrature using `Nq` quadrature points.
"""
struct GL{Nq} <: QuadratureStyle end
Base.show(io::IO, ::GL{Nq}) where {Nq} =
    print(io, Nq, "-point Gauss-Legendre quadrature")

@inline polynomial_degree(::GL{Nq}) where {Nq} = Int(Nq - 1)
@inline degrees_of_freedom(::GL{Nq}) where {Nq} = Int(Nq)
unique_degrees_of_freedom(::GL{Nq}) where {Nq} = Int(Nq)

@generated function quadrature_points(::Type{FT}, ::GL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.neither)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    Uniform{Nq}()

Uniformly-spaced quadrature.
"""
struct Uniform{Nq} <: QuadratureStyle end

@inline polynomial_degree(::Uniform{Nq}) where {Nq} = Int(Nq - 1)
@inline degrees_of_freedom(::Uniform{Nq}) where {Nq} = Int(Nq)

@generated function quadrature_points(::Type{FT}, ::Uniform{Nq}) where {FT, Nq}
    points = SVector{Nq}(range(-1 + 1 / Nq, step = 2 / Nq, length = Nq))
    weights = SVector{Nq}(ntuple(i -> 2 / Nq, Nq))
    :($points, $weights)
end

"""
    ClosedUniform{Nq}()

Uniformly-spaced quadrature including boundary.
"""
struct ClosedUniform{Nq} <: QuadratureStyle end

@inline polynomial_degree(::ClosedUniform{Nq}) where {Nq} = Int(Nq - 1)
@inline degrees_of_freedom(::ClosedUniform{Nq}) where {Nq} = Int(Nq)

@generated function quadrature_points(
    ::Type{FT},
    ::ClosedUniform{Nq},
) where {FT, Nq}
    points = SVector{Nq}(range(FT(-1), FT(1), length = Nq))
    weights = SVector{Nq}(
        1 / (Nq - 1),
        ntuple(i -> 2 / (Nq - 1), Nq - 2)...,
        1 / (Nq - 1),
    )
    :($points, $weights)
end


"""
    barycentric_weights(x::SVector{Nq}) where {Nq}

The barycentric weights associated with the array of point locations `x`:

```math
w_j = \\frac{1}{\\prod_{k \\ne j} (x_i - x_j)}
```

See [Berrut2004](@cite), equation 3.2.
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
    interpolation_matrix(x::SVector, r::SVector{Nq})

The matrix which interpolates the Lagrange polynomial of degree `Nq-1` through
the points `r`, to points `x`. The matrix coefficients are computed using the
Barycentric formula of [Berrut2004](@cite), section 4:
```math
I_{ij} = \\begin{cases}
1 & \\text{if } x_i = r_j, \\\\
0 & \\text{if } x_i = r_k \\text{ for } k \\ne j, \\\\
\\frac{\\displaystyle \\frac{w_j}{x_i - r_j}}{\\displaystyle \\sum_k \\frac{w_k}{x_i - r_k}} & \\text{otherwise,}
\\end{cases}
```
where ``w_j`` are the barycentric weights, see [`barycentric_weights`](@ref).
"""
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

`V_{ij}` contains the `j-1`th Legendre polynomial evaluated at `points[i]`.
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
    differentiation_matrix(r::SVector{Nq, T}) where {Nq, T}

The spectral differentiation matrix for the Lagrange polynomial of degree `Nq-1`
interpolating at points `r`.

The matrix coefficients are computed using the [Berrut2004](@cite), section 9.3:
```math
D_{ij} = \\begin{cases}
    \\displaystyle
    \\frac{w_j}{w_i (x_i - x_j)} &\\text{ if } i \\ne j \\\\
    -\\sum_{k \\ne j} D_{kj} &\\text{ if } i = j
\\end{cases}
```
where ``w_j`` are the barycentric weights, see [`barycentric_weights`](@ref).
"""
function differentiation_matrix(r::SVector{Nq, T}) where {Nq, T}
    wb = barycentric_weights(r)
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

"""
    differentiation_matrix(FT, quadstyle::QuadratureStyle)

The spectral differentiation matrix at the quadrature points of `quadstyle`,
using floating point types `FT`.
"""
@generated function differentiation_matrix(
    ::Type{FT},
    quadstyle::QuadratureStyle,
) where {FT}
    differentiation_matrix(quadrature_points(FT, quadstyle())[1])
end


end # module
