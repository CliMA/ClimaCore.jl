
module Quadratures

import GaussQuadrature
import StaticArrays: SVector, SMatrix, MMatrix
import LinearAlgebra: Diagonal

export QuadratureStyle,
    GLL, GL, polynomial_degree, degrees_of_freedom, quadrature_points

"""
    QuadratureStyle{Nq}

Supertype for quadrature rules with `Nq` points on the reference interval `[-1, 1]`.

Subtypes:

  - [`GLL`](@ref): Gauss-Legendre-Lobatto quadrature, which includes the endpoints.
  - [`GL`](@ref): Gauss-Legendre quadrature.
  - [`Uniform`](@ref): uniformly spaced midpoint quadrature.
  - `ClosedUniform`: uniformly spaced quadrature including the endpoints.

Subtypes implement [`quadrature_points`](@ref) and `unique_degrees_of_freedom`.
"""
abstract type QuadratureStyle{Nq} end

"""
    polynomial_degree(quadstyle::QuadratureStyle) -> Int

Return the polynomial degree `Nq - 1` of the quadrature rule `quadstyle`.
"""
@inline polynomial_degree(::QuadratureStyle{Nq}) where {Nq} = Nq - 1


"""
    degrees_of_freedom(quadstyle::QuadratureStyle) -> Int

Return the number of quadrature points `Nq` of the quadrature rule `quadstyle`.
"""
@inline degrees_of_freedom(::QuadratureStyle{Nq}) where {Nq} = Nq

"""
    requires_dss(quadstyle::QuadratureStyle) -> Bool

Return whether `quadstyle` requires direct stiffness summation, i.e. whether its nodes
are shared between neighboring elements.
"""
requires_dss(quadstyle) =
    unique_degrees_of_freedom(quadstyle) < degrees_of_freedom(quadstyle)

"""
    quadrature_points(::Type{FT}, quadstyle::QuadratureStyle) -> (points, weights)

Return the points and weights of the quadrature rule `quadstyle` on `[-1, 1]` as a tuple
of two `SVector`s with element type `FT`.
"""
function quadrature_points end


"""
    GLL{Nq}()

Gauss-Legendre-Lobatto quadrature using `Nq` quadrature points.
"""
struct GLL{Nq} <: QuadratureStyle{Nq} end

Base.show(io::IO, ::GLL{Nq}) where {Nq} =
    print(io, Nq, "-point Gauss-Legendre-Lobatto quadrature")

unique_degrees_of_freedom(::GLL{Nq}) where {Nq} = Nq - 1
@generated function quadrature_points(::Type{FT}, ::GLL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.both)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    GL{Nq}()

Gauss-Legendre quadrature using `Nq` quadrature points.
"""
struct GL{Nq} <: QuadratureStyle{Nq} end

Base.show(io::IO, ::GL{Nq}) where {Nq} =
    print(io, Nq, "-point Gauss-Legendre quadrature")

unique_degrees_of_freedom(::GL{Nq}) where {Nq} = Nq
@generated function quadrature_points(::Type{FT}, ::GL{Nq}) where {FT, Nq}
    points, weights = GaussQuadrature.legendre(FT, Nq, GaussQuadrature.neither)
    :($(SVector{Nq}(points)), $(SVector{Nq}(weights)))
end

"""
    Uniform{Nq}()

Uniformly spaced midpoint quadrature with `Nq` points; the endpoints of `[-1, 1]` are not
included.
"""
struct Uniform{Nq} <: QuadratureStyle{Nq} end

unique_degrees_of_freedom(::Uniform{Nq}) where {Nq} = Nq
@generated function quadrature_points(::Type{FT}, ::Uniform{Nq}) where {FT, Nq}
    points = SVector{Nq}(range(-1 + FT(1 / Nq), step = FT(2 / Nq), length = Nq))
    weights = SVector{Nq}(ntuple(i -> FT(2 / Nq), Nq))
    :($points, $weights)
end

"""
    ClosedUniform{Nq}()

Uniformly spaced trapezoidal quadrature with `Nq` points, including the endpoints of
`[-1, 1]`.
"""
struct ClosedUniform{Nq} <: QuadratureStyle{Nq} end

unique_degrees_of_freedom(::ClosedUniform{Nq}) where {Nq} = Nq - 1
@generated function quadrature_points(
    ::Type{FT},
    ::ClosedUniform{Nq},
) where {FT, Nq}
    points = SVector{Nq}(range(FT(-1), FT(1), length = Nq))
    weights = SVector{Nq}(
        FT(1 / (Nq - 1)),
        ntuple(i -> FT(2 / (Nq - 1)), Nq - 2)...,
        FT(1 / (Nq - 1)),
    )
    :($points, $weights)
end


"""
    barycentric_weights(x::SVector{Nq})
    barycentric_weights(::Type{FT}, quadstyle::QuadratureStyle)

Return the barycentric weights associated with the point locations `x`, or with the
quadrature points of `quadstyle` in float type `FT`:

```math
w_j = \\frac{1}{\\prod_{k \\ne j} (x_k - x_j)}
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
    interpolation_matrix(x::Vector, r)
    interpolation_matrix(::Type{FT}, quadto::QuadratureStyle, quadfrom::QuadratureStyle)

Return the matrix that interpolates the Lagrange polynomial of degree `Nq - 1` through
the points `r` to the points `x`. The third method uses the quadrature points of
`quadfrom` and `quadto` in float type `FT`. The matrix coefficients are computed with the
barycentric formula of [Berrut2004](@cite), section 4:

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
                # skip over the equal boundary condition
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

function interpolation_matrix(points_to::Vector, points_from)
    T = eltype(points_to)
    bw = barycentric_weights(points_from)
    M = zeros(T, length(points_to), length(points_from))
    for i in 1:length(points_to)
        x_to = points_to[i]
        skip_row = false
        for j in 1:length(points_from)
            if x_to == points_from[j]
                # assign to one to avoid singularity condition
                M[i, j] = one(T)
                # skip over the equal boundary condition
                skip_row = true
            end
            skip_row && break
        end
        skip_row && continue
        w = bw ./ (x_to .- points_from)
        M[i, :] .= w ./ sum(w)
    end
    return M
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
    orthonormal_poly(points::SVector, quad::GLL)

Return the matrix `V` whose entry `V[i, j]` is the orthonormal Legendre polynomial of
degree `j - 1` evaluated at `points[i]`, i.e. the map from the modal to the nodal
representation for the polynomial space of `quad`.
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
    differentiation_matrix(r::SVector{Nq, T})

Return the spectral differentiation matrix for the Lagrange polynomial of degree `Nq - 1`
interpolating at the points `r`.

The matrix coefficients are computed following [Berrut2004](@cite), section 9.3:

```math
D_{ij} = \\begin{cases}
    \\displaystyle
    \\frac{w_j}{w_i (x_i - x_j)} &\\text{ if } i \\ne j, \\\\
    \\displaystyle
    \\sum_{k \\ne i} \\frac{1}{x_i - x_k} &\\text{ if } i = j,
\\end{cases}
```

where ``w_j`` are the barycentric weights, see [`barycentric_weights`](@ref). The rows
of ``D`` sum to zero.
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
    differentiation_matrix(::Type{FT}, quadstyle::QuadratureStyle)

Return the spectral differentiation matrix at the quadrature points of `quadstyle`, in
float type `FT`.
"""
@generated function differentiation_matrix(
    ::Type{FT},
    quadstyle::QuadratureStyle,
) where {FT}
    differentiation_matrix(quadrature_points(FT, quadstyle())[1])
end


end # module
