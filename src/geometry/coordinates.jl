abstract type Abstract2DPoint end

components(p::Abstract2DPoint) =
    SVector(ntuple(i -> getfield(p, i), nfields(p)))

function Base.:(==)(p1::T, p2::T) where {T <: Abstract2DPoint}
    return components(p1) == components(p2)
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {T <: Abstract2DPoint}
    return isapprox(components(p1), components(p2); kwargs...)
end

struct Cartesian2DPoint{FT} <: Abstract2DPoint
    x1::FT
    x2::FT
end

Base.eltype(::Type{Cartesian2DPoint{FT}}) where {FT} = FT

"""
    interpolate(coords::NTuple{4}, ξ1, ξ2)

Interpolate between `coords` by parameters `ξ1, ξ2` in the interval `[-1,1]`.

The type of interpolation is determined by the element type of `coords`:
- `SVector`: use bilinear interpolation
"""
function interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: Cartesian2DPoint}
    Cartesian2DPoint(
        ((1 - ξ1) * (1 - ξ2)) / 4 * coords[1].x1 +
        ((1 + ξ1) * (1 - ξ2)) / 4 * coords[2].x1 +
        ((1 - ξ1) * (1 + ξ2)) / 4 * coords[3].x1 +
        ((1 + ξ1) * (1 + ξ2)) / 4 * coords[4].x1,
        ((1 - ξ1) * (1 - ξ2)) / 4 * coords[1].x2 +
        ((1 + ξ1) * (1 - ξ2)) / 4 * coords[2].x2 +
        ((1 - ξ1) * (1 + ξ2)) / 4 * coords[3].x2 +
        ((1 + ξ1) * (1 + ξ2)) / 4 * coords[4].x2,
    )
end
