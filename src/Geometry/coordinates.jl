
abstract type CartesianPoint{FT} end

components(p::CartesianPoint{FT}) where {FT} =
    SVector(ntuple(i -> getfield(p, i), nfields(p)))

function Base.eltype(::Type{<:CartesianPoint{FT}}) where {FT}
    return FT
end

function Base.:(==)(p1::T, p2::T) where {FT, T <: CartesianPoint{FT}}
    return components(p1) == components(p2)
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {FT, T <: CartesianPoint{FT}}
    return isapprox(components(p1), components(p2); kwargs...)
end

abstract type Abstract1DPoint{FT} <: CartesianPoint{FT} end
struct Cartesian1DPoint{FT} <: Abstract1DPoint{FT}
    x1::FT
end
struct Cartesian3Point{FT} <: Abstract1DPoint{FT}
    x3::FT
end

abstract type Abstract2DPoint{FT} <: CartesianPoint{FT} end
struct Cartesian2DPoint{FT} <: Abstract2DPoint{FT}
    x1::FT
    x2::FT
end

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


"""
    interpolate(coords::NTuple{2}, ξ1)

Interpolate between `coords` by parameters `ξ1` in the interval `[-1,1]`.

The type of interpolation is determined by the element type of `coords`
"""
function interpolate(coords::NTuple{2, V}, ξ1) where {V <: Cartesian3Point}
    Cartesian3Point(
        ((1 - ξ1) / 2 * coords[1].x3) +
         ((1 + ξ1) / 2 * coords[2].x3))
end