abstract type AbstractPoint{FT} end

abstract type Abstract1DPoint{FT} <: AbstractPoint{FT} end
abstract type Abstract2DPoint{FT} <: AbstractPoint{FT} end

"""
    @pointtype name fieldname1 ...

Define a subtype `name` of `AbstractPoint` with appropriate conversion functions.
"""
macro pointtype(name, fields...)
    if length(fields) == 1
        supertype = :(Abstract1DPoint{FT})
    elseif length(fields) == 2
        supertype = :(Abstract2DPoint{FT})
    end

    esc(quote
        struct $name{FT} <: $supertype
            $([:($field::FT) for field in fields]...)
        end
        Base.eltype(::$name{FT}) where {FT} = FT
        Base.eltype(::Type{$name{FT}}) where {FT} = FT
        unionalltype(::$name{FT}) where {FT} = $name
        unionalltype(::Type{$name{FT}}) where {FT} = $name
    end)
end
@pointtype XPoint x
@pointtype YPoint y
@pointtype ZPoint z

@pointtype XZPoint x z

product_coordinates(xp::XPoint, zp::ZPoint) = XZPoint(xp.x, zp.z)
product_coordinates(xp::XPoint, z::Float64) = XZPoint(xp.x, z)


@pointtype Cartesian3Point x3
@pointtype Cartesian2DPoint x1 x2




components(p::Abstract2DPoint) =
    SVector(ntuple(i -> getfield(p, i), nfields(p)))

function Base.:(==)(p1::T, p2::T) where {T <: Abstract2DPoint}
    return components(p1) == components(p2)
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {T <: Abstract2DPoint}
    return isapprox(components(p1), components(p2); kwargs...)
end

"""
    interpolate(coords::NTuple{4}, ξ1, ξ2)

Interpolate between `coords` by parameters `ξ1, ξ2` in the interval `[-1,1]`.

The type of interpolation is determined by the element type of `coords`:
- `SVector`: use bilinear interpolation
"""
function interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: Abstract2DPoint}
    VV = unionalltype(V)
    VV(
        ((1 - ξ1) * (1 - ξ2)) / 4 * getfield(coords[1], 1) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * getfield(coords[2], 1) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * getfield(coords[3], 1) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * getfield(coords[4], 1),
        ((1 - ξ1) * (1 - ξ2)) / 4 * getfield(coords[1], 2) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * getfield(coords[2], 2) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * getfield(coords[3], 2) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * getfield(coords[4], 2),
    )
end

"""
    interpolate(coords::NTuple{2}, ξ1)
Interpolate between `coords` by parameters `ξ1` in the interval `[-1,1]`.
The type of interpolation is determined by the element type of `coords`
"""
function interpolate(coords::NTuple{2, V}, ξ1) where {V <: Abstract1DPoint}
    VV = unionalltype(V)
    VV(
        (1 - ξ1) / 2 * getfield(coords[1], 1) +
        (1 + ξ1) / 2 * getfield(coords[2], 1),
    )
end
