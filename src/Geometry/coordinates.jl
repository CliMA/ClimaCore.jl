abstract type AbstractPoint{FT} end

abstract type Abstract1DPoint{FT} <: AbstractPoint{FT} end
abstract type Abstract2DPoint{FT} <: AbstractPoint{FT} end
abstract type Abstract3DPoint{FT} <: AbstractPoint{FT} end

"""
    @pointtype name fieldname1 ...

Define a subtype `name` of `AbstractPoint` with appropriate conversion functions.
"""
macro pointtype(name, fields...)
    if length(fields) == 1
        supertype = :(Abstract1DPoint{FT})
    elseif length(fields) == 2
        supertype = :(Abstract2DPoint{FT})
    elseif length(fields) == 3
        supertype = :(Abstract3DPoint{FT})
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

@pointtype XYPoint x y
@pointtype XZPoint x z

@pointtype XYZPoint x y z

@pointtype Cartesian1Point x1
@pointtype Cartesian2Point x2
@pointtype Cartesian3Point x3

@pointtype Cartesian12Point x1 x2
@pointtype Cartesian123Point x1 x2 x3
# TODO: deprecate these
const Cartesian2DPoint = Cartesian12Point
const Cartesian3DPoint = Cartesian123Point


product_coordinates(xp::XPoint{FT}, zp::ZPoint{FT}) where {FT} =
    XZPoint{FT}(xp.x, zp.z)
product_coordinates(xp::XPoint{FT}, z::FT) where {FT} = XZPoint{FT}(xp.x, z)

product_coordinates(xyp::XYPoint{FT}, zp::ZPoint{FT}) where {FT} =
    XYZPoint{FT}(xyp.x, xyp.y, zp.z)
product_coordinates(xyp::XYPoint{FT}, z::FT) where {FT} =
    XYZPoint{FT}(xyp.x, xyp.y, z)

# TODO: get rid of these and refactor to consistent point types
ZPoint(pt::Cartesian3Point{FT}) where {FT} = ZPoint{FT}(pt.x3)
XYPoint(pt::Cartesian2DPoint{FT}) where {FT} = XYPoint{FT}(pt.x1, pt.x2)
XYZPoint(pt::Cartesian3DPoint{FT}) where {FT} =
    XYZPoint{FT}(pt.x1, pt.x2, pt.x3)

Cartesian3Point(pt::ZPoint{FT}) where {FT} = Cartesian3Point{FT}(pt.z)
Cartesian2DPoint(pt::XYPoint{FT}) where {FT} = Cartesian2DPoint{FT}(pt.x, pt.y)
Cartesian3DPoint(pt::XYZPoint{FT}) where {FT} =
    Cartesian3DPoint{FT}(pt.x, pt.y, pt.z)

components(p::AbstractPoint) = SVector(ntuple(i -> getfield(p, i), nfields(p)))

function Base.:(==)(p1::T, p2::T) where {T <: AbstractPoint}
    return components(p1) == components(p2)
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {T <: AbstractPoint}
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
