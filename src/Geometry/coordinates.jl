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
    esc(
        quote
            struct $name{FT} <: $supertype
                $([:($field::FT) for field in fields]...)
            end
            Base.convert(::Type{$name{FT}}, pt2::$name) where {FT} =
                $name{FT}($((:(pt2.$field) for field in fields)...))
            $name{FT}(pt::$name) where {FT} = convert($name{FT}, pt)
            Base.eltype(::$name{FT}) where {FT} = FT
            Base.eltype(::Type{$name{FT}}) where {FT} = FT
            unionalltype(::$name{FT}) where {FT} = $name
            unionalltype(::Type{$name{FT}}) where {FT} = $name
            Base.promote_rule(
                ::Type{$name{FT1}},
                ::Type{$name{FT2}},
            ) where {FT1, FT2} = $name{promote_type(FT1, FT2)}
        end,
    )
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

@pointtype LatLongPoint lat long


const Cartesian2DPoint = Cartesian12Point
@deprecate Cartesian2DPoint Cartesian12Point

const Cartesian3DPoint = Cartesian123Point
@deprecate Cartesian3DPoint Cartesian123Point

product_coordinates(xp::XPoint, yp::YPoint) = XYPoint(promote(xp.x, yp.y)...)
product_coordinates(xp::XPoint, zp::ZPoint) = XZPoint(promote(xp.x, zp.z)...)

product_coordinates(xyp::XYPoint, zp::ZPoint) =
    XYZPoint(promote(xyp.x, xyp.y, zp.z)...)

# TODO: get rid of these and refactor to consistent point types
ZPoint(pt::Cartesian3Point{FT}) where {FT} = ZPoint{FT}(pt.x3)
XYPoint(pt::Cartesian12Point{FT}) where {FT} = XYPoint{FT}(pt.x1, pt.x2)
XYZPoint(pt::Cartesian123Point{FT}) where {FT} =
    XYZPoint{FT}(pt.x1, pt.x2, pt.x3)

Cartesian3Point(pt::ZPoint{FT}) where {FT} = Cartesian3Point{FT}(pt.z)
Cartesian12Point(pt::XYPoint{FT}) where {FT} = Cartesian12Point{FT}(pt.x, pt.y)
Cartesian123Point(pt::XYZPoint{FT}) where {FT} =
    Cartesian123Point{FT}(pt.x, pt.y, pt.z)

component(p::AbstractPoint{FT}, i::Symbol) where {FT} = getfield(p, i)::FT
component(p::AbstractPoint{FT}, i::Integer) where {FT} = getfield(p, i)::FT

ncomponents(p::AbstractPoint) = nfields(p)
components(p::AbstractPoint) =
    SVector(ntuple(i -> component(p, i), ncomponents(p)))

_coordinate_type(ptyp::Type{Abstract1DPoint}, ::Val{1}) = ptyp
_coordinate(p::Abstract1DPoint, ::Val{1}) = p

_coordinate_type(::Type{XYPoint{FT}}, ::Val{1}) where {FT} = XPoint{FT}
_coordinate_type(::Type{XYPoint{FT}}, ::Val{2}) where {FT} = YPoint{FT}
_coordinate(p::XYPoint, ::Val{1}) = XPoint(p.x)
_coordinate(p::XYPoint, ::Val{2}) = YPoint(p.y)

_coordinate_type(::Type{XZPoint{FT}}, ::Val{1}) where {FT} = XPoint{FT}
_coordinate_type(::Type{XZPoint{FT}}, ::Val{2}) where {FT} = ZPoint{FT}
_coordinate(p::XZPoint, ::Val{1}) = XPoint(p.x)
_coordinate(p::XZPoint, ::Val{2}) = ZPoint(p.z)

_coordinate_type(::Type{XYZPoint{FT}}, ::Val{1}) where {FT} = XPoint{FT}
_coordinate_type(::Type{XYZPoint{FT}}, ::Val{2}) where {FT} = YPoint{FT}
_coordinate_type(::Type{XYZPoint{FT}}, ::Val{3}) where {FT} = ZPoint{FT}
_coordinate(p::XYZPoint, ::Val{1}) = XPoint(p.x)
_coordinate(p::XYZPoint, ::Val{2}) = YPoint(p.y)
_coordinate(p::XYZPoint, ::Val{3}) = ZPoint(p.z)

coordinate_type(ptyp::Type{<:AbstractPoint}, ax::Int) =
    _coordinate_type(ptyp, Val(ax))
coordinate_type(ptyp::Type{<:AbstractPoint}, ax::Integer) =
    _coordinate_type(ptyp, Int(ax))

coordinate(pt::AbstractPoint, ax::Int) = _coordinate(pt, Val(ax))
coordinate(pt::AbstractPoint, ax::Integer) = _coordinate(pt, Int(ax))

# TODO: we need to rationalize point operations with vectors
#Base.:(-)(p1::T, p2::T) where {T <: AbstractPoint} =
#    unionalltype(T)((components(p1) - components(p2))...)

# the following are needed for linranges to work correctly with coordinate values
Base.:(+)(p1::T, p2::T) where {T <: AbstractPoint} =
    unionalltype(T)((components(p1) + components(p2))...)
Base.:(*)(p::T, x::Number) where {T <: AbstractPoint} =
    unionalltype(T)((components(p) * x)...)
Base.:(*)(x::Number, p::AbstractPoint) = p * x
Base.:(/)(p::T, x::Number) where {T <: AbstractPoint} =
    unionalltype(T)((components(p) / x)...)

function Base.:(==)(p1::T, p2::T) where {T <: AbstractPoint}
    return components(p1) == components(p2)
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {T <: AbstractPoint}
    return isapprox(components(p1), components(p2); kwargs...)
end

LinearAlgebra.norm(pt::Cartesian2DPoint, p::Real = 2) =
    LinearAlgebra.norm((pt.x1, pt.x2), p)
LinearAlgebra.norm(pt::Cartesian3DPoint, p::Real = 2) =
    LinearAlgebra.norm((pt.x1, pt.x2, pt.x3), p)


function LatLongPoint(pt::Cartesian123Point)
    lat = atand(pt.x3, hypot(pt.x2, pt.x1))
    long = atand(pt.x2, pt.x1)
    LatLongPoint(lat, long)
end

function Cartesian123Point(pt::LatLongPoint)
    x1 = cosd(pt.long) * cosd(pt.lat)
    x2 = sind(pt.long) * cosd(pt.lat)
    x3 = sind(pt.lat)
    Cartesian123Point(x1, x2, x3)
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
        ((1 - ξ1) * (1 - ξ2)) / 4 * component(coords[1], 1) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * component(coords[2], 1) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * component(coords[3], 1) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * component(coords[4], 1),
        ((1 - ξ1) * (1 - ξ2)) / 4 * component(coords[1], 2) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * component(coords[2], 2) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * component(coords[3], 2) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * component(coords[4], 2),
    )
end

function interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: Abstract3DPoint}
    VV = unionalltype(V)
    VV(
        ((1 - ξ1) * (1 - ξ2)) / 4 * component(coords[1], 1) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * component(coords[2], 1) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * component(coords[3], 1) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * component(coords[4], 1),
        ((1 - ξ1) * (1 - ξ2)) / 4 * component(coords[1], 2) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * component(coords[2], 2) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * component(coords[3], 2) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * component(coords[4], 2),
        ((1 - ξ1) * (1 - ξ2)) / 4 * component(coords[1], 3) +
        ((1 + ξ1) * (1 - ξ2)) / 4 * component(coords[2], 3) +
        ((1 - ξ1) * (1 + ξ2)) / 4 * component(coords[3], 3) +
        ((1 + ξ1) * (1 + ξ2)) / 4 * component(coords[4], 3),
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
        (1 - ξ1) / 2 * component(coords[1], 1) +
        (1 + ξ1) / 2 * component(coords[2], 1),
    )
end

function spherical_bilinear_interpolate((x1, x2, x3, x4), ξ1, ξ2)
    r = norm(x1) # assume all are same radius        
    x = Geometry.interpolate((x1, x2, x3, x4), ξ1, ξ2)
    x = x * (r / norm(x))
end
