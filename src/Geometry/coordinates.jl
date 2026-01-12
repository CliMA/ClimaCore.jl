"""
    AbstractPoint

Represents a point in space.

The following types are supported:
- `XPoint(x)`
- `YPoint(y)`
- `ZPoint(z)`
- `XYPoint(x, y)`
- `XZPoint(x, z)`
- `XYZPoint(x, y, z)`
- `LatPoint(lat)`
- `LongPoint(long)`
- `LatLongPoint(lat, long)`
- `LatLongZPoint(lat, long, z)`
- `Cartesian1Point(x1)`
- `Cartesian2Point(x2)`
- `Cartesian3Point(x3)`
- `Cartesian12Point(x1, x2)`
- `Cartesian13Point(x1, x3)`
- `Cartesian123Point(x1, x2, x3)`
"""
abstract type AbstractPoint{FT} end

"""
    float_type(T)

Return the floating point type backing `T`: `T` can either be an object or a type.
"""
float_type(::Type{<:AbstractPoint{FT}}) where {FT} = FT
float_type(::AbstractPoint{FT}) where {FT} = FT

abstract type Abstract1DPoint{FT} <: AbstractPoint{FT} end
abstract type Abstract2DPoint{FT} <: AbstractPoint{FT} end
abstract type Abstract3DPoint{FT} <: AbstractPoint{FT} end

Base.show(io::IO, point::Abstract1DPoint) =
    print(io, nameof(typeof(point)), "(", component(point, 1), ")")
Base.show(io::IO, point::Abstract2DPoint) = print(io,
    nameof(typeof(point)), "(", component(point, 1), ", ", component(point, 2), ")",
)
Base.show(io::IO, point::Abstract3DPoint) = print(io,
    nameof(typeof(point)),
    "(", component(point, 1), ", ", component(point, 2), ", ", component(point, 3), ")",
)

"""
    @pointtype name fieldname1 ...

Define a subtype `name` of `AbstractPoint` with appropriate conversion functions.
"""
macro pointtype(name, fields...)
    if length(fields) == 1
        supertype = :(Abstract1DPoint{FT})
        coord = fields[1]
        tofloat_expr = quote
            tofloat(p::$name) = p.$coord
        end
    elseif length(fields) == 2
        supertype = :(Abstract2DPoint{FT})
        tofloat_expr = :()
    elseif length(fields) == 3
        supertype = :(Abstract3DPoint{FT})
        tofloat_expr = :()
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
            function Base.:(==)(x::$name, y::$name)
                (&)($([:(x.$field == y.$field) for field in fields]...))
            end
            Base.promote_rule(::Type{$name{FT1}}, ::Type{$name{FT2}}) where {FT1, FT2} =
                $name{promote_type(FT1, FT2)}
            $tofloat_expr
        end,
    )
end

@pointtype XPoint x
@pointtype YPoint y
@pointtype ZPoint z
@pointtype PPoint p

@pointtype XYPoint x y
@pointtype XZPoint x z
@pointtype YZPoint y z

@pointtype XYZPoint x y z

@pointtype Cartesian1Point x1
@pointtype Cartesian2Point x2
@pointtype Cartesian3Point x3

@pointtype Cartesian12Point x1 x2
@pointtype Cartesian13Point x1 x3
@pointtype Cartesian123Point x1 x2 x3

Cartesian123Point(pt::Cartesian123Point) = pt
Cartesian123Point(pt::Cartesian1Point{FT}) where {FT} =
    Cartesian123Point(pt.x1, zero(FT), zero(FT))
Cartesian123Point(pt::Cartesian2Point{FT}) where {FT} =
    Cartesian123Point(zero(FT), pt.x2, zero(FT))
Cartesian123Point(pt::Cartesian3Point{FT}) where {FT} =
    Cartesian123Point(zero(FT), zero(FT), pt.x3)
Cartesian123Point(pt::Cartesian12Point{FT}) where {FT} =
    Cartesian123Point(pt.x1, pt.x2, zero(FT))
Cartesian123Point(pt::Cartesian13Point{FT}) where {FT} =
    Cartesian123Point(pt.x1, zero(FT), pt.x3)

@pointtype LatPoint lat
@pointtype LongPoint long
@pointtype LatLongPoint lat long
@pointtype LatLongZPoint lat long z
@pointtype LatLongPPoint lat long p

product_coordinates(xp::XPoint, yp::YPoint) = XYPoint(promote(xp.x, yp.y)...)
product_coordinates(xp::XPoint, zp::ZPoint) = XZPoint(promote(xp.x, zp.z)...)
product_coordinates(yp::YPoint, zp::ZPoint) = YZPoint(promote(yp.y, zp.z)...)

product_coordinates(xyp::XYPoint, zp::ZPoint) = XYZPoint(promote(xyp.x, xyp.y, zp.z)...)

product_coordinates(latp::LatPoint, longp::LongPoint) =
    LatLongPoint(promote(latp.lat, longp.long)...)
product_coordinates(longp::LongPoint, latp::LatPoint) =
    LatLongPoint(promote(latp.lat, longp.long)...)
product_coordinates(latlongp::LatLongPoint, zp::ZPoint) =
    LatLongZPoint(promote(latlongp.lat, latlongp.long, zp.z)...)
product_coordinates(latlongp::LatLongPoint, pressure_p::PPoint) =
    LatLongPPoint(promote(latlongp.lat, latlongp.long, pressure_p.p)...)

component(p::AbstractPoint{FT}, i::Symbol) where {FT} = getfield(p, i)::FT
component(p::AbstractPoint{FT}, i::Integer) where {FT} = getfield(p, i)::FT

@inline ncomponents(p::AbstractPoint) = nfields(p)
@inline ncomponents(::Type{P}) where {P <: AbstractPoint} = fieldcount(P)
components(p::AbstractPoint) = SVector(ntuple(i -> component(p, i), ncomponents(p)))

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

_coordinate_type(::Type{YZPoint{FT}}, ::Val{1}) where {FT} = YPoint{FT}
_coordinate_type(::Type{YZPoint{FT}}, ::Val{2}) where {FT} = ZPoint{FT}
_coordinate(p::YZPoint, ::Val{1}) = YPoint(p.x)
_coordinate(p::YZPoint, ::Val{2}) = ZPoint(p.z)

_coordinate_type(::Type{XYZPoint{FT}}, ::Val{1}) where {FT} = XPoint{FT}
_coordinate_type(::Type{XYZPoint{FT}}, ::Val{2}) where {FT} = YPoint{FT}
_coordinate_type(::Type{XYZPoint{FT}}, ::Val{3}) where {FT} = ZPoint{FT}
_coordinate(p::XYZPoint, ::Val{1}) = XPoint(p.x)
_coordinate(p::XYZPoint, ::Val{2}) = YPoint(p.y)
_coordinate(p::XYZPoint, ::Val{3}) = ZPoint(p.z)

coordinate_type(ptyp::Type{<:AbstractPoint}, ax::Int) = _coordinate_type(ptyp, Val(ax))
coordinate_type(ptyp::Type{<:AbstractPoint}, ax::Integer) = _coordinate_type(ptyp, Int(ax))

coordinate(pt::AbstractPoint, ax::Int) = _coordinate(pt, Val(ax))
coordinate(pt::AbstractPoint, ax::Integer) = _coordinate(pt, Int(ax))

# the following are needed for linranges to work correctly with coordinate values
Base.:(-)(p1::T) where {T <: AbstractPoint} = unionalltype(T)((-components(p1)...))
Base.:(-)(p1::T, p2::T) where {T <: AbstractPoint} =
    unionalltype(T)((components(p1) - components(p2))...)
Base.:(+)(p1::T, p2::T) where {T <: AbstractPoint} =
    unionalltype(T)((components(p1) + components(p2))...)
Base.:(*)(p::T, x::Number) where {T <: AbstractPoint} =
    unionalltype(T)((components(p) * x)...)
Base.:(*)(x::Number, p::AbstractPoint) = p * x
Base.:(/)(p::T, x::Number) where {T <: AbstractPoint} =
    unionalltype(T)((components(p) / x)...)

Base.LinRange(start::T, stop::T, length::Integer) where {T <: Abstract1DPoint} =
    Base.LinRange{T}(start, stop, length)

# we add our own method to this so that `BigFloat` coordinate ranges are computed accurately.
function Base.lerpi(j::Integer, d::Integer, a::T, b::T) where {T <: Abstract1DPoint}
    T(Base.lerpi(j, d, component(a, 1), component(b, 1)))
end

function Base.isapprox(p1::T, p2::T; kwargs...) where {T <: AbstractPoint}
    return isapprox(components(p1), components(p2); kwargs...)
end
Base.isless(x::T, y::T) where {T <: Abstract1DPoint} =
    isless(component(x, 1), component(y, 1))
Base.isless(x::Abstract1DPoint, y::Abstract1DPoint) = isless(promote(x, y)...)


"""
    bilinear_interpolate(coords::NTuple{4}, ξ1, ξ2)

Bilinear interpolate between `coords` by parameters `ξ1, ξ2`, each in the interval `[-1,1]`.

`coords` should be a 4-tuple of coordinates in counter-clockwise order.

```
      4-------3
 ^    |       |
 |    |       |
ξ2    |       |
      1-------2

        ξ1-->
```
"""
function bilinear_interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: SVector}
    w1 = (1 - ξ1) * (1 - ξ2) / 4
    w2 = (1 + ξ1) * (1 - ξ2) / 4
    w3 = (1 + ξ1) * (1 + ξ2) / 4
    w4 = (1 - ξ1) * (1 + ξ2) / 4
    w1 .* coords[1] .+ w2 .* coords[2] .+ w3 .* coords[3] .+ w4 .* coords[4]
end

function bilinear_interpolate(coords::NTuple{4, V}, ξ1, ξ2) where {V <: AbstractPoint}
    VV = unionalltype(V)
    c = bilinear_interpolate(map(components, coords), ξ1, ξ2)
    VV(c...)
end


"""
    bilinear_invert(cc::NTuple{4, V}) where {V<:SVector{2}}

Solve find `ξ1` and `ξ2` such that

    bilinear_interpolate(coords, ξ1, ξ2) == 0

See also [`bilinear_interpolate`](@ref).
"""
function bilinear_invert(vert_coords::NTuple{4, V}) where {V <: SVector{2}}
    # 1) express as 2 equations
    #  ca' * M * [1,ξ1,ξ2,ξ1*ξ2] == 0
    #  cb' * M * [1,ξ1,ξ2,ξ1*ξ2] == 0
    ca = SVector(map(v -> v[1], vert_coords))
    cb = SVector(map(v -> v[2], vert_coords))
    M = @SMatrix [
        1 -1 -1 1
        1 1 -1 -1
        1 1 1 1
        1 -1 1 -1
    ]

    # 2) get into the form
    #  a1 + a2 * ξ1 + a3 * ξ2 + a4 * ξ1 * ξ2 == 0  (A)
    #  b1 + b2 * ξ1 + b3 * ξ2 + b4 * ξ1 * ξ2 == 0  (B)
    (a1, a2, a3, a4) = M' * ca
    (b1, b2, b3, b4) = M' * cb

    # 3) rearrange (A):
    #   -(a1 + a3*ξ2) = ξ1 * (a2 + a4*ξ2)
    # 4) multiply (B) by (a2 + a4*ξ2), and eliminate ξ1
    #    b1 * (a2 + a4*ξ2) - b2 * (a1 + a3*ξ2) + b3 * (a2 + a4*ξ2) * ξ2 - b4 * (a1 + a3*ξ2) * ξ2 == 0
    # 5) collect coefficients of powers of ξ2
    c0 = b1 * a2 - b2 * a1
    c1 = b1 * a4 - b2 * a3 + b3 * a2 - b4 * a1
    c2 = b3 * a4 - b4 * a3
    # 6) solve quadratic equation
    Δ = c1 * c1 - 4 * c2 * c0
    if c1 >= 0
        ξ2a = (-c1 - sqrt(Δ)) / (2 * c2)
        ξ2b = 2 * c0 / (-c1 - sqrt(Δ))
    else
        ξ2a = 2 * c0 / (-c1 + sqrt(Δ))
        ξ2b = (-c1 + sqrt(Δ)) / (2 * c2)
    end
    # 7) find which solution is smallest in magnitude (closest to the interval [-1,1]):
    ξ2 = abs(ξ2a) < abs(ξ2b) ? ξ2a : ξ2b
    # 8) solve for ξ1
    ξ1 = -(a1 + a3 * ξ2) / (a2 + a4 * ξ2)
    return SVector(ξ1, ξ2)
end


"""
    interpolate(coords::NTuple{2}, ξ1)

Interpolate between `coords` by parameters `ξ1` in the interval `[-1,1]`.
The type of interpolation is determined by the element type of `coords`
"""
function linear_interpolate(
    coords::NTuple{2, V},
    ξ1,
) where {V <: Abstract1DPoint}
    VV = unionalltype(V)
    VV(
        (1 - ξ1) / 2 * component(coords[1], 1) +
        (1 + ξ1) / 2 * component(coords[2], 1),
    )
end

function spherical_bilinear_interpolate((x1, x2, x3, x4), ξ1, ξ2, radius)
    x = bilinear_interpolate((x1, x2, x3, x4), ξ1, ξ2)
    return x * (radius / norm(components(x)))
end
