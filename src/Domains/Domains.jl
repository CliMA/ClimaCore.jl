module Domains

import ..Geometry
using IntervalSets
export RectangleDomain


abstract type AbstractDomain end

abstract type HorizontalDomain <: AbstractDomain end
abstract type VerticalDomain <: AbstractDomain end

const BCTagType = Union{Nothing, Tuple{Symbol, Symbol}}

struct IntervalDomain{FT, B} <:
       VerticalDomain where {FT <: AbstractFloat, B <: BCTagType}
    x3min::FT
    x3max::FT
    x3boundary::B
end


"""
    IntervalDomain(x3min, x3max; x3boundary::Tuple{Symbol,Symbol})

Construct a `IntervalDomain` in the vertical, the closed interval is given by `x3min`, `x3max` arugments.
Because vertical domains are not periodic, the `x3boundary` boundary tag keyword arugment must be supplied.
"""
IntervalDomain(x3min, x3max; x3boundary::Tuple{Symbol, Symbol}) =
    IntervalDomain(x3min, x3max, x3boundary)

"""
    IntervalDomain(x3::ClosedInterval; x3boundary::Tuple{Symbol,Symbol})

Construct a `IntervalDomain` in the vertical.
Because vertical domains are not periodic, the `x3boundary` boundary tag keyword arugment must be supplied.
"""
IntervalDomain(x3::ClosedInterval; x3boundary::Tuple{Symbol, Symbol}) =
    IntervalDomain(float(x3.left), float(x3.right), x3boundary)

Base.eltype(::IntervalDomain{FT}) where {FT} = FT
coordinate_type(::IntervalDomain{FT}) where {FT} = Geometry.Cartesian3Point{FT}

function Base.show(io::IO, domain::IntervalDomain)
    print(
        io,
        "IntervalDomain($(domain.x3min) .. $(domain.x3max), x3boundary = $(domain.x3boundary))",
    )
end
# coordinates (x1,x2)

struct RectangleDomain{FT, B1, B2} <: HorizontalDomain where {
    FT <: AbstractFloat,
    B1 <: BCTagType,
    B2 <: BCTagType,
}
    x1min::FT
    x1max::FT
    x2min::FT
    x2max::FT
    x1boundary::B1
    x2boundary::B2
end

"""
    RectangleDomain(x1::ClosedInterval, x2::ClosedInterval;
        x1boundary::Tuple{Symbol,Symbol},
        x2boundary::Tuple{Symbol,Symbol},
        x1periodic = false,
        x2periodic = false,
    )

Construct a `RectangularDomain` in the horizontal.
If a given x1 or x2 boundary is not periodic, then `x1boundary` or `x2boundary` boundary tag keyword arguments must be supplied.
"""
function RectangleDomain(
    x1::ClosedInterval,
    x2::ClosedInterval;
    x1periodic = false,
    x2periodic = false,
    x1boundary::BCTagType = nothing,
    x2boundary::BCTagType = nothing,
)
    if !x1periodic && !(x1boundary isa Tuple{Symbol, Symbol})
        throw(
            ArgumentError(
                "if `x1periodic=false` then an `x1boundary::Tuple{Symbol,Symbol}` boundary tag keyword argument must be supplied, got: $(x1boundary)",
            ),
        )
    end
    if !x2periodic && !(x2boundary isa Tuple{Symbol, Symbol})
        throw(
            ArgumentError(
                "if `x2periodic=false` then an `x2boundary::Tuple{Symbol,Symbol}` boundary tag keyword argument must be supplied, got: $(x2boundary)",
            ),
        )
    end
    RectangleDomain(
        float(x1.left),
        float(x1.right),
        float(x2.left),
        float(x2.right),
        x1periodic ? nothing : x1boundary,
        x2periodic ? nothing : x2boundary,
    )
end

function Base.show(io::IO, domain::RectangleDomain)
    print(
        io,
        "RectangleDomain($(domain.x1min)..$(domain.x1max), $(domain.x2min)..$(domain.x2max)",
    )
    if domain.x1boundary === nothing
        print(io, ", x1periodic=true")
    else
        print(io, ", x1boundary=$(domain.x1boundary)")
    end
    if domain.x2boundary === nothing
        print(io, ", x2periodic=true")
    else
        print(io, ", x2boundary=$(domain.x2boundary)")
    end
    print(io, ")")
end
coordinate_type(::RectangleDomain{FT}) where {FT} =
    Geometry.Cartesian2DPoint{FT}

# coordinates (-pi/2 < lat < pi/2, -pi < lon < pi)
struct SphereDomain{FT} <: HorizontalDomain
    radius::FT
end


end # module
