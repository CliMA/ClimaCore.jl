module Domains

import ..Geometry
using IntervalSets
export RectangleDomain


abstract type HorizontalDomain end
abstract type VerticalDomain end

Base.@kwdef struct IntervalDomain{FT} <: VerticalDomain
    x3min::FT
    x3max::FT
end

coordinate_type(::IntervalDomain{FT}) where {FT} = Geometry.Cartesian3Point{FT}



# coordinates (x1,x2)

struct RectangleDomain{FT, B1, B2} <: HorizontalDomain
    x1min::FT
    x1max::FT
    x2min::FT
    x2max::FT
    x1boundary::B1
    x2boundary::B2
end

"""
    RectangleDomain(x1::ClosedInterval, x2::ClosedInterval;
        x1boundary = (:west, :east),
        x2boundary = (:south, :north),
        x1periodic = false,
        x2periodic = false,
    )

Construct a `RectangularDomain` in the horizontal. 
"""
RectangleDomain(
    x1::ClosedInterval,
    x2::ClosedInterval;
    x1boundary = (:west, :east),
    x2boundary = (:south, :north),
    x1periodic = false,
    x2periodic = false,
) = RectangleDomain(
    float(x1.left),
    float(x1.right),
    float(x2.left),
    float(x2.right),
    x1periodic ? nothing : x1boundary,
    x2periodic ? nothing : x2boundary,
)

function Base.show(io::IO, domain::RectangleDomain)
    print(
        io,
        "RectangleDomain($(domain.x1min)..$(domain.x1max), $(domain.x2min)..$(domain.x2max)",
    )
    if domain.x1boundary == nothing
        print(io, ", x1periodic=true")
    else
        print(io, ", x1boundary=$(domain.x1boundary)")
    end
    if domain.x2boundary == nothing
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
