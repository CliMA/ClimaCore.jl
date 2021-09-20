module Domains

import ..Geometry
using IntervalSets
export RectangleDomain, Unstructured2DDomain


abstract type AbstractDomain end

abstract type HorizontalDomain <: AbstractDomain end
abstract type VerticalDomain <: AbstractDomain end

const BCTagType = Union{Nothing, Tuple{Symbol, Symbol}}

struct IntervalDomain{CT, B} <: VerticalDomain where {
    CT <: Geometry.Abstract1DPoint{FT},
    B <: BCTagType,
} where {FT}
    coord_min::CT
    coord_max::CT
    boundary_tags::B
end

"""
    IntervalDomain(coord⁻, coord⁺; boundary_tags::Tuple{Symbol,Symbol})

Construct a `IntervalDomain`, the closed interval is given by `coord⁻`, `coord⁺` coordinate arguments.
Because `IntervalDomain` does not support periodic boundary conditions, the `boundary_tags` keyword arugment must be supplied.
"""
function IntervalDomain(
    coord⁻::Geometry.Abstract1DPoint,
    coord⁺::Geometry.Abstract1DPoint;
    boundary_tags::Tuple{Symbol, Symbol},
)
    coords = promote(coord⁻, coord⁺)
    IntervalDomain(first(coords), last(coords), boundary_tags)
end

"""
    IntervalDomain(coords::ClosedInterval; boundary_tags::Tuple{Symbol,Symbol})

Construct a `IntervalDomain`, over the closed coordinate interval `coords`
Because `IntervalDomain` does not support periodic boundary conditions, the `boundary_tags` keyword arugment must be supplied.
"""
IntervalDomain(coords::ClosedInterval; boundary_tags::Tuple{Symbol, Symbol}) =
    IntervalDomain(coords.left, coords.right, boundary_tags)

coordinate_type(::IntervalDomain{CT}) where {CT} = CT
Base.eltype(domain::IntervalDomain) = coordinate_type(domain)

function Base.show(io::IO, domain::IntervalDomain)
    print(
        io,
        "IntervalDomain($(domain.coord_min) .. $(domain.coord_max), boundary_tags = $(domain.boundary_tags))",
    )
end
# coordinates (x1,x2)

struct RectangleDomain{CT, B1, B2} <: HorizontalDomain where {
    CT <: Geometry.Abstract2DPoint{FT},
    B1 <: BCTagType,
    B2 <: BCTagType,
} where {FT}
    x1x2min::CT
    x1x2max::CT
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
    x1::ClosedInterval{X1CT},
    x2::ClosedInterval{X2CT};
    x1periodic = false,
    x2periodic = false,
    x1boundary::BCTagType = nothing,
    x2boundary::BCTagType = nothing,
) where {X1CT <: Geometry.Abstract1DPoint, X2CT <: Geometry.Abstract1DPoint}
    UX1CT = Geometry.unionalltype(X1CT)
    UX2CT = Geometry.unionalltype(X2CT)
    if UX1CT === UX2CT
        throw(
            ArgumentError(
                "x1 and x2 domain axis coordinates cannot be the same type: `$(UX1CT)`",
            ),
        )
    end
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
    x1x2min, x1x2max = promote(
        Geometry.product_coordinates(x1.left, x2.left),
        Geometry.product_coordinates(x1.right, x2.right),
    )
    RectangleDomain(
        x1x2min,
        x1x2max,
        x1periodic ? nothing : x1boundary,
        x2periodic ? nothing : x2boundary,
    )
end

function Base.show(io::IO, domain::RectangleDomain{CT}) where {CT}
    x1min = Geometry.coordinate(domain.x1x2min, 1)
    x2min = Geometry.coordinate(domain.x1x2min, 2)
    x1max = Geometry.coordinate(domain.x1x2max, 1)
    x2max = Geometry.coordinate(domain.x1x2max, 2)
    print(io, "RectangleDomain($(x1min)..$(x1max), $(x2min)..$(x2max)")
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
coordinate_type(::RectangleDomain{CT}) where {CT} = CT

# coordinates (-pi/2 < lat < pi/2, -pi < lon < pi)
struct SphereDomain{FT} <: HorizontalDomain
    radius::FT
end

struct Unstructured2DDomain{FT} <: HorizontalDomain where {FT <: AbstractFloat} end

coordinate_type(::Unstructured2DDomain{FT}) where {FT} =
    Geometry.Cartesian12Point{FT}

end # module
