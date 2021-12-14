module Domains

import ..Geometry: Geometry, float_type
using IntervalSets
export RectangleDomain


abstract type AbstractDomain end

const BCTagType = Union{Nothing, Tuple{Symbol, Symbol}}

float_type(domain::AbstractDomain) = float_type(coordinate_type(domain))

"""
    boundary_names(obj::Union{AbstractDomain, AbstractMesh, AbstractTopology})

The boundary names of a spatial domain. This is a tuple of `Symbol`s.
"""
function boundary_names end

struct IntervalDomain{CT, B} <: AbstractDomain where {
    CT <: Geometry.Abstract1DPoint{FT},
    B <: BCTagType,
} where {FT}
    coord_min::CT
    coord_max::CT
    boundary_names::B
end

isperiodic(domain::IntervalDomain) = isnothing(domain.boundary_names)
boundary_names(domain::IntervalDomain) =
    isperiodic(domain) ? () : domain.boundary_names

"""
    IntervalDomain(coord⁻, coord⁺; periodic=true)
    IntervalDomain(coord⁻, coord⁺; boundary_names::Tuple{Symbol,Symbol})

Construct a `IntervalDomain`, the closed interval is given by `coord⁻`, `coord⁺` coordinate arguments.

Either a `periodic` or `boundary_names` keyword argument is required.
"""
function IntervalDomain(
    coord_min::Geometry.Abstract1DPoint,
    coord_max::Geometry.Abstract1DPoint;
    periodic = false,
    boundary_tags = nothing, # TODO: deprecate this
    boundary_names::BCTagType = boundary_tags,
)
    if !periodic && isnothing(boundary_names)
        throw(
            ArgumentError(
                "if `periodic=false` then an `boundary_names::Tuple{Symbol,Symbol}` keyword argument is required.",
            ),
        )
    end
    IntervalDomain(promote(coord_min, coord_max)..., boundary_names)
end
IntervalDomain(coords::ClosedInterval; kwargs...) =
    IntervalDomain(coords.left, coords.right; kwargs...)

coordinate_type(::IntervalDomain{CT}) where {CT} = CT
Base.eltype(domain::IntervalDomain) = coordinate_type(domain)

function Base.show(io::IO, domain::IntervalDomain)
    print(io, "IntervalDomain($(domain.coord_min) .. $(domain.coord_max); ")
    if isperiodic(domain)
        print(io, "periodic=true)")
    else
        print(io, "boundary_names = $(domain.boundary_names))")
    end
end

struct RectangleDomain{I1 <: IntervalDomain, I2 <: IntervalDomain} <:
       AbstractDomain
    interval1::I1
    interval2::I2
end
Base.:*(interval1::IntervalDomain, interval2::IntervalDomain) =
    RectangleDomain(interval1, interval2)

function boundary_names(domain::RectangleDomain)
    (boundary_names(domain.interval1)..., boundary_names(domain.interval2)...)
end

"""
    RectangleDomain(x1::ClosedInterval, x2::ClosedInterval;
        x1boundary::Tuple{Symbol,Symbol},
        x2boundary::Tuple{Symbol,Symbol},
        x1periodic = false,
        x2periodic = false,
    )

Construct a `RectangularDomain` in the horizontal.
If a given x1 or x2 boundary is not periodic, then `x1boundary` or `x2boundary` boundary name keyword arguments must be supplied.
"""
function RectangleDomain(
    x1::ClosedInterval{X1CT},
    x2::ClosedInterval{X2CT};
    x1periodic = false,
    x2periodic = false,
    x1boundary::BCTagType = nothing,
    x2boundary::BCTagType = nothing,
) where {X1CT <: Geometry.Abstract1DPoint, X2CT <: Geometry.Abstract1DPoint}
    interval1 =
        IntervalDomain(x1; periodic = x1periodic, boundary_names = x1boundary)
    interval2 =
        IntervalDomain(x2; periodic = x2periodic, boundary_names = x2boundary)
    return interval1 * interval2
end


function Base.show(io::IO, domain::RectangleDomain)
    x1min = domain.interval1.coord_min
    x2min = domain.interval2.coord_min
    x1max = domain.interval1.coord_max
    x2max = domain.interval2.coord_max
    print(io, "RectangleDomain($(x1min)..$(x1max), $(x2min)..$(x2max)")
    if isperiodic(domain.interval1)
        print(io, ", x1periodic=true")
    else
        print(io, ", x1boundary=$(domain.interval1.boundary_names)")
    end
    if isperiodic(domain.interval2)
        print(io, ", x2periodic=true")
    else
        print(io, ", x2boundary=$(domain.interval2.boundary_names)")
    end
    print(io, ")")
end
coordinate_type(domain::RectangleDomain) = typeof(
    Geometry.product_coordinates(
        domain.interval1.coord_min,
        domain.interval2.coord_min,
    ),
)

# coordinates (-pi/2 < lat < pi/2, -pi < lon < pi)
struct SphereDomain{FT} <: AbstractDomain where {FT <: AbstractFloat}
    radius::FT
end

boundary_names(::SphereDomain) = ()
coordinate_type(::SphereDomain{FT}) where {FT} = Geometry.Cartesian123Point{FT}

end # module
