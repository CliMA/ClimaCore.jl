module Domains

import ..Geometry: Geometry, float_type
import IntervalSets
export RectangleDomain

"""
    AbstractDomain

A domain represents a region of space.
"""
abstract type AbstractDomain end

function Base.summary(io::IO, domain::AbstractDomain)
    print(io, nameof(typeof(domain)))
end

const BCTagType = Union{Nothing, Tuple{Symbol, Symbol}}

float_type(domain::AbstractDomain) = float_type(coordinate_type(domain))

"""
    boundary_names(obj::Union{AbstractDomain, AbstractMesh, AbstractTopology})

The boundary names passed to the IntervalDomain (a tuple, or `nothing`).
"""
function boundary_names end

"""
    unique_boundary_names(obj::Union{AbstractDomain, AbstractMesh, AbstractTopology})

A tuple or vector of unique boundary names of a spatial domain.
"""
function unique_boundary_names end

struct IntervalDomain{CT, B} <:
       AbstractDomain where {CT <: Geometry.Abstract1DPoint{FT}, B} where {FT}
    coord_min::CT
    coord_max::CT
end

isperiodic(::IntervalDomain{CT, B}) where {CT, B} = B == nothing
unique_boundary_names(domain::IntervalDomain{CT, B}) where {CT, B} =
    isperiodic(domain) ? Symbol[] : unique(B)
boundary_names(::IntervalDomain{CT, B}) where {CT, B} = B

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
    boundary_names::BCTagType = nothing,
)
    if !periodic && isnothing(boundary_names)
        throw(
            ArgumentError(
                "if `periodic=false` then a `boundary_names::Tuple{Symbol,Symbol}` keyword argument is required.",
            ),
        )
    end
    c = promote(coord_min, coord_max)
    boundary_names = if isnothing(boundary_names)
        boundary_names
    else
        Tuple(boundary_names)
    end
    IntervalDomain{eltype(c), boundary_names}(c...)
end
IntervalDomain(coords::IntervalSets.ClosedInterval; kwargs...) =
    IntervalDomain(coords.left, coords.right; kwargs...)

"""
    z_max(domain::IntervalDomain)

The domain maximum along the z-direction.
"""
z_max(domain::IntervalDomain) = domain.coord_max.z

"""
    z_min(domain::IntervalDomain)

The domain minimum along the z-direction.
"""
z_min(domain::IntervalDomain) = domain.coord_min.z

coordinate_type(::IntervalDomain{CT}) where {CT} = CT
Base.eltype(domain::IntervalDomain) = coordinate_type(domain)

function print_interval(io::IO, domain::IntervalDomain{CT}) where {CT}
    print(
        io,
        fieldname(CT, 1),
        " ∈ [",
        Geometry.component(domain.coord_min, 1),
        ",",
        Geometry.component(domain.coord_max, 1),
        "] ",
    )
    if isperiodic(domain)
        print(io, "(periodic)")
    else
        print(io, boundary_names(domain))
    end
end
function Base.show(io::IO, domain::IntervalDomain)
    print(io, nameof(typeof(domain)), ": ")
    print_interval(io, domain)
end

struct RectangleDomain{I1 <: IntervalDomain, I2 <: IntervalDomain} <:
       AbstractDomain
    interval1::I1
    interval2::I2
end
Base.:*(interval1::IntervalDomain, interval2::IntervalDomain) =
    RectangleDomain(interval1, interval2)

unique_boundary_names(domain::RectangleDomain) = unique(
    Symbol[
        unique_boundary_names(domain.interval1)...,
        unique_boundary_names(domain.interval2)...,
    ],
)::Vector{Symbol}

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
    x1::IntervalSets.ClosedInterval{X1CT},
    x2::IntervalSets.ClosedInterval{X2CT};
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
    print(io, nameof(typeof(domain)), ": ")
    print_interval(io, domain.interval1)
    print(io, " × ")
    print_interval(io, domain.interval2)
end

coordinate_type(domain::RectangleDomain) = typeof(
    Geometry.product_coordinates(
        domain.interval1.coord_min,
        domain.interval2.coord_min,
    ),
)

"""
    SphereDomain(radius)

A domain representing the surface of a sphere with radius `radius`.
"""
struct SphereDomain{FT} <: AbstractDomain where {FT <: AbstractFloat}
    radius::FT
end
Base.show(io::IO, domain::SphereDomain) =
    print(io, nameof(typeof(domain)), ": radius = ", domain.radius)

boundary_names(::SphereDomain) = ()
unique_boundary_names(::SphereDomain) = Symbol[]
coordinate_type(::SphereDomain{FT}) where {FT} = Geometry.Cartesian123Point{FT}

end # module
