module Domains

import ..Geometry: Geometry, float_type
import IntervalSets
export RectangleDomain

"""
    AbstractDomain

Supertype for domains, which represent regions of space.

Subtypes:

  - [`IntervalDomain`](@ref): a closed interval along one coordinate direction.
  - [`RectangleDomain`](@ref): the product of two interval domains.
  - [`SphereDomain`](@ref): the surface of a sphere.
"""
abstract type AbstractDomain end

function Base.summary(io::IO, domain::AbstractDomain)
    print(io, nameof(typeof(domain)))
end

const BCTagType = Union{Nothing, Tuple{Symbol, Symbol}}

float_type(domain::AbstractDomain) = float_type(coordinate_type(domain))

"""
    boundary_names(obj::Union{AbstractDomain, AbstractMesh, AbstractTopology})

Return a tuple or vector of the unique boundary names of a spatial domain, mesh, or
topology. Periodic directions contribute no names.
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
    isperiodic(domain) ? () : unique(domain.boundary_names)

"""
    IntervalDomain(coord_min, coord_max; periodic = false, boundary_names = nothing)
    IntervalDomain(coords::ClosedInterval; kwargs...)

Construct an `IntervalDomain`, the closed interval between the 1D coordinates `coord_min`
and `coord_max` (or the endpoints of `coords`).

# Keyword Arguments

  - `periodic = false`: whether the interval is periodic.
  - `boundary_names = nothing`: a `Tuple{Symbol, Symbol}` naming the lower and upper
    boundaries. Required when `periodic` is `false`; passing neither keyword throws an
    `ArgumentError`.
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
    IntervalDomain(promote(coord_min, coord_max)..., boundary_names)
end
IntervalDomain(coords::IntervalSets.ClosedInterval; kwargs...) =
    IntervalDomain(coords.left, coords.right; kwargs...)

"""
    z_max(domain::IntervalDomain)

Return the domain maximum along the `z` direction.
"""
z_max(domain::IntervalDomain) = domain.coord_max.z

"""
    z_min(domain::IntervalDomain)

Return the domain minimum along the `z` direction.
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
        print(io, domain.boundary_names)
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

boundary_names(domain::RectangleDomain) = unique(
    Symbol[
        boundary_names(domain.interval1)...,
        boundary_names(domain.interval2)...,
    ],
)::Vector{Symbol}

"""
    RectangleDomain(interval1::IntervalDomain, interval2::IntervalDomain)
    RectangleDomain(x1::ClosedInterval, x2::ClosedInterval;
        x1periodic = false,
        x2periodic = false,
        x1boundary = nothing,
        x2boundary = nothing,
    )

Construct a `RectangleDomain`, the product of two interval domains in the horizontal.
`interval1 * interval2` is equivalent to the first form.

# Keyword Arguments

  - `x1periodic = false`, `x2periodic = false`: whether each direction is periodic.
  - `x1boundary = nothing`, `x2boundary = nothing`: `Tuple{Symbol, Symbol}` boundary names
    for each direction. Required for every direction that is not periodic.
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

Domain representing the surface of a sphere with radius `radius` [m]. Its coordinate
type is `Geometry.Cartesian123Point`, and it has no boundaries.
"""
struct SphereDomain{FT} <: AbstractDomain where {FT <: AbstractFloat}
    radius::FT
end
Base.show(io::IO, domain::SphereDomain) =
    print(io, nameof(typeof(domain)), ": radius = ", domain.radius)

boundary_names(::SphereDomain) = ()
coordinate_type(::SphereDomain{FT}) where {FT} = Geometry.Cartesian123Point{FT}

end # module
