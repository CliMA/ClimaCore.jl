module Helpers

import ...Meshes, ...Geometry, ...Domains


#####
##### Mesh helpers
#####

"""
    DefaultSliceXMesh(
        ::Type{<:AbstractFloat}; # defaults to Float64
        x_min::Real,
        x_max::Real,
        periodic_x::Bool,
        x_elem::Integer,
    )

A convenience constructor, which builds an `IntervalMesh`.
"""
DefaultSliceXMesh(; kwargs...) = DefaultSliceXMesh(Float64; kwargs...)
function DefaultSliceXMesh(
    ::Type{FT};
    x_min::Real,
    x_max::Real,
    periodic_x::Bool,
    x_elem::Integer,
) where {FT}

    x1boundary = periodic_x ? nothing : (:east, :west)
    h_domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(x_min),
        Geometry.XPoint{FT}(x_max);
        periodic = periodic_x,
        boundary_names = x1boundary,
    )
    return Meshes.IntervalMesh(h_domain; nelems = x_elem)
end

"""
    DefaultZMesh(
        ::Type{<:AbstractFloat}; # defaults to Float64
        z_min::Real,
        z_max::Real,
        z_elem::Integer,
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
    )

A convenience constructor, which builds an `IntervalMesh`.
"""
DefaultZMesh(; kwargs...) = DefaultZMesh(Float64; kwargs...)
function DefaultZMesh(
    ::Type{FT};
    z_min::Real,
    z_max::Real,
    z_elem::Integer,
    stretch::Meshes.StretchingRule = Meshes.Uniform(),
) where {FT}
    z_boundary_names = (:bottom, :top)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z_min),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = z_boundary_names,
    )
    return Meshes.IntervalMesh(z_domain, stretch; nelems = z_elem)
end

"""
    DefaultRectangleXYMesh(
        ::Type{<:AbstractFloat}; # defaults to Float64
        x_min::Real,
        x_max::Real,
        y_min::Real,
        y_max::Real,
        periodic_x::Bool,
        periodic_y::Bool,
    )

A convenience constructor, which builds a
`RectilinearMesh` with a rectangular domain
composed of interval domains.
"""
DefaultRectangleXYMesh(; kwargs...) = DefaultRectangleXYMesh(Float64; kwargs...)
function DefaultRectangleXYMesh(
    ::Type{FT};
    x_min::Real,
    x_max::Real,
    y_min::Real,
    y_max::Real,
    x_elem::Integer,
    y_elem::Integer,
    periodic_x::Bool,
    periodic_y::Bool,
) where {FT <: AbstractFloat}
    x1boundary = periodic_x ? nothing : (:east, :west)
    x2boundary = periodic_y ? nothing : (:south, :north)

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint{FT}(x_min),
            Geometry.XPoint{FT}(x_max);
            periodic = periodic_x,
            boundary_names = x1boundary,
        ),
        Domains.IntervalDomain(
            Geometry.YPoint{FT}(y_min),
            Geometry.YPoint{FT}(y_max);
            periodic = periodic_y,
            boundary_names = x2boundary,
        ),
    )
    return Meshes.RectilinearMesh(domain, x_elem, y_elem)
end

end # module
