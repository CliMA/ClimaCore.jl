module Helpers

import ...Meshes, ...Geometry, ...Domains


#####
##### Mesh helpers
#####

"""
    DefaultSliceXMesh(; kwargs...)
    DefaultSliceXMesh(
        ::Type{FT};
        x_min::Real,
        x_max::Real,
        periodic_x::Bool,
        x_elem::Integer,
    )

Build the `Meshes.IntervalMesh` along `x` used by the slice grids.

The float type `FT` defaults to `Float64`. When `periodic_x` is `false`, the boundaries
are named `:west` and `:east`.
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
    DefaultZMesh(; kwargs...)
    DefaultZMesh(
        ::Type{FT};
        z_min::Real,
        z_max::Real,
        z_elem::Integer,
        stretch::Meshes.StretchingRule = Meshes.Uniform(),
    )

Build the vertical `Meshes.IntervalMesh` used by the extruded grids.

The float type `FT` defaults to `Float64`. The boundaries are named `:bottom` and `:top`.
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
    DefaultRectangleXYMesh(; kwargs...)
    DefaultRectangleXYMesh(
        ::Type{FT};
        x_min::Real,
        x_max::Real,
        y_min::Real,
        y_max::Real,
        x_elem::Integer,
        y_elem::Integer,
        periodic_x::Bool,
        periodic_y::Bool,
    )

Build the `Meshes.RectilinearMesh` on a rectangular domain composed of two interval
domains, as used by the rectangle and box grids.

The float type `FT` defaults to `Float64`. Non-periodic boundaries are named `:west`,
`:east`, `:south`, and `:north`.
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
