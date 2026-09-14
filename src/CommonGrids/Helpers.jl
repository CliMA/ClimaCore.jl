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
