
"""
    AbstractGlobalGeometry

Determines the conversion from local coordinates and vector bases to a Cartesian basis.
"""
abstract type AbstractGlobalGeometry end

Cartesian123Point(pt::AbstractPoint, global_geometry::AbstractGlobalGeometry) =
    Cartesian123Point(CartesianPoint(pt, global_geometry))

(::Type{<:CartesianVector{<:Any, I}})(
    u::AxisVector,
    global_geometry::AbstractGlobalGeometry,
    local_geometry::LocalGeometry,
) where {I} = transform(
    CartesianAxis{I}(),
    CartesianVector(u, global_geometry, local_geometry),
)


"""
    CartesianGlobalGeometry()

Specifies that the local coordinates align with the Cartesian coordinates, e.g.
`XYZPoint` aligns with `Cartesian123Point`, and `UVWVector` aligns with
`Cartesian123Vector`.
"""
struct CartesianGlobalGeometry <: AbstractGlobalGeometry end

# coordinates
CartesianPoint(pt::XPoint{FT}, ::CartesianGlobalGeometry) where {FT} =
    Cartesian1Point{FT}(pt.x)
CartesianPoint(pt::ZPoint{FT}, ::CartesianGlobalGeometry) where {FT} =
    Cartesian3Point{FT}(pt.z)
CartesianPoint(pt::XYPoint{FT}, ::CartesianGlobalGeometry) where {FT} =
    Cartesian12Point{FT}(pt.x, pt.y)
CartesianPoint(pt::XZPoint{FT}, ::CartesianGlobalGeometry) where {FT} =
    Cartesian13Point{FT}(pt.x, pt.z)
CartesianPoint(pt::XYZPoint{FT}, ::CartesianGlobalGeometry) where {FT} =
    Cartesian123Point{FT}(pt.x, pt.y, pt.z)

# vectors
CartesianVector(
    u::CartesianVector,
    ::CartesianGlobalGeometry,
    ::LocalGeometry,
) = u
function CartesianVector(
    u::AxisVector,
    ::CartesianGlobalGeometry,
    local_geometry::LocalGeometry{I},
) where {I}
    u_local = LocalVector(u, local_geometry)
    AxisVector(CartesianAxis{I}(), components(u_local))
end


#=
LocalVector(u::CartesianVector{T,I}, ::CartesianGlobalGeometry) where {T,I} =
    AxisVector(LocalAxis{I}(), components(u))
=#


"""
    SphericalGlobalGeometry(radius)

Specifies that the local coordinates are specified in reference to a sphere of
radius `radius`. The `x1` axis is aligned with the zero longitude line.

The local vector basis has `u` in the zonal direction (with east being
positive), `v` in the meridonal (north positive), and `w` in the radial
direction (outward positive). For a point located at the pole, we take the limit
along the zero longitude line:
- at the north pole, this corresponds to `u` being aligned with the `x2`
  direction, `v` being aligned with the negative `x1` direction, and `w` being
  aligned with the `x3` direction.
- at the south pole, this corresponds to `u` being aligned with the `x2`
  direction, `v` being aligned with the `x1` direction, and `w` being aligned
  with the negative `x3` direction.
"""
struct SphericalGlobalGeometry{FT} <: AbstractGlobalGeometry
    radius::FT
end

# coordinates
function CartesianPoint(pt::LatLongPoint, global_geom::SphericalGlobalGeometry)
    r = global_geom.radius
    x1 = r * cosd(pt.long) * cosd(pt.lat)
    x2 = r * sind(pt.long) * cosd(pt.lat)
    x3 = r * sind(pt.lat)
    Cartesian123Point(x1, x2, x3)
end
function LatLongPoint(pt::Cartesian123Point, ::SphericalGlobalGeometry)
    ϕ = atand(pt.x3, hypot(pt.x2, pt.x1))
    # IEEE754 spec states that atand(±0.0, −0.0) == ±180, however to make the UV
    # orienation consistent, we define the longitude to be zero at the poles
    if abs(ϕ) == 90
        λ = zero(ϕ)
    else
        λ = atand(pt.x2, pt.x1)
    end
    LatLongPoint(ϕ, λ)
end

"""
    great_circle_distance(pt1::LatLongPoint, pt2::LatLongPoint, global_geometry::SphericalGlobalGeometry)

Compute the great circle (spherical geodesic) distance between `pt1` and `pt2`.
"""
function great_circle_distance(
    pt1::LatLongPoint,
    pt2::LatLongPoint,
    global_geom::SphericalGlobalGeometry,
)
    r = global_geom.radius
    ϕ1 = pt1.lat
    λ1 = pt1.long
    ϕ2 = pt2.lat
    λ2 = pt2.long
    Δλ = λ1 - λ2
    return r * atan(
        hypot(
            cosd(ϕ2) * sind(Δλ),
            cosd(ϕ1) * sind(ϕ2) - sind(ϕ1) * cosd(ϕ2) * cosd(Δλ),
        ),
        cosd(ϕ1) * cosd(ϕ2) * cosd(Δλ) + sind(ϕ1) * sind(ϕ2),
    )
end

# vectors
CartesianVector(
    u::CartesianVector,
    ::SphericalGlobalGeometry,
    ::LocalGeometry,
) = u
CartesianVector(
    u::AxisVector,
    global_geometry::SphericalGlobalGeometry,
    local_geometry::LocalGeometry,
) = CartesianVector(
    UVWVector(u, local_geometry),
    global_geometry,
    local_geometry.coordinates,
)
function CartesianVector(
    u::UVWVector,
    ::SphericalGlobalGeometry,
    coord::LatLongPoint,
)
    ϕ = coord.lat
    λ = coord.long
    G = @SMatrix [
        -sind(λ) -sind(ϕ)*cosd(λ) cosd(λ)*cosd(ϕ)
        cosd(λ) -sind(ϕ)*sind(λ) sind(λ)*cosd(ϕ)
        0 cosd(ϕ) sind(ϕ)
    ]
    AxisVector(Cartesian123Axis(), G * Geometry.components(u))
end
