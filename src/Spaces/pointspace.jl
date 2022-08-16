abstract type AbstractPointSpace <: AbstractSpace end

local_geometry_data(space::AbstractPointSpace) = space.local_geometry

"""
    PointSpace <: AbstractSpace

A zero-dimensional space.
"""
struct PointSpace{LG} <: AbstractPointSpace
    local_geometry::LG
end

function PointSpace(local_geometry::LG) where {LG <: Geometry.LocalGeometry}
    FT = Geometry.undertype(LG)
    local_geometry_data = DataLayouts.DataF{LG}(Array{FT})
    local_geometry_data[] = local_geometry
    return PointSpace(local_geometry_data)
end

function PointSpace(coord::Geometry.Abstract1DPoint{FT}) where {FT}
    CoordType = typeof(coord)
    AIdx = Geometry.coordinate_axis(CoordType)
    local_geometry = Geometry.LocalGeometry(
        coord,
        FT(1.0),
        FT(1.0),
        Geometry.AxisTensor(
            (Geometry.LocalAxis{AIdx}(), Geometry.CovariantAxis{AIdx}()),
            FT(1.0),
        ),
    )
    return PointSpace(local_geometry)
end
