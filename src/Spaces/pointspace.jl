abstract type AbstractPointSpace <: AbstractSpace end

local_geometry_data(space::AbstractPointSpace) = space.local_geometry

"""
    PointSpace <: AbstractSpace

A zero-dimensional space.
"""
struct PointSpace{C <: ClimaComms.AbstractCommsContext, LG} <:
       AbstractPointSpace
    context::C
    local_geometry::LG
end

ClimaComms.device(space::PointSpace) = ClimaComms.device(space.context)
ClimaComms.context(space::PointSpace) = space.context

@deprecate PointSpace(x::Geometry.LocalGeometry) PointSpace(
    ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice()),
    x,
) false

function PointSpace(
    context::ClimaComms.AbstractCommsContext,
    local_geometry::LG,
) where {LG <: Geometry.LocalGeometry}
    FT = Geometry.undertype(LG)
    # TODO: inherit array type
    local_geometry_data = DataLayouts.DataF{LG}(Array{FT})
    local_geometry_data[] = local_geometry
    return PointSpace(context, local_geometry_data)
end

function PointSpace(
    context::ClimaComms.AbstractCommsContext,
    coord::Geometry.Abstract1DPoint{FT},
) where {FT}
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
    return PointSpace(context, local_geometry)
end
