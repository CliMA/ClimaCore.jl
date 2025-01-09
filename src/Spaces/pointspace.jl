abstract type AbstractPointSpace <: AbstractSpace end

local_geometry_data(space::AbstractPointSpace) = space.local_geometry

"""
    PointSpace <: AbstractSpace

A zero-dimensional space.
"""
struct PointSpace{
    C <: ClimaComms.AbstractCommsContext,
    LG <: DataLayouts.Data0D,
} <: AbstractPointSpace
    context::C
    local_geometry::LG
end

local_geometry_type(::Type{PointSpace{C, LG}}) where {C, LG} = eltype(LG) # calls eltype from DataLayouts

ClimaComms.device(space::PointSpace) = ClimaComms.device(space.context)
ClimaComms.context(space::PointSpace) = space.context

PointSpace(x::T) where {T} = PointSpace(ClimaComms.context(), x)

function PointSpace(device::ClimaComms.AbstractDevice, x)
    context = ClimaComms.SingletonCommsContext(device)
    return PointSpace(context, x)
end

function PointSpace(
    context::ClimaComms.AbstractCommsContext,
    local_geometry::LG,
) where {LG <: Geometry.LocalGeometry}
    FT = Geometry.undertype(LG)
    ArrayType = ClimaComms.array_type(ClimaComms.device(context))
    local_geometry_data = DataLayouts.DataF{LG}(Array{FT})
    local_geometry_data[] = local_geometry
    return PointSpace(context, Adapt.adapt(ArrayType, local_geometry_data))
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
