abstract type AbstractPointSpace <: AbstractSpace end

local_geometry_data(space::AbstractPointSpace) = space.local_geometry

"""
    PointSpace <: AbstractSpace

A zero-dimensional space: a single point, the horizontal space of one column
(`Spaces.level` of a [`FiniteDifferenceSpace`](@ref)). For `N` disconnected
horizontal points see [`MultiPointSpace`](@ref), and for `N` independent
columns see [`MultiColumnFiniteDifferenceSpace`](@ref).
"""
struct PointSpace{
    C <: ClimaComms.AbstractCommsContext,
    LG <: DataLayouts.DataLayout{<:Any, 0},
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

PointSpace(context::ClimaComms.AbstractCommsContext, data::DataLayouts.DataLayout) =
    PointSpace(context, view(data)) # view of a DataLayout is always a DataF

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
        Geometry.Tensor(
            FT(1) * I,
            (
                Geometry.Components{Geometry.Orthonormal, AIdx}(),
                Geometry.Components{Geometry.Covariant, AIdx}(),
            ),
        ),
    )
    return PointSpace(context, local_geometry)
end

all_nodes(::PointSpace) = (1,)

node_horizontal_length_scale(space::PointSpace) = 1

for f in (:level, :slab, :column)
    @eval $f(space::PointSpace, indices...) =
        all(isone, indices) ? space : throw(ArgumentError("Space has only one point"))
end
