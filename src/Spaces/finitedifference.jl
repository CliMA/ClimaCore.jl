abstract type AbstractFiniteDifferenceSpace <: AbstractSpace end

"""
    FiniteDifferenceSpace(
        grid::Grids.FiniteDifferenceGrid,
        staggering::Staggering, 
    )



"""
struct FiniteDifferenceSpace{
    G <: Grids.AbstractFiniteDifferenceGrid,
    S <: Staggering,
} <: AbstractFiniteDifferenceSpace
    grid::G
    staggering::S
end
FiniteDifferenceSpace(
    topology::Topologies.IntervalTopology,
    staggering::Staggering,
) = FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(topology), staggering)


const FaceFiniteDifferenceSpace{G} = FiniteDifferenceSpace{G, CellFace}
const CenterFiniteDifferenceSpace{G} = FiniteDifferenceSpace{G, CellCenter}

grid(space::AbstractFiniteDifferenceSpace) = getfield(space, :grid)
staggering(space::FiniteDifferenceSpace) = getfield(space, :staggering)

space(grid::Grids.AbstractFiniteDifferenceGrid, staggering::Staggering) =
    FiniteDifferenceSpace(grid, staggering)

function Base.show(io::IO, space::FiniteDifferenceSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(
        io,
        space isa CenterFiniteDifferenceSpace ? "CenterFiniteDifferenceSpace" :
        "FaceFiniteDifferenceSpace",
        ":",
    )
    print(iio, " "^(indent + 2), "context: ")
    Topologies.print_context(iio, ClimaComms.context(space))
    println(iio)
    print(iio, " "^(indent + 2), "mesh: ", topology(space).mesh)
end

FaceFiniteDifferenceSpace(grid::Grids.AbstractFiniteDifferenceGrid) =
    FiniteDifferenceSpace(grid, CellFace())
CenterFiniteDifferenceSpace(grid::Grids.AbstractFiniteDifferenceGrid) =
    FiniteDifferenceSpace(grid, CellCenter())

FaceFiniteDifferenceSpace(space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace(grid(space), CellFace())
CenterFiniteDifferenceSpace(space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace(grid(space), CellCenter())

FaceFiniteDifferenceSpace(topology::Topologies.IntervalTopology) =
    FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(topology), CellFace())
CenterFiniteDifferenceSpace(topology::Topologies.IntervalTopology) =
    FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(topology), CellCenter())

FaceFiniteDifferenceSpace(mesh::Meshes.IntervalMesh) =
    FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(mesh), CellFace())
CenterFiniteDifferenceSpace(mesh::Meshes.IntervalMesh) =
    FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(mesh), CellCenter())

@inline function Base.getproperty(space::FiniteDifferenceSpace, name::Symbol)
    if name == :topology
        Base.depwarn(
            "`space.topology` is deprecated, use `Spaces.topology(space)` instead",
            :getproperty,
        )
        return topology(space)
    elseif name == :global_geometry
        Base.depwarn(
            "`space.global_geometry` is deprecated, use `Spaces.global_geometry(space)` instead",
            :getproperty,
        )
        return global_geometry(space)
    elseif name == :center_local_geometry
        Base.depwarn(
            "`space.center_local_geometry` is deprecated, use `local_geometry_data(grid(space), Grids.CellCenter())` instead",
            :getproperty,
        )
        return local_geometry_data(space, Grids.CellCenter())
    elseif name == :face_local_geometry
        Base.depwarn(
            "`space.face_local_geometry` is deprecated, use `local_geometry_data(grid(space), Grids.CellFace())` instead",
            :getproperty,
        )
        return local_geometry_data(space, Grids.CellFace())
    end
    return getfield(space, name)
end

Adapt.adapt_structure(to, space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace(Adapt.adapt(to, grid(space)), staggering(space))



nlevels(space::FiniteDifferenceSpace) = length(space)
# TODO: deprecate?
Base.length(space::FiniteDifferenceSpace) = length(coordinates_data(space))



Base.@deprecate z_component(::Type{T}) where {T} Δz_metric_component(T) false

"""
    Δz_metric_component(::Type{<:Goemetry.AbstractPoint})

The index of the z-component of an abstract point
in an `AxisTensor`.
"""
Δz_metric_component(::Type{<:Geometry.LatLongZPoint}) = 9
Δz_metric_component(::Type{<:Geometry.Cartesian3Point}) = 1
Δz_metric_component(::Type{<:Geometry.Cartesian13Point}) = 4
Δz_metric_component(::Type{<:Geometry.Cartesian123Point}) = 9
Δz_metric_component(::Type{<:Geometry.XYZPoint}) = 9
Δz_metric_component(::Type{<:Geometry.ZPoint}) = 1
Δz_metric_component(::Type{<:Geometry.XZPoint}) = 4

Base.@deprecate dz_data(space::AbstractSpace) Δz_data(space) false

"""
    Δz_data(space::AbstractSpace)

A DataLayout containing the `Δz` on a given space `space`.
"""
function Δz_data(space::AbstractSpace)
    lg = local_geometry_data(space)
    data_layout_type = eltype(lg.coordinates)
    return getproperty(
        lg.∂x∂ξ.components.data,
        Δz_metric_component(data_layout_type),
    )
end

function left_boundary_name(space::AbstractSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
    propertynames(boundaries)[1]
end

function right_boundary_name(space::AbstractSpace)
    boundaries = Topologies.boundaries(Spaces.vertical_topology(space))
    propertynames(boundaries)[2]
end

Base.@propagate_inbounds function level(
    space::FaceFiniteDifferenceSpace,
    v::PlusHalf,
)
    @inbounds local_geometry = level(local_geometry_data(space), v.i + 1)
    PointSpace(local_geometry)
end
Base.@propagate_inbounds function level(
    space::CenterFiniteDifferenceSpace,
    v::Int,
)
    local_geometry = level(local_geometry_data(space), v)
    PointSpace(local_geometry)
end
