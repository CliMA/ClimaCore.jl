abstract type AbstractFiniteDifferenceSpace <: AbstractSpace end

"""
    FiniteDifferenceSpace(
        grid::Grids.FiniteDifferenceGrid,
        staggering::Staggering
    )

A 1D finite-difference space, that lives on
either:

 - cell centers (where `staggering` is [`Grids.CellCenter`](@ref)) or
 - cell faces (where `staggering` is [`Grids.CellFace`](@ref))
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

local_geometry_type(::Type{FiniteDifferenceSpace{G, S}}) where {G, S} =
    local_geometry_type(G)

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

FaceFiniteDifferenceSpace(
    device::ClimaComms.AbstractDevice,
    mesh::Meshes.IntervalMesh,
) = FiniteDifferenceSpace(Grids.FiniteDifferenceGrid(device, mesh), CellFace())
CenterFiniteDifferenceSpace(
    device::ClimaComms.AbstractDevice,
    mesh::Meshes.IntervalMesh,
) = FiniteDifferenceSpace(
    Grids.FiniteDifferenceGrid(device, mesh),
    CellCenter(),
)

Adapt.adapt_structure(to, space::FiniteDifferenceSpace) =
    FiniteDifferenceSpace(Adapt.adapt(to, grid(space)), staggering(space))

"""
    face_space(space::FiniteDifferenceSpace)

Return face-centered space corresponding to `space`.

If `space` is already face-centered, return itself.
"""
function face_space(space::FiniteDifferenceSpace)
    return FiniteDifferenceSpace(grid(space), CellFace())
end

"""
    center_space(space::FiniteDifferenceSpace)

Return center-centered space corresponding to `space`.

If `space` is already center-centered, return itself.
"""
function center_space(space::FiniteDifferenceSpace)
    return FiniteDifferenceSpace(grid(space), CellCenter())
end

nlevels(space::FiniteDifferenceSpace) = length(space)
# TODO: deprecate?
Base.length(space::FiniteDifferenceSpace) = length(coordinates_data(space))

"""
    Δz_data(space::AbstractSpace)

A DataLayout containing the `Δz` on a given space `space`.
"""
function Δz_data(space::AbstractSpace)
    lg = local_geometry_data(space)
    return getproperty(
        lg.∂x∂ξ.components.data,
        Geometry.Δz_metric_component(eltype(lg.coordinates)),
    )
    return DataLayouts.get_Δz(lg)
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
    PointSpace(ClimaComms.context(space), local_geometry)
end
Base.@propagate_inbounds function level(
    space::CenterFiniteDifferenceSpace,
    v::Int,
)
    local_geometry = level(local_geometry_data(space), v)
    PointSpace(ClimaComms.context(space), local_geometry)
end
