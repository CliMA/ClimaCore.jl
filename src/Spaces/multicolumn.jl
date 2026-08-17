"""
    PointCloudSpace

A horizontal space of N independent (lat, lon) points.  This is the N-column
analogue of [`PointSpace`](@ref), which is the single-column level space.

Like [`SpectralElementSpace2D`](@ref), the wrapped `grid` is either:

  - a `Grids.PointCloudGrid` which is the level-agnostic horizontal space of a
    [`MultiColumnFiniteDifferenceSpace`](@ref) (returned by
    [`Spaces.horizontal_space`](@ref))
  - a `Grids.LevelGrid` of the extruded multi-column grid which is a single
    vertical level (returned by [`Spaces.level`](@ref)), carrying full 3-D local
    geometry.
"""
struct PointCloudSpace{G <: Grids.AbstractGrid} <: AbstractSpace
    grid::G
end

grid(space::PointCloudSpace) = getfield(space, :grid)
staggering(space::PointCloudSpace) = nothing

space(grid::Grids.PointCloudGrid, ::Nothing) = PointCloudSpace(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedPointCloudGrid}, ::Nothing) =
    PointCloudSpace(grid)

ClimaComms.context(space::PointCloudSpace) =
    ClimaComms.context(grid(space))
ClimaComms.device(space::PointCloudSpace) = ClimaComms.device(grid(space))

local_geometry_data(space::PointCloudSpace) =
    local_geometry_data(grid(space), nothing)
local_geometry_type(::Type{PointCloudSpace{G}}) where {G} =
    local_geometry_type(G)

Adapt.adapt_structure(to, space::PointCloudSpace) =
    PointCloudSpace(Adapt.adapt(to, grid(space)))

"""
    MultiColumnFiniteDifferenceSpace

A space of N independent vertical columns at arbitrary horizontal (lat, lon)
locations on a sphere.  This is the N-column generalisation of
[`Spaces.FiniteDifferenceSpace`](@ref) (the single-column space):

  - The data layout is `VIJFH{LG, Nv, 1, 1, N}` (same vertical structure for every
    column; full 3-D local geometry including lat/lon/z coordinates).
  - [`Spaces.level`](@ref) returns a [`PointCloudSpace`](@ref) (N points at that
    z-level) rather than a spectral-element horizontal space.
  - [`Spaces.column`](@ref) returns a single-column
    [`Spaces.FiniteDifferenceSpace`](@ref).
  - [`Fields.bycolumn`](@ref) iterates over each column independently.

There is no horizontal connectivity between columns; DSS and horizontal
spectral-element operators are not supported.
"""
struct MultiColumnFiniteDifferenceSpace{
    G <: Grids.AbstractExtrudedFiniteDifferenceGrid,
    S <: Staggering,
} <: AbstractSpace
    grid::G
    staggering::S
end

local_geometry_type(::Type{MultiColumnFiniteDifferenceSpace{G, S}}) where {G, S} =
    local_geometry_type(G)

grid(space::MultiColumnFiniteDifferenceSpace) = getfield(space, :grid)
staggering(space::MultiColumnFiniteDifferenceSpace) = getfield(space, :staggering)

const FaceMultiColumnFiniteDifferenceSpace{G} =
    MultiColumnFiniteDifferenceSpace{G, CellFace}
const CenterMultiColumnFiniteDifferenceSpace{G} =
    MultiColumnFiniteDifferenceSpace{G, CellCenter}

# Convenience constructors mirroring CenterExtrudedFiniteDifferenceSpace(space)
FaceMultiColumnFiniteDifferenceSpace(space::MultiColumnFiniteDifferenceSpace) =
    MultiColumnFiniteDifferenceSpace(grid(space), CellFace())
CenterMultiColumnFiniteDifferenceSpace(space::MultiColumnFiniteDifferenceSpace) =
    MultiColumnFiniteDifferenceSpace(grid(space), CellCenter())

# Override the generic `space(refspace::AbstractSpace, staggering) = space(grid(refspace), staggering)`
# so that we return MultiColumnFiniteDifferenceSpace rather than ExtrudedFiniteDifferenceSpace.
space(refspace::MultiColumnFiniteDifferenceSpace, s::Staggering) =
    MultiColumnFiniteDifferenceSpace(grid(refspace), s)

function face_space(space::MultiColumnFiniteDifferenceSpace)
    MultiColumnFiniteDifferenceSpace(grid(space), CellFace())
end
function center_space(space::MultiColumnFiniteDifferenceSpace)
    MultiColumnFiniteDifferenceSpace(grid(space), CellCenter())
end

Adapt.adapt_structure(to, space::MultiColumnFiniteDifferenceSpace) =
    MultiColumnFiniteDifferenceSpace(
        Adapt.adapt(to, grid(space)),
        staggering(space),
    )

issubspace(space1::PointCloudSpace, space2::PointCloudSpace) =
    horizontal_grid(grid(space1)) === horizontal_grid(grid(space2))
issubspace(subspace::PointCloudSpace, space::MultiColumnFiniteDifferenceSpace) =
    grid(subspace) === grid(space).horizontal_grid ||
    (grid(subspace) isa Grids.LevelGrid && grid(subspace).full_grid === grid(space))
issubspace(subspace::FiniteDifferenceSpace, space::MultiColumnFiniteDifferenceSpace) =
    grid(subspace) === grid(space).vertical_grid ||
    (grid(subspace) isa Grids.ColumnGrid && grid(subspace).full_grid === grid(space))

level(space::PointCloudSpace, v) =
    isone(v) ? space : throw(ArgumentError("Space only has one level"))
Base.@propagate_inbounds level(space::MultiColumnFiniteDifferenceSpace, v) =
    PointCloudSpace(level(grid(space), staggered_level_index(space, v)))

Base.@propagate_inbounds slab(space::PointCloudSpace, v, h) =
    isone(v) ? slab(space, h) : throw(ArgumentError("Space only has one level"))
Base.@propagate_inbounds slab(space::PointCloudSpace, h) =
    PointSpace(ClimaComms.context(space), slab(local_geometry_data(space), h))
Base.@propagate_inbounds slab(space::MultiColumnFiniteDifferenceSpace, v, h) =
    PointSpace(
        ClimaComms.context(space),
        slab(local_geometry_data(space), integer_level_index(space, v), h),
    )

Base.@propagate_inbounds column(space::PointCloudSpace, indices...) =
    PointSpace(ClimaComms.context(space), column(local_geometry_data(space), indices...))
Base.@propagate_inbounds column(space::MultiColumnFiniteDifferenceSpace, indices...) =
    FiniteDifferenceSpace(column(grid(space), indices...), space.staggering)

ncolumns(space::PointCloudSpace) =
    DataLayouts.nelems(local_geometry_data(space))

ncolumns(space::MultiColumnFiniteDifferenceSpace) =
    DataLayouts.nelems(
        Grids.local_geometry_data(grid(space).horizontal_grid, nothing),
    )

nlevels(space::MultiColumnFiniteDifferenceSpace) =
    DataLayouts.nlevels(local_geometry_data(space))

horizontal_space(space::MultiColumnFiniteDifferenceSpace) =
    PointCloudSpace(grid(space).horizontal_grid)

# No DSS / mask machinery needed.
get_mask(space::PointCloudSpace) = DataLayouts.NoMask()
get_mask(space::MultiColumnFiniteDifferenceSpace) = DataLayouts.NoMask()
set_mask!(::Any, ::MultiColumnFiniteDifferenceSpace) = nothing

"""
    obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace)

Return the [`PointCloudSpace`](@ref) corresponding to the top face (surface) of
`cs`.
"""
obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace) =
    horizontal_space(cs)

function Base.show(io::IO, space::PointCloudSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, "PointCloudSpace:")
    print(iio, " "^(indent + 2), "context: ")
    Topologies.print_context(iio, ClimaComms.context(space))
    println(iio)
    print(iio, " "^(indent + 2), "points: ", ncolumns(space))
end

function Base.show(io::IO, space::MultiColumnFiniteDifferenceSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(
        io,
        space isa CenterMultiColumnFiniteDifferenceSpace ?
        "CenterMultiColumnFiniteDifferenceSpace" :
        "FaceMultiColumnFiniteDifferenceSpace",
        ":",
    )
    print(iio, " "^(indent + 2), "context: ")
    Topologies.print_context(iio, ClimaComms.context(space))
    println(iio)
    println(iio, " "^(indent + 2), "columns: ", ncolumns(space))
    print(iio, " "^(indent + 2), "levels:  ", nlevels(space))
end
