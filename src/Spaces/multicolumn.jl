"""
    MultiPointSpace(grid)

Horizontal space of `N` independent (lat, lon) points. This is the `N`-column
analog of [`PointSpace`](@ref), which is the single-column level space.

Like [`SpectralElementSpace2D`](@ref), the wrapped `grid` is either

  - a `Grids.MultiPointGrid`, the level-agnostic horizontal grid of a
    [`MultiColumnFiniteDifferenceSpace`](@ref) (returned by
    `Spaces.horizontal_space`), or
  - a `Grids.LevelGrid` of the extruded multi-column grid at a single vertical
    level (returned by `Spaces.level`), carrying full 3D local geometry.
"""
struct MultiPointSpace{G <: Grids.AbstractGrid} <: AbstractSpace
    grid::G
end

grid(space::MultiPointSpace) = getfield(space, :grid)
staggering(space::MultiPointSpace) = nothing

space(grid::Grids.MultiPointGrid, ::Nothing) = MultiPointSpace(grid)
space(grid::Grids.LevelGrid{<:Grids.ExtrudedMultiPointGrid}, ::Nothing) =
    MultiPointSpace(grid)

ClimaComms.context(space::MultiPointSpace) =
    ClimaComms.context(grid(space))
ClimaComms.device(space::MultiPointSpace) = ClimaComms.device(grid(space))

local_geometry_data(space::MultiPointSpace) =
    local_geometry_data(grid(space), nothing)
local_geometry_type(::Type{MultiPointSpace{G}}) where {G} =
    local_geometry_type(G)

Adapt.adapt_structure(to, space::MultiPointSpace) =
    MultiPointSpace(Adapt.adapt(to, grid(space)))

# Deprecated alias of `MultiPointSpace`.
Base.@deprecate_binding PointCloudSpace MultiPointSpace false

"""
    MultiColumnFiniteDifferenceSpace(grid, staggering)

Space of `N` independent vertical columns at arbitrary horizontal (lat, lon)
locations on a sphere. This is the `N`-column generalization of
[`Spaces.FiniteDifferenceSpace`](@ref), the single-column space:

  - The data layout is `VIJFH{LG, Nv, 1, 1, N}`: the same vertical structure for
    every column, with full 3D local geometry including lat, lon, and z coordinates.
  - `Spaces.level` returns a [`MultiPointSpace`](@ref) (`N` points at that level)
    rather than a spectral element horizontal space.
  - `Spaces.column` returns a single-column [`Spaces.FiniteDifferenceSpace`](@ref).
  - `Fields.bycolumn` iterates over each column independently.

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

# Override the generic `space(refspace::AbstractSpace, staggering)`, which forwards to
# `space(grid(refspace), staggering)`, so that the result is a
# `MultiColumnFiniteDifferenceSpace` rather than an `ExtrudedFiniteDifferenceSpace`.
space(refspace::MultiColumnFiniteDifferenceSpace, s::Staggering) =
    MultiColumnFiniteDifferenceSpace(grid(refspace), s)
space(grid::Grids.ExtrudedMultiPointGrid, s::Staggering) =
    MultiColumnFiniteDifferenceSpace(grid, s)

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

issubspace(space1::MultiPointSpace, space2::MultiPointSpace) =
    horizontal_grid(grid(space1)) === horizontal_grid(grid(space2))
issubspace(subspace::MultiPointSpace, space::MultiColumnFiniteDifferenceSpace) =
    grid(subspace) === grid(space).horizontal_grid ||
    (grid(subspace) isa Grids.LevelGrid && grid(subspace).full_grid === grid(space))
issubspace(subspace::FiniteDifferenceSpace, space::MultiColumnFiniteDifferenceSpace) =
    grid(subspace) === grid(space).vertical_grid ||
    (grid(subspace) isa Grids.ColumnGrid && grid(subspace).full_grid === grid(space))

level(space::MultiPointSpace, v) =
    isone(v) ? space : throw(ArgumentError("Space only has one level"))
Base.@propagate_inbounds level(space::MultiColumnFiniteDifferenceSpace, v) =
    MultiPointSpace(level(grid(space), staggered_level_index(space, v)))

Base.@propagate_inbounds slab(space::MultiPointSpace, v, h) =
    isone(v) ? slab(space, h) : throw(ArgumentError("Space only has one level"))
Base.@propagate_inbounds slab(space::MultiPointSpace, h) =
    PointSpace(ClimaComms.context(space), slab(local_geometry_data(space), h))
Base.@propagate_inbounds slab(space::MultiColumnFiniteDifferenceSpace, v, h) =
    PointSpace(
        ClimaComms.context(space),
        slab(local_geometry_data(space), integer_level_index(space, v), h),
    )

Base.@propagate_inbounds column(space::MultiPointSpace, indices...) =
    PointSpace(ClimaComms.context(space), column(local_geometry_data(space), indices...))
Base.@propagate_inbounds column(space::MultiColumnFiniteDifferenceSpace, indices...) =
    FiniteDifferenceSpace(column(grid(space), indices...), space.staggering)

ncolumns(space::MultiPointSpace) =
    DataLayouts.nelems(local_geometry_data(space))

ncolumns(space::MultiColumnFiniteDifferenceSpace) =
    DataLayouts.nelems(
        Grids.local_geometry_data(grid(space).horizontal_grid, nothing),
    )

nlevels(space::MultiColumnFiniteDifferenceSpace) =
    DataLayouts.nlevels(local_geometry_data(space))

horizontal_space(space::MultiColumnFiniteDifferenceSpace) =
    MultiPointSpace(grid(space).horizontal_grid)
horizontal_space(space::MultiPointSpace) = space

# No DSS / mask machinery needed.
get_mask(space::MultiPointSpace) = DataLayouts.NoMask()
get_mask(space::MultiColumnFiniteDifferenceSpace) = DataLayouts.NoMask()
set_mask!(::Any, ::MultiColumnFiniteDifferenceSpace) = nothing

"""
    obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace)

Return the horizontal [`MultiPointSpace`](@ref) of `cs`, which serves as the
surface space of the columns.
"""
obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace) =
    horizontal_space(cs)

function Base.show(io::IO, space::MultiPointSpace)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, "MultiPointSpace:")
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
