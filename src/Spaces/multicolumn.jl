"""
    PointCloudSpace

A horizontal space of N independent points (lat, lon, z), produced by taking a
vertical level of a [`MultiColumnFiniteDifferenceSpace`](@ref).  This is the
N-column analogue of [`PointSpace`](@ref), which is the single-column level
space.

`full_grid` references the extruded grid the level was taken from (analogous
to `Grids.LevelGrid.full_grid`), and `local_geometry` holds an `IFH{LG, 1, N}`
data layout.
"""
struct PointCloudSpace{
    C <: ClimaComms.AbstractCommsContext,
    G,
    LG,
} <: AbstractSpace
    context::C
    full_grid::G  # the AbstractExtrudedFiniteDifferenceGrid this level was sliced from
    local_geometry::LG  # IFH{FullLocalGeometry{(1,2,3), LatLongZPoint, ...}, 1, N}
end

ClimaComms.context(space::PointCloudSpace) = space.context
ClimaComms.device(space::PointCloudSpace) =
    ClimaComms.device(space.context)

local_geometry_data(space::PointCloudSpace) = space.local_geometry
local_geometry_type(::Type{PointCloudSpace{C, G, LG}}) where {C, G, LG} =
    eltype(LG)

# PointCloudSpace has no grid/staggering; override the generic
# local_geometry_data(space) = local_geometry_data(grid(space), staggering(space))
# so that it works without them.
local_geometry_data(space::PointCloudSpace, ::Nothing) =
    space.local_geometry

"""
    MultiColumnFiniteDifferenceSpace

A space of N independent vertical columns at arbitrary horizontal (lat, lon)
locations on a sphere.  This is the N-column generalisation of
[`Spaces.FiniteDifferenceSpace`](@ref) (the single-column space):

- The data layout is `VIFH{LG, Nv, 1, N}` (same vertical structure for every
  column; full 3-D local geometry including lat/lon/z coordinates).
- [`Spaces.level`](@ref) returns a [`PointCloudSpace`](@ref) (N points
  at that z-level) rather than a spectral-element horizontal space.
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

# ---- column / level extraction ----------------------------------------

"""
    column(space::MultiColumnFiniteDifferenceSpace, colidx::Grids.ColumnIndex)

Return a single-column [`FiniteDifferenceSpace`](@ref) for column `colidx`.
"""
function column(
    space::MultiColumnFiniteDifferenceSpace,
    colidx::Grids.ColumnIndex,
)
    column_grid = Grids.column(grid(space), colidx)
    FiniteDifferenceSpace(column_grid, space.staggering)
end
column(space::MultiColumnFiniteDifferenceSpace, i, h) =
    column(space, Grids.ColumnIndex((i,), h))
column(space::MultiColumnFiniteDifferenceSpace, i, j, h) =
    column(space, Grids.ColumnIndex((i,), h))

"""
    column(space::PointCloudSpace, i, h)

Return the single-point [`PointSpace`](@ref) for column `h` of a
[`PointCloudSpace`](@ref).  This is the analogue of
`column(::SpectralElementSpace1D, i, h)` and enables `field[colidx]` indexing
inside [`Fields.bycolumn`](@ref) loops over a
[`MultiColumnFiniteDifferenceSpace`](@ref).
"""
Base.@propagate_inbounds function column(space::PointCloudSpace, i, h)
    local_geometry = column(local_geometry_data(space), i, h)
    PointSpace(ClimaComms.context(space), local_geometry)
end
Base.@propagate_inbounds column(space::PointCloudSpace, i, j, h) =
    column(space, i, h)

"""
    level(space::MultiColumnFiniteDifferenceSpace, v)

Return the [`PointCloudSpace`](@ref) (N-point horizontal slice) at
vertical level `v`.
"""
Base.@propagate_inbounds function level(
    space::CenterMultiColumnFiniteDifferenceSpace,
    v::Int,
)
    data = level(local_geometry_data(space), v)
    PointCloudSpace(ClimaComms.context(space), grid(space), data)
end
Base.@propagate_inbounds function level(
    space::FaceMultiColumnFiniteDifferenceSpace,
    v::PlusHalf,
)
    data = level(local_geometry_data(space), v.i + 1)
    PointCloudSpace(ClimaComms.context(space), grid(space), data)
end

# ---- space properties --------------------------------------------------

ncolumns(space::MultiColumnFiniteDifferenceSpace) =
    size(Grids.local_geometry_data(grid(space).horizontal_grid, nothing), 5)  # Nh dim of IFH

nlevels(space::MultiColumnFiniteDifferenceSpace) =
    size(local_geometry_data(space), 4)  # Nv dim of VIFH

horizontal_space(space::MultiColumnFiniteDifferenceSpace) =
    level(MultiColumnFiniteDifferenceSpace(grid(space), CellCenter()), 1)

# No DSS / mask machinery needed.
get_mask(space::PointCloudSpace) = DataLayouts.NoMask()
get_mask(space::MultiColumnFiniteDifferenceSpace) = DataLayouts.NoMask()
set_mask!(::Any, ::MultiColumnFiniteDifferenceSpace) = nothing

# A PointCloudSpace is a subspace of a MultiColumnFiniteDifferenceSpace
# when it was sliced from the same underlying grid.  This enables broadcasting
# a level field (IFH) across a full multi-column field (VIFH), analogous to
# issubspace(::SpectralElementSpace2D{<:LevelGrid}, ::ExtrudedFiniteDifferenceSpace).
function issubspace(
    level_space::PointCloudSpace,
    full_space::MultiColumnFiniteDifferenceSpace,
)
    return level_space.full_grid === grid(full_space)
end

"""
    obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace)

Return the [`PointCloudSpace`](@ref) corresponding to the top face
(surface) of `cs`.  Mirrors the single-column
`obtain_surface_space(::CenterFiniteDifferenceSpace)` pattern.
"""
function obtain_surface_space(cs::CenterMultiColumnFiniteDifferenceSpace)
    fs = face_space(cs)
    return level(fs, PlusHalf(nlevels(fs) - 1))
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
    println(iio, " "^(indent + 2), "levels:  ", nlevels(space))
end
