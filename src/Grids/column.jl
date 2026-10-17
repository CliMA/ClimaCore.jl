"""
    ColumnIndex(ij, h)

Index into a column of a field. Passing a `ColumnIndex` to `getindex` of a `Field`
returns the field restricted to that column.

# Fields

  - `ij`: Tuple of horizontal nodal indices within the element.
  - `h`: Horizontal element index.

# Examples

```julia
colidx = ColumnIndex((1, 1), 1)
field[colidx]
```
"""
struct ColumnIndex{N}
    ij::NTuple{N, Int}
    h::Int
end


"""
    ColumnGrid(full_grid, indices)

View of the column at `indices` of the extruded grid `full_grid`.
"""
struct ColumnGrid{
    G <: Union{AbstractExtrudedFiniteDifferenceGrid, PlaceholderGrid},
    I <: Tuple{Vararg{Integer}},
} <: AbstractFiniteDifferenceGrid
    full_grid::G
    indices::I
end

Adapt.@adapt_structure ColumnGrid

local_geometry_type(::Type{<:ColumnGrid{G}}) where {G} = local_geometry_type(G)

# Throw for out-of-range indices instead of reading another column (slice loops skip this).
# Compare extents rather than slice (a view per call); throw with integers only, since data
# passed to a non-inlined function makes a kernel copy all its arguments to local memory.
Base.@propagate_inbounds function column(
    grid::AbstractExtrudedFiniteDifferenceGrid,
    indices...,
)
    @boundscheck check_column_indices(local_geometry_data(grid, CellCenter()), indices)
    return unchecked_column(grid, indices...)
end
@inline unchecked_column(grid::AbstractExtrudedFiniteDifferenceGrid, indices...) =
    ColumnGrid(grid, indices)
# Columns of grids with one horizontal dimension are indexed by (i, h), i.e., (i, 1, h).
@inline full_column_indices((i, h)::NTuple{2, Integer}) = (i, 1, h)
@inline full_column_indices(indices) = indices
@inline function check_column_indices(data, indices)
    (i, j, h) = full_column_indices(indices)
    (; Ni, Nj) = DataLayouts.vijh_params(data)
    Nh = DataLayouts.nelems(data)
    (1 <= i <= Ni) & (1 <= j <= Nj) & (1 <= h <= Nh) ||
        throw_column_indices((Ni, Nj, Nh), (i, j, h))
    return nothing
end
@noinline throw_column_indices(extents, indices) =
    throw(BoundsError(CartesianIndices(extents), indices))

topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)
vertical_topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)

# A column grid's indices are valid for its full grid, so kernels need not repeat the check.
local_geometry_data(colgrid::ColumnGrid, staggering::Staggering) =
    @inbounds column(local_geometry_data(colgrid.full_grid, staggering), colgrid.indices...)
global_geometry(colgrid::ColumnGrid) = global_geometry(colgrid.full_grid)

issubgrid(subgrid::ColumnGrid, grid::ColumnGrid) = subgrid === grid
maybe_issubgrid(subgrid::ColumnGrid, grid::ColumnGrid) = true
for f in (:issubgrid, :maybe_issubgrid)
    @eval $f(subgrid::ColumnGrid, grid::AbstractGrid) = $f(subgrid.full_grid, grid)
    @eval $f(subgrid::AbstractGrid, grid::ColumnGrid) =
        grid.full_grid isa DeviceExtrudedFiniteDifferenceGrid ?
        throw(ArgumentError("Cannot compare device-side slices of extruded grids")) :
        $f(subgrid, grid.full_grid.vertical_grid)
end
