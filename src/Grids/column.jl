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
    G <: AbstractExtrudedFiniteDifferenceGrid,
    I <: Tuple{Vararg{Integer}},
} <: AbstractFiniteDifferenceGrid
    full_grid::G
    indices::I
end

Adapt.@adapt_structure ColumnGrid

local_geometry_type(::Type{<:ColumnGrid{G}}) where {G} = local_geometry_type(G)

column(grid::AbstractExtrudedFiniteDifferenceGrid, indices...) = ColumnGrid(grid, indices)

topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)
vertical_topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)

local_geometry_data(colgrid::ColumnGrid, staggering::Staggering) =
    column(local_geometry_data(colgrid.full_grid, staggering), colgrid.indices...)
global_geometry(colgrid::ColumnGrid) = global_geometry(colgrid.full_grid)
