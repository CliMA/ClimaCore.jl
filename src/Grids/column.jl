



"""
    ColumnIndex(ij,h)

An index into a column of a field. This can be used as an argument to `getindex`
of a `Field`, to return a field on that column.

# Example
```julia
colidx = ColumnIndex((1,1),1)
field[colidx]
```
"""
struct ColumnIndex{N}
    ij::NTuple{N, Int}
    h::Int
end


"""
    ColumnGrid(
        full_grid :: ExtrudedFiniteDifferenceGrid, 
        colidx    :: ColumnIndex,
    )

A view into a column of a `ExtrudedFiniteDifferenceGrid`. This can be used as an
"""
struct ColumnGrid{G <: ExtrudedFiniteDifferenceGrid, C <: ColumnIndex} <:
       AbstractFiniteDifferenceGrid
    full_grid::G
    colidx::C
end


column(grid::ExtrudedFiniteDifferenceGrid, colidx::ColumnIndex) =
    ColumnGrid(grid, colidx)

topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)
vertical_topology(colgrid::ColumnGrid) = vertical_topology(colgrid.full_grid)

local_geometry_data(colgrid::ColumnGrid, staggering::Staggering) = column(
    local_geometry_data(colgrid.full_grid, staggering::Staggering),
    colgrid.colidx,
)
global_geometry(colgrid::ColumnGrid) = global_geometry(colgrid.full_grid)
