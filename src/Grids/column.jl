



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
    ColumnViewGrid(
        full_grid :: ExtrudedFiniteDifferenceGrid, 
        colidx    :: ColumnIndex,
    )

A view into a column of a `ExtrudedFiniteDifferenceGrid`. This can be used as an
"""
struct ColumnViewGrid{
    G <: AbstractExtrudedFiniteDifferenceGrid,
    C <: ColumnIndex,
} <: AbstractFiniteDifferenceGrid
    full_grid::G
    colidx::C
end

local_geometry_type(::Type{ColumnGrid{G, C}}) where {G, C} =
    local_geometry_type(G)

column(grid::AbstractExtrudedFiniteDifferenceGrid, colidx::ColumnIndex) =
    ColumnViewGrid(grid, colidx)

topology(colgrid::ColumnViewGrid) = vertical_topology(colgrid.full_grid)
vertical_topology(colgrid::ColumnViewGrid) =
    vertical_topology(colgrid.full_grid)

local_geometry_data(colgrid::ColumnViewGrid, staggering::Staggering) = column(
    local_geometry_data(colgrid.full_grid, staggering::Staggering),
    colgrid.colidx.ij...,
    colgrid.colidx.h,
)
global_geometry(colgrid::ColumnViewGrid) = global_geometry(colgrid.full_grid)
