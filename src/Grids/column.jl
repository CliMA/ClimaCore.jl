



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
struct ColumnGrid{
    VG <: FiniteDifferenceGrid,
    GG <: Geometry.AbstractGlobalGeometry,
    C <: ColumnIndex,
} <: AbstractFiniteDifferenceGrid
    vertical_grid::VG
    global_geometry::GG
    colidx::C
end

local_geometry_type(::Type{ColumnGrid{VG, C}}) where {VG, C} =
    local_geometry_type(VG)

function column(grid::AbstractExtrudedFiniteDifferenceGrid, colidx::ColumnIndex)
    ColumnGrid(grid.vertical_grid, grid.global_geometry, colidx)
end

topology(colgrid::ColumnGrid) = vertical_topology(colgrid.vertical_grid)
vertical_topology(colgrid::ColumnGrid) = vertical_topology(colgrid.vertical_grid)

local_geometry_data(colgrid::ColumnGrid, staggering::Staggering) = column(
    local_geometry_data(colgrid.vertical_grid, staggering::Staggering),
    colgrid.colidx.ij...,
    colgrid.colidx.h,
)
global_geometry(colgrid::ColumnGrid) = colgrid.global_geometry
