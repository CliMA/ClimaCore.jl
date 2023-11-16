struct LevelGrid{
    G <: AbstractExtrudedFiniteDifferenceGrid,
    L <: Union{Int, PlusHalf{Int}},
} <: AbstractGrid
    full_grid::G
    level::L
end

quadrature_style(levelgrid::LevelGrid) =
    quadrature_style(levelgrid.full_grid.horizontal_grid)

level(
    grid::AbstractExtrudedFiniteDifferenceGrid,
    level::Union{Int, PlusHalf{Int}},
) = LevelGrid(grid, level)

topology(levelgrid::LevelGrid) = topology(levelgrid.full_grid)

local_geometry_data(levelgrid::LevelGrid{<:Any, Int}, ::Nothing) = level(
    local_geometry_data(levelgrid.full_grid, CellCenter()),
    levelgrid.level,
)
local_geometry_data(levelgrid::LevelGrid{<:Any, PlusHalf{Int}}, ::Nothing) =
    level(
        local_geometry_data(levelgrid.full_grid, CellFace()),
        levelgrid.level + half,
    )
global_geometry(levlgrid::LevelGrid) = global_geometry(levlgrid.full_grid)

## GPU compatibility
Adapt.adapt_structure(to, grid::LevelGrid) =
    LevelGrid(Adapt.adapt(to, grid.full_grid), grid.level)
