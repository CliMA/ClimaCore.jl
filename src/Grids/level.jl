struct LevelGrid{
    G <: ExtrudedFiniteDifferenceGrid,
    L <: Union{Int, PlusHalf{Int}},
} <: AbstractGrid
    full_grid::G
    level::L
end

level(grid::ExtrudedFiniteDifferenceGrid, level::Union{Int, PlusHalf{Int}}) =
    LevelGrid(grid, level)

topology(levelgrid::LevelGrid) = topology(levelgrid.full_grid)

local_geometry_data(colgrid::LevelGrid{<:Any, Int}, ::Nothing) = level(
    local_geometry_data(levelgrid.full_grid, CellCenter()),
    levelgrid.level,
)
local_geometry_data(colgrid::LevelGrid{<:Any, PlusHalf{Int}}, ::Nothing) =
    level(
        local_geometry_data(levelgrid.full_grid, CellFace()),
        levelgrid.level + half,
    )
