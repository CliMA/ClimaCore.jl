struct LevelGrid{
    G <: AbstractExtrudedFiniteDifferenceGrid,
    L <: Union{Int, PlusHalf{Int}},
} <: AbstractGrid
    full_grid::G
    level::L
end

quadrature_style(levelgrid::LevelGrid) = quadrature_style(levelgrid.full_grid)

level(
    grid::AbstractExtrudedFiniteDifferenceGrid,
    level::Union{Int, PlusHalf{Int}},
) = LevelGrid(grid, level)

topology(levelgrid::LevelGrid) = topology(levelgrid.full_grid)

# The DSS weights for extruded spaces are currently the same as the weights for
# horizontal spaces. If we ever need to use extruded weights, this method will
# need to extract the weights at a particular level.
dss_weights(levelgrid::LevelGrid, _) = dss_weights(levelgrid.full_grid, nothing)

local_geometry_type(::Type{LevelGrid{G, L}}) where {G, L} =
    local_geometry_type(G)

local_geometry_data(levelgrid::LevelGrid{<:Any, Int}, ::Nothing) = level(
    local_geometry_data(levelgrid.full_grid, CellCenter()),
    levelgrid.level,
)
local_geometry_data(levelgrid::LevelGrid{<:Any, PlusHalf{Int}}, ::Nothing) =
    level(
        local_geometry_data(levelgrid.full_grid, CellFace()),
        levelgrid.level + half,
    )
global_geometry(levelgrid::LevelGrid) = global_geometry(levelgrid.full_grid)

## GPU compatibility
Adapt.adapt_structure(to, grid::LevelGrid) =
    LevelGrid(Adapt.adapt(to, grid.full_grid), grid.level)

## aliases
const LevelCubedSphereSpectralElementGrid2D =
    LevelGrid{<:ExtrudedCubedSphereSpectralElementGrid3D}
const LevelRectilinearSpectralElementGrid2D =
    LevelGrid{<:ExtrudedRectilinearSpectralElementGrid3D}
