struct LevelGrid{
    Q,
    G <: AbstractExtrudedFiniteDifferenceGrid,
    L <: Union{Int, PlusHalf{Int}},
} <: AbstractGrid
    quadrature_style::Q
    full_grid::G
    level::L
end

quadrature_style(levelgrid::LevelGrid) = levelgrid.quadrature_style

level(
    grid::AbstractExtrudedFiniteDifferenceGrid,
    level::Union{Int, PlusHalf{Int}},
) = LevelGrid(quadrature_style(grid.horizontal_grid), grid, level)

topology(levelgrid::LevelGrid) = topology(levelgrid.full_grid)

# The DSS weights for extruded spaces are currently the same as the weights for
# horizontal spaces. If we ever need to use extruded weights, this method will
# need to extract the weights at a particular level.
dss_weights(levelgrid::LevelGrid, _) = dss_weights(levelgrid.full_grid, nothing)

local_geometry_type(::Type{LevelGrid{Q, G, L}}) where {Q, G, L} =
    local_geometry_type(G)
local_geometry_data(levelgrid::LevelGrid{<:Any, <:Any, Int}, ::Nothing) = level(
    local_geometry_data(levelgrid.full_grid, CellCenter()),
    levelgrid.level,
)
local_geometry_data(
    levelgrid::LevelGrid{<:Any, <:Any, PlusHalf{Int}},
    ::Nothing,
) = level(
    local_geometry_data(levelgrid.full_grid, CellFace()),
    levelgrid.level + half,
)
global_geometry(levlgrid::LevelGrid) = global_geometry(levlgrid.full_grid)

## GPU compatibility
Adapt.adapt_structure(to, grid::LevelGrid) = LevelGrid(
    grid.quadrature_style,
    Adapt.adapt(to, grid.full_grid),
    grid.level,
)

## aliases
const LevelCubedSphereSpectralElementGrid2D =
    LevelGrid{<:Any, <:ExtrudedCubedSphereSpectralElementGrid3D}
const LevelRectilinearSpectralElementGrid2D =
    LevelGrid{<:Any, <:ExtrudedRectilinearSpectralElementGrid3D}
