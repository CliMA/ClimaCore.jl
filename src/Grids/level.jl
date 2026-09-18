"""
    LevelGrid(full_grid::AbstractExtrudedFiniteDifferenceGrid, level)

Horizontal grid at a single vertical level of an extruded grid, as returned by
`level(grid, v)`: an integer `level` selects a cell center and a
`PlusHalf{Int}` selects a cell face. It shares the topology, quadrature, DSS
weights, and global geometry of `full_grid`, and its local geometry is the
corresponding level of the local geometry of `full_grid`.
"""
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

# A level of a discontinuous grid is discontinuous: without this the fallback
# reports `CG()`, and `Spaces.weighted_dss!` on a level field then builds a
# buffer and passes the grid's `nothing` DSS weights to `dss_transform!`.
discretization(levelgrid::LevelGrid) = discretization(levelgrid.full_grid)

ClimaComms.context(levelgrid::LevelGrid) = ClimaComms.context(levelgrid.full_grid)
ClimaComms.device(levelgrid::LevelGrid) = ClimaComms.device(levelgrid.full_grid)

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
hypsography(levelgrid::LevelGrid) = hypsography(levelgrid.full_grid)

issubgrid(subgrid::LevelGrid, grid::LevelGrid) = subgrid === grid
issubgrid(subgrid::LevelGrid, grid::AbstractGrid) =
    subgrid === grid || subgrid.full_grid === grid
issubgrid(subgrid::AbstractGrid, grid::LevelGrid) =
    subgrid === grid || issubgrid(subgrid, grid.full_grid)

## GPU compatibility
Adapt.adapt_structure(to, grid::LevelGrid) =
    LevelGrid(Adapt.adapt(to, grid.full_grid), grid.level)

## aliases
const LevelCubedSphereSpectralElementGrid2D =
    LevelGrid{<:ExtrudedCubedSphereSpectralElementGrid3D}
const LevelRectilinearSpectralElementGrid2D =
    LevelGrid{<:ExtrudedRectilinearSpectralElementGrid3D}
