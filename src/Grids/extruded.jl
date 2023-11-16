#####
##### Hybrid grid
#####

abstract type HypsographyAdaption end

"""
    Flat()

No surface hypsography.
"""
struct Flat <: HypsographyAdaption end

abstract type AbstractExtrudedFiniteDifferenceGrid <: AbstractGrid end

"""
    ExtrudedFiniteDifferenceGrid(
        horizontal_space::AbstractSpace,
        vertical_space::FiniteDifferenceSpace,
        hypsography::HypsographyAdaption = Flat(),
    )

Construct an `ExtrudedFiniteDifferenceGrid` from the horizontal and vertical spaces.
"""
mutable struct ExtrudedFiniteDifferenceGrid{
    H <: AbstractGrid,
    V <: FiniteDifferenceGrid,
    A <: HypsographyAdaption,
    GG <: Geometry.AbstractGlobalGeometry,
    LG,
} <: AbstractExtrudedFiniteDifferenceGrid
    horizontal_grid::H
    vertical_grid::V
    hypsography::A
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end

@memoize WeakValueDict function ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Union{SpectralElementGrid1D, SpectralElementGrid2D},
    vertical_grid::FiniteDifferenceGrid,
    hypsography::Flat = Flat(),
)
    global_geometry = horizontal_grid.global_geometry
    center_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.center_local_geometry,
        )
    face_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.face_local_geometry,
        )

    return ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        hypsography,
        global_geometry,
        center_local_geometry,
        face_local_geometry,
    )
end

topology(grid::ExtrudedFiniteDifferenceGrid) = topology(grid.horizontal_grid)

vertical_topology(grid::ExtrudedFiniteDifferenceGrid) =
    topology(grid.vertical_grid)

local_dss_weights(grid::ExtrudedFiniteDifferenceGrid) =
    local_dss_weights(grid.horizontal_grid)


local_geometry_data(grid::AbstractExtrudedFiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::AbstractExtrudedFiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::AbstractExtrudedFiniteDifferenceGrid) =
    grid.global_geometry

quadrature_style(grid::ExtrudedFiniteDifferenceGrid) =
    quadrature_style(grid.horizontal_grid)


## GPU compatibility
struct DeviceExtrudedFiniteDifferenceGrid{VT, Q, GG, LG} <:
       AbstractExtrudedFiniteDifferenceGrid
    vertical_topology::VT
    quadrature_style::Q
    global_geometry::GG
    center_local_geometry::LG
    face_local_geometry::LG
end

Adapt.adapt_structure(to, grid::ExtrudedFiniteDifferenceGrid) =
    DeviceExtrudedFiniteDifferenceGrid(
        Adapt.adapt(to, vertical_topology(grid)),
        Adapt.adapt(to, grid.horizontal_grid.quadrature_style),
        Adapt.adapt(to, grid.global_geometry),
        Adapt.adapt(to, grid.center_local_geometry),
        Adapt.adapt(to, grid.face_local_geometry),
    )

quadrature_style(grid::DeviceExtrudedFiniteDifferenceGrid) =
    grid.quadrature_style
vertical_topology(grid::DeviceExtrudedFiniteDifferenceGrid) =
    grid.vertical_topology

## aliases

const ExtrudedSpectralElementGrid2D =
    ExtrudedFiniteDifferenceGrid{<:SpectralElementGrid1D}
const ExtrudedSpectralElementGrid3D =
    ExtrudedFiniteDifferenceGrid{<:SpectralElementGrid2D}
const ExtrudedRectilinearSpectralElementGrid3D =
    ExtrudedFiniteDifferenceGrid{<:RectilinearSpectralElementGrid2D}
const ExtrudedCubedSphereSpectralElementGrid3D =
    ExtrudedFiniteDifferenceGrid{<:CubedSphereSpectralElementGrid2D}
