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
    CLG,
    FLG,
} <: AbstractExtrudedFiniteDifferenceGrid
    horizontal_grid::H
    vertical_grid::V
    hypsography::A
    global_geometry::GG
    center_local_geometry::CLG
    face_local_geometry::FLG
end

Adapt.@adapt_structure ExtrudedFiniteDifferenceGrid

local_geometry_type(
    ::Type{ExtrudedFiniteDifferenceGrid{H, V, A, GG, CLG, FLG}},
) where {H, V, A, GG, CLG, FLG} = eltype(CLG) # calls eltype from DataLayouts

function ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Union{SpectralElementGrid1D, SpectralElementGrid2D},
    vertical_grid::FiniteDifferenceGrid,
    hypsography::HypsographyAdaption = Flat();
    deep = false,
)
    if horizontal_grid.global_geometry isa Geometry.SphericalGlobalGeometry
        radius = horizontal_grid.global_geometry.radius
        if deep
            global_geometry = Geometry.DeepSphericalGlobalGeometry(radius)
        else
            global_geometry = Geometry.ShallowSphericalGlobalGeometry(radius)
        end
    else
        global_geometry = horizontal_grid.global_geometry
    end
    ExtrudedFiniteDifferenceGrid(
        horizontal_grid,
        vertical_grid,
        hypsography,
        global_geometry,
    )
end

# memoized constructor
function ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Union{SpectralElementGrid1D, SpectralElementGrid2D},
    vertical_grid::FiniteDifferenceGrid,
    hypsography::HypsographyAdaption,
    global_geometry::Geometry.AbstractGlobalGeometry,
)
    get!(
        Cache.OBJECT_CACHE,
        (
            ExtrudedFiniteDifferenceGrid,
            horizontal_grid,
            vertical_grid,
            hypsography,
            global_geometry,
        ),
    ) do
        _ExtrudedFiniteDifferenceGrid(
            horizontal_grid,
            vertical_grid,
            hypsography,
            global_geometry,
        )
    end
end

# Non-memoized constructor. Should not generally be called, but can be defined for other Hypsography types
function _ExtrudedFiniteDifferenceGrid(
    horizontal_grid::Union{SpectralElementGrid1D, SpectralElementGrid2D},
    vertical_grid::FiniteDifferenceGrid,
    hypsography::Flat,
    global_geometry::Geometry.AbstractGlobalGeometry,
)
    center_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.center_local_geometry,
            Ref(global_geometry),
        )
    face_local_geometry =
        Geometry.product_geometry.(
            horizontal_grid.local_geometry,
            vertical_grid.face_local_geometry,
            Ref(global_geometry),
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

# Since ∂z/∂ξ₃ and r are continuous across element boundaries, we can reuse
# the horizontal weights instead of calling compute_dss_weights on the
# extruded local geometry.
dss_weights(grid::AbstractExtrudedFiniteDifferenceGrid, ::Staggering) =
    dss_weights(grid.horizontal_grid, nothing)

local_geometry_data(grid::AbstractExtrudedFiniteDifferenceGrid, ::CellCenter) =
    grid.center_local_geometry
local_geometry_data(grid::AbstractExtrudedFiniteDifferenceGrid, ::CellFace) =
    grid.face_local_geometry
global_geometry(grid::AbstractExtrudedFiniteDifferenceGrid) =
    grid.global_geometry

quadrature_style(grid::ExtrudedFiniteDifferenceGrid) =
    quadrature_style(grid.horizontal_grid)


## GPU compatibility
struct DeviceExtrudedFiniteDifferenceGrid{VT, Q, GG, CLG, FLG} <:
       AbstractExtrudedFiniteDifferenceGrid
    vertical_topology::VT
    quadrature_style::Q
    global_geometry::GG
    center_local_geometry::CLG
    face_local_geometry::FLG
end

local_geometry_type(
    ::Type{DeviceExtrudedFiniteDifferenceGrid{VT, Q, GG, CLG, FLG}},
) where {VT, Q, GG, CLG, FLG} = eltype(CLG) # calls eltype from DataLayouts

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
