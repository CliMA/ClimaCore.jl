module Grids

import ClimaComms, Adapt, ForwardDiff, LinearAlgebra
import LinearAlgebra: det, norm
import ..DataLayouts, ..Domains, ..Meshes, ..Topologies, ..Geometry, ..Quadratures
import ..Utilities: PlusHalf, half, Cache
import ..slab, ..column, ..level
import ..DeviceSideDevice, ..DeviceSideContext

using StaticArrays

"""
    Grids.AbstractGrid

Abstract supertype of grids. Subtypes define the following methods:

  - [`topology`](@ref): the topology of the grid.
  - `ClimaComms.context` and `ClimaComms.device` (default to those of the topology).
  - `Meshes.domain` (defaults to that of the topology).
  - [`local_geometry_data`](@ref): the `DataLayout` object containing the local
    geometry of the grid.
"""
abstract type AbstractGrid end

"""
    Grids.topology(grid::AbstractGrid)

Return the topology of `grid`.
"""
function topology end

"""
    Grids.local_geometry_data(
        grid       :: AbstractGrid,
        staggering :: Union{Staggering, Nothing},
    )

Return the `DataLayout` object containing the local geometry of `grid` at the
given `staggering`.

If the grid is not staggered, `staggering` is `nothing`.
"""
function local_geometry_data end

"""
    Grids.local_geometry_type(::Type{<:AbstractGrid})

Return the `LocalGeometry` element type of a grid type. The fallback for
unrecognized types is `Union{}`.
"""
function local_geometry_type end

# Fallback, but this requires user error-handling
local_geometry_type(::Type{T}) where {T} = Union{}

function dss_weights end
function quadrature_style end
function vertical_topology end



ClimaComms.context(grid::AbstractGrid) = ClimaComms.context(topology(grid))
ClimaComms.device(grid::AbstractGrid) = ClimaComms.device(topology(grid))

Meshes.domain(grid::AbstractGrid) = Meshes.domain(topology(grid))

include("finitedifference.jl")
include("spectralelement.jl")
include("multipoint.jl")
include("extruded.jl")
include("column.jl")
include("level.jl")

function Base.show(io::IO, grid::AbstractGrid)
    indent = get(io, :indent, 0)
    iio = IOContext(io, :indent => indent + 2)
    println(io, nameof(typeof(grid)), ":")
    if has_horizontal(grid)
        # some reduced spaces (like slab space) do not have topology
        println(iio, " "^(indent + 2), "horizontal:")
        print(iio, " "^(indent + 4), "context: ")
        Topologies.print_context(iio, topology(grid).context)
        println(iio)
        println(iio, " "^(indent + 4), "mesh: ", topology(grid).mesh)
        print(iio, " "^(indent + 4), "quadrature: ", quadrature_style(grid))
    end
    if has_vertical(grid)
        has_horizontal(grid) && println(iio, "")
        println(iio, " "^(indent + 2), "vertical:")
        print(iio, " "^(indent + 4), "mesh: ", vertical_topology(grid).mesh)
    end
end

"""
    has_horizontal(::AbstractGrid)

Return `true` if the grid has a horizontal part.
"""
function has_horizontal end
has_horizontal(::AbstractGrid) = false
has_horizontal(::ExtrudedFiniteDifferenceGrid) = true
has_horizontal(::DeviceSpectralElementGrid2D) = true
has_horizontal(::SpectralElementGrid2D) = true
has_horizontal(::SpectralElementGrid1D) = true
has_horizontal(::MultiPointGrid) = true

"""
    has_vertical(::AbstractGrid)

Return `true` if the grid has a vertical part.
"""
function has_vertical end
has_vertical(::AbstractGrid) = false
has_vertical(::FiniteDifferenceGrid) = true
has_vertical(::ExtrudedFiniteDifferenceGrid) = true

"""
    get_mask(grid::AbstractGrid)

Return the mask of `grid`; `DataLayouts.NoMask()` for grids without a mask.
"""
get_mask(::AbstractGrid) = DataLayouts.NoMask()
get_mask(grid::ExtrudedFiniteDifferenceGrid) = grid.horizontal_grid.mask
get_mask(::ExtrudedFiniteDifferenceGrid{<:MultiPointGrid}) = DataLayouts.NoMask()

"""
    set_mask!(fn, grid)
    set_mask!(grid, data::DataLayouts.DataLayout)

Set the active-node mask of `grid`. With `fn`, the mask is `fn(coord)` evaluated at
every coordinate of the horizontal grid; with `data`, the mask is copied from
`data`. The mask maps are then rebuilt with `DataLayouts.set_mask_maps!`. Does
nothing if the grid mask is a `DataLayouts.NoMask`. Returns `nothing`.
"""
function set_mask! end

set_mask!(fn, grid::ExtrudedFiniteDifferenceGrid) =
    set_mask!(fn, grid.horizontal_grid)
function set_mask!(fn, grid::SpectralElementGrid2D)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = fn(grid.local_geometry.coordinates)
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

set_mask!(grid::ExtrudedFiniteDifferenceGrid, data::DataLayouts.DataLayout) =
    set_mask!(grid.horizontal_grid, data)
function set_mask!(grid::SpectralElementGrid2D, data::DataLayouts.DataLayout)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = data
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

end # module
