module Grids

import ClimaComms, Adapt, ForwardDiff, LinearAlgebra
import LinearAlgebra: det, norm
import ..DataLayouts: slab_index, vindex
import ..DataLayouts,
    ..Domains, ..Meshes, ..Topologies, ..Geometry, ..Quadratures
import ..Utilities: PlusHalf, half, Cache
import ..slab, ..column, ..level
import ..DeviceSideDevice, ..DeviceSideContext

using StaticArrays

"""
    Grids.AbstractGrid

Grids should define the following


- [`topology`](@ref): the topology of the grid
- [`mesh`](@ref): the mesh of the grid
- [`domain`](@ref): the domain of the grid
- `ClimaComms.context`
- `ClimaComms.device`

- [`local_geometry_data`](@ref): the `DataLayout` object containing the local geometry data information

"""
abstract type AbstractGrid end

"""
    Grids.topology(grid::AbstractGrid)

Get the topology of a grid.
"""
function topology end

"""
    Grids.local_geometry_data(
        grid       :: AbstractGrid,
        staggering :: Union{Staggering, Nothing},
    )

Get the `DataLayout` object containing the local geometry data information of
the `grid` with staggering `staggering`.

If the grid is not staggered, `staggering` should be `nothing`.
"""
function local_geometry_data end

"""
    Grids.local_geometry_type(::Type)

Get the `LocalGeometry` type.
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

Returns a bool indicating that the grid has a vertical part.
"""
function has_horizontal end
has_horizontal(::AbstractGrid) = false
has_horizontal(::ExtrudedFiniteDifferenceGrid) = true
has_horizontal(::DeviceSpectralElementGrid2D) = true
has_horizontal(::SpectralElementGrid2D) = true
has_horizontal(::SpectralElementGrid1D) = true

"""
    has_vertical(::AbstractGrid)

Returns a bool indicating that the space has a vertical part.
"""
function has_vertical end
has_vertical(::AbstractGrid) = false
has_vertical(::FiniteDifferenceGrid) = true
has_vertical(::ExtrudedFiniteDifferenceGrid) = true

"""
    get_mask(grid::AbstractGrid)

Retrieve the mask for the grid (defaults to DataLayouts.NoMask).
"""
get_mask(::AbstractGrid) = DataLayouts.NoMask()
get_mask(grid::ExtrudedFiniteDifferenceGrid) = grid.horizontal_grid.mask

"""
    set_mask!(fn::Function, grid)
    set_mask!(grid, ::DataLayouts.AbstractData)

Set the mask using the function `fn`, which is called for all coordinates on the
given grid.
"""
function set_mask! end

set_mask!(fn, grid::ExtrudedFiniteDifferenceGrid) =
    set_mask!(fn, grid.horizontal_grid)
function set_mask!(
    fn,
    grid::Union{SpectralElementGrid2D, ExtrudedFiniteDifferenceGrid},
)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = fn(grid.local_geometry.coordinates)
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

set_mask!(grid::ExtrudedFiniteDifferenceGrid, data::DataLayouts.AbstractData) =
    set_mask!(grid.horizontal_grid, data)
function set_mask!(grid::SpectralElementGrid2D, data::DataLayouts.AbstractData)
    if !(grid.mask isa DataLayouts.NoMask)
        @. grid.mask.is_active = data
        DataLayouts.set_mask_maps!(grid.mask)
    end
    return nothing
end

end # module
