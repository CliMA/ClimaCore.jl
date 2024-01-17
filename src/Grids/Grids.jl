module Grids

import ClimaComms, Adapt, ForwardDiff, LinearAlgebra
import LinearAlgebra: det, norm
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

function vertical_topology end

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

function local_dss_weights end
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



end # module
