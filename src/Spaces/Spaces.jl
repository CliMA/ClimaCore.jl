"""
    Meshes

- domain
- topology
- coordinates
- metric terms (inverse partial derivatives)
- quadrature rules and weights

## References / notes
 - [ceed](https://ceed.exascaleproject.org/ceed-code/)
 - [QA](https://github.com/CliMA/ClimateMachine.jl/blob/ans/sphere/test/Numerics/DGMethods/compressible_navier_stokes_equations/sphere/sphere_helper_functions.jl)

"""
module Spaces

using ClimaComms
using Adapt

import ..slab, ..column, ..level
import ..Utilities: PlusHalf, half
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts,
    ..Geometry, ..Domains, ..Meshes, ..Topologies, ..Grids, ..Quadratures

import ..Domains: z_max, z_min
import ..Meshes: n_elements_per_panel_direction

import ..DeviceSideDevice, ..DeviceSideContext

import ..Grids:
    Staggering,
    CellFace,
    CellCenter,
    topology,
    vertical_topology,
    local_geometry_type,
    local_geometry_data,
    global_geometry,
    local_dss_weights,
    quadrature_style

import ClimaComms
using StaticArrays, ForwardDiff, LinearAlgebra, Adapt

"""
    AbstractSpace

Should define
- `grid`
- `staggering`


- `space` constructor

"""
abstract type AbstractSpace end

function grid end
function staggering end


ClimaComms.context(space::AbstractSpace) = ClimaComms.context(grid(space))
ClimaComms.device(space::AbstractSpace) = ClimaComms.device(grid(space))

topology(space::AbstractSpace) = topology(grid(space))
vertical_topology(space::AbstractSpace) = vertical_topology(grid(space))


local_geometry_data(space::AbstractSpace) =
    local_geometry_data(grid(space), staggering(space))

function n_elements_per_panel_direction(space::AbstractSpace)
    hspace = Spaces.horizontal_space(space)
    hmesh = topology(hspace).mesh
    return Meshes.n_elements_per_panel_direction(hmesh)
end

global_geometry(space::AbstractSpace) = global_geometry(grid(space))

space(refspace::AbstractSpace, staggering::Staggering) =
    space(grid(refspace), staggering)

issubspace(::AbstractSpace, ::AbstractSpace) = false

undertype(space::AbstractSpace) =
    Geometry.undertype(eltype(local_geometry_data(space)))

coordinates_data(space::AbstractSpace) = local_geometry_data(space).coordinates
coordinates_data(grid::Grids.AbstractGrid) =
    local_geometry_data(grid).coordinates
coordinates_data(staggering, grid::Grids.AbstractGrid) =
    local_geometry_data(staggering, grid).coordinates

include("pointspace.jl")
include("spectralelement.jl")
include("finitedifference.jl")
include("extruded.jl")
include("triangulation.jl")
include("dss.jl")


function center_space(space::AbstractSpace)
    error("`center_space` can only be called with vertical/extruded spaces")
end

function face_space(space::AbstractSpace)
    error("`center_space` can only be called with vertical/extruded spaces")
end

weighted_jacobian(space::Spaces.AbstractSpace) = local_geometry_data(space).WJ

"""
    Spaces.local_area(space::Spaces.AbstractSpace)

The length/area/volume of `space` local to the current context. See
[`Spaces.area`](@ref)
"""
local_area(space::Spaces.AbstractSpace) = Base.sum(weighted_jacobian(space))

"""
    Spaces.area(space::Spaces.AbstractSpace)

The length/area/volume of `space`. This is computed as the sum of the quadrature
weights ``W_i`` multiplied by the Jacobian determinants ``J_i``:
```math
\\sum_i W_i J_i \\approx \\int_\\Omega \\, d \\Omega
```

If `space` is distributed, this uses a `ClimaComms.allreduce` operation.
"""
area(space::Spaces.AbstractSpace) =
    ClimaComms.allreduce(ClimaComms.context(space), local_area(space), +)

ClimaComms.array_type(space::AbstractSpace) =
    ClimaComms.array_type(ClimaComms.device(space))

"""
    z_max(::AbstractSpace)

The domain maximum along the z-direction.
"""
function z_max(space::Spaces.AbstractSpace)
    mesh = Topologies.mesh(Spaces.vertical_topology(space))
    domain = Topologies.domain(mesh)
    return Domains.z_max(domain)
end

"""
    z_min(::AbstractSpace)

The domain minimum along the z-direction.
"""
function z_min(space::Spaces.AbstractSpace)
    mesh = Topologies.mesh(Spaces.vertical_topology(space))
    domain = Topologies.domain(mesh)
    return Domains.z_min(domain)
end

end # module
