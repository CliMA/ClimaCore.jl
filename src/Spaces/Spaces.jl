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
using CUDA

import ..slab, ..column, ..level
import ..Utilities: PlusHalf, half
import ..DataLayouts, ..Geometry, ..Domains, ..Meshes, ..Topologies
import ClimaComms
using StaticArrays, ForwardDiff, LinearAlgebra, UnPack, Adapt
using Memoize, WeakValueDicts

abstract type AbstractGrid end


abstract type AbstractSpace end

grid(space::AbstractSpace) = space.grid
staggering(space::AbstractSpace) = nothing

function space end


issubspace(::AbstractSpace, ::AbstractSpace) = false

undertype(space::AbstractSpace) =
    Geometry.undertype(eltype(local_geometry_data(space)))

coordinates_data(space::AbstractSpace) = local_geometry_data(space).coordinates
coordinates_data(grid::AbstractGrid) = local_geometry_data(grid).coordinates
coordinates_data(staggering, grid::AbstractGrid) =
    local_geometry_data(staggering, grid).coordinates

ClimaComms.context(space::Spaces.AbstractSpace) =
    ClimaComms.context(Spaces.topology(space))

include("quadrature.jl")
import .Quadratures

include("pointspace.jl")
include("spectralelement.jl")
include("finitedifference.jl")
include("extruded.jl")
include("triangulation.jl")
include("dss_transform.jl")
include("dss.jl")


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

end # module
