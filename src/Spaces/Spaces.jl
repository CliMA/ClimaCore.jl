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

import ..slab, ..column, ..level
import ..Utilities: PlusHalf
import ..DataLayouts, ..Geometry, ..Domains, ..Meshes, ..Topologies
import ..Device
using StaticArrays, ForwardDiff, LinearAlgebra, UnPack

abstract type AbstractSpace end

undertype(space::AbstractSpace) =
    Geometry.undertype(eltype(local_geometry_data(space)))

coordinates_data(space::AbstractSpace) = local_geometry_data(space).coordinates

include("quadrature.jl")
import .Quadratures

include("pointspace.jl")
include("spectralelement.jl")
include("finitedifference.jl")
include("extruded.jl")
include("triangulation.jl")
include("dss.jl")
include("dss2.jl")

horizontal_space(space::ExtrudedFiniteDifferenceSpace) = space.horizontal_space
horizontal_space(space::AbstractSpace) = space

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
    ClimaComms.allreduce(comm_context(space), local_area(space), +)


end # module
