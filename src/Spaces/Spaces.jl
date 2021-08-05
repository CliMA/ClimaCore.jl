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

import ..DataLayouts, ..Geometry, ..Domains, ..Meshes, ..Topologies
import ..slab
using StaticArrays, ForwardDiff, LinearAlgebra, UnPack

abstract type AbstractSpace end

local_geometry_data(space::AbstractSpace) =
    error("local geometry not defined for space: $(summary(space))")

undertype(space::AbstractSpace) =
    DataLayouts.basetype(local_geometry_data(space))

include("quadrature.jl")
import .Quadratures

include("dss.jl")
include("spectralelement.jl")
include("finitedifference.jl")
include("hybrid.jl")

end # module
