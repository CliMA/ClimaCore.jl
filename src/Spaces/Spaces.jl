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
include("hybrid.jl")
include("triangulation.jl")
include("dss.jl")

end # module
