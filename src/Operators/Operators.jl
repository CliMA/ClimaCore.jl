module Operators

import ..slab
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..Geometry:
    Geometry, Cartesian12Vector, Covariant12Vector, Contravariant12Vector
import ..Spaces: Spaces, AbstractSpace, Quadratures
import ..Topologies
import ..Fields: Fields, Field
using ..RecursiveApply

import LinearAlgebra
using StaticArrays

include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")

end # module
