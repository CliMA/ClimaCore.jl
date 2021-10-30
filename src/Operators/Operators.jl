module Operators

using StaticArrays: maximum
import ..slab
import ..column
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..Geometry: Geometry, Covariant12Vector, Contravariant12Vector, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Fields: Fields, Field
using ..RecursiveApply

import LinearAlgebra
using StaticArrays

include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")

end # module
