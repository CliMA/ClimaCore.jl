module Operators

import LinearAlgebra, Folds
using StaticArrays

import ..slab, ..slab_args, ..column, ..column_args
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..Geometry: Geometry, Covariant12Vector, Contravariant12Vector, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Fields: Fields, Field

using ..RecursiveApply

include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")
include("remapping.jl")

end # module
