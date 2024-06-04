module Operators

import LinearAlgebra, Adapt

using StaticArrays

import Base.Broadcast: Broadcasted

import ..broadcast_flatten
import ..slab, ..slab_args, ..column, ..column_args
import ClimaComms
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..Geometry: Geometry, Covariant12Vector, Contravariant12Vector, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Grids
import ..Fields: Fields, Field

using ..RecursiveApply

include("common.jl")
include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")
include("stencilcoefs.jl")
include("operator2stencil.jl")
include("pointwisestencil.jl")
include("remapping.jl")
include("integrals.jl")
include("thomas_algorithm.jl")

end # module
