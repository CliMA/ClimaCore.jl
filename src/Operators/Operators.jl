module Operators

import LinearAlgebra, Adapt

using StaticArrays

import Base.Broadcast: Broadcasted

import ..slab, ..slab_args, ..column, ..column_args
import ClimaComms
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..DataLayouts: vindex
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
include("remapping.jl")
include("integrals.jl")
include("columnwise.jl")

end # module
