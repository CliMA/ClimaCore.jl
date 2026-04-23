module Operators

import LinearAlgebra, Adapt

using StaticArrays

import Base.Broadcast: Broadcasted

import ..slab, ..slab_args, ..column, ..column_args
import ClimaComms
import ..Utilities: new, safe_eltype, is_auto_broadcastable
import ..Utilities: add_auto_broadcasters, drop_auto_broadcasters
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts: DataLayouts, Data2D, DataSlab2D
import ..DataLayouts: vindex
import ..Geometry: Geometry, Covariant12Vector, Contravariant12Vector, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Grids
import ..Fields: Fields, Field

include("common.jl")
include("spectralelement.jl")
include("numericalflux.jl")
include("finitedifference.jl")
include("remapping.jl")
include("integrals.jl")
include("columnwise.jl")

end # module
