module Operators

import LinearAlgebra, Adapt

using StaticArrays

import Base.Broadcast: Broadcasted

import ..level, ..slab, ..column
import ClimaComms
import ..Utilities: Utilities, Cache, new, unwrap, is_auto_broadcastable
import ..Utilities: add_auto_broadcasters, drop_auto_broadcasters
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..DataLayouts
import ..Geometry: Geometry, ⊗
import ..Spaces: Spaces, Quadratures, AbstractSpace
import ..Topologies
import ..Meshes
import ..Grids
import ..Fields: Fields, Field
import UnrolledUtilities: unrolled_map, unrolled_filter

include("common.jl")
include("spectralelement.jl")
include("numericalflux.jl")
include("dg_fluxes.jl")
include("finitedifference.jl")
include("remapping.jl")
include("integrals.jl")
include("deprecated.jl")

end # module
