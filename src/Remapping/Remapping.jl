module Remapping

using LinearAlgebra, StaticArrays

import ClimaComms
import ..DataLayouts,
    ..Geometry,
    ..Domains,
    ..Spaces,
    ..Grids,
    ..Topologies,
    ..Meshes,
    ..Operators,
    ..Quadratures,
    ..Fields,
    ..Hypsography
import ClimaCore.Utilities: half
import ClimaCore.Spaces: cuda_synchronize

using ..RecursiveApply

include("remapping_utils.jl")
include("interpolate_array.jl")
include("distributed_remapping.jl")

end
