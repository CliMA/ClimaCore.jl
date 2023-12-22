module Remapping

using LinearAlgebra, StaticArrays

import ClimaComms
import ..DataLayouts,
    ..Geometry,
    ..Spaces,
    ..Grids,
    ..Topologies,
    ..Meshes,
    ..Operators,
    ..Quadratures,
    ..Fields,
    ..Hypsography
import ClimaCore.Utilities: half

using ..RecursiveApply
using CUDA

include("interpolate_array.jl")
include("distributed_remapping.jl")

end
