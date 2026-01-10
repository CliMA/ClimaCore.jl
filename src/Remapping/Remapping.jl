module Remapping

using LinearAlgebra, StaticArrays

export AbstractRemappingMethod, SpectralElementRemapping, BilinearRemapping

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

include("remapping_utils.jl")
include("interpolate_array.jl")
include("distributed_remapping.jl")
include("interpolate_pressure.jl")

end
