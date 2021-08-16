module Geometry

using ..RecursiveApply

import StaticArrays: SVector

export ⊗, Cartesian12Vector

include("coordinates.jl")
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")

end # module
