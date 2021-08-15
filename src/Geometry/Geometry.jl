module Geometry

export ⊗, Cartesian12Vector, Tensor

import StaticArrays: SVector

include("coordinates.jl")
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")

end # module
