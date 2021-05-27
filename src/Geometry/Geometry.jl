module Geometry

export ⊗, Cartesian12Vector, Tensor

import StaticArrays: SVector

include("coordinates.jl")
include("localgeometry.jl")
include("vectors.jl")

end # module
