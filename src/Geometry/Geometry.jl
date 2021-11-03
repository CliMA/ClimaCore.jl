module Geometry

using ..RecursiveApply
import LinearAlgebra

import StaticArrays: SVector

export ⊗

include("coordinates.jl")
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")
include("globalgeometry.jl")

end # module
