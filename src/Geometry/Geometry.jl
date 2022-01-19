module Geometry

using ..RecursiveApply
import LinearAlgebra

using StaticArrays

export ⊗

include("coordinates.jl")
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")
include("globalgeometry.jl")

end # module
