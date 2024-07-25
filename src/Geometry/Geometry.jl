module Geometry

using ..RecursiveApply
import LinearAlgebra

using StaticArrays

export ⊗
export UVector, VVector, WVector, UVVector, UWVector, VWVector, UVWVector
export Covariant1Vector,
    Covariant2Vector,
    Covariant3Vector,
    Covariant12Vector,
    Covariant13Vector,
    Covariant23Vector,
    Covariant123Vector
export Contravariant1Vector,
    Contravariant2Vector,
    Contravariant3Vector,
    Contravariant12Vector,
    Contravariant13Vector,
    Contravariant23Vector,
    Contravariant123Vector



include("coordinates.jl")
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")
include("globalgeometry.jl")

end # module
