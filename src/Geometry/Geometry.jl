module Geometry

using ..Utilities: AutoBroadcaster, nested_broadcast, nested_broadcast_result_type
import LinearAlgebra: det, dot, norm, norm_sqr, cross, UniformScaling, Adjoint
import Random
using StaticArrays, UnrolledUtilities

export ⊗
export UVector, VVector, WVector, UVVector, UWVector, VWVector, UVWVector
export Covariant1Vector, Covariant2Vector, Covariant3Vector,
    Covariant12Vector, Covariant13Vector, Covariant23Vector,
    Covariant123Vector
export Contravariant1Vector, Contravariant2Vector, Contravariant3Vector,
    Contravariant12Vector, Contravariant13Vector, Contravariant23Vector,
    Contravariant123Vector



include("coordinates.jl")
include("tensors.jl")
include("localgeometry.jl")
include("conversions.jl")
include("globalgeometry.jl")
include("mul_with_projection.jl")
include("auto_broadcaster_methods.jl")

"""
    Δz_metric_component(::Type{<:AbstractPoint})

The index of the z-component of an abstract point
in a `Tensor`.
"""
Δz_metric_component(::Any) = 9

end # module
