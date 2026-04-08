module Geometry

using ..RecursiveApply
import LinearAlgebra: det, dot, norm, norm_sqr, cross, UniformScaling
import UnrolledUtilities: unrolled_findfirst, unrolled_map, unrolled_foreach,
    unrolled_unique, unrolled_allunique, unrolled_in, unrolled_filter
using StaticArrays

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
include("rmul_with_projection.jl")

"""
    Δz_metric_component(::Type{<:AbstractPoint})

The index of the z-component of an abstract point
in a `Tensor`.
"""
Δz_metric_component(::Type{<:LatLongZPoint}) = 9
Δz_metric_component(::Type{<:Cartesian3Point}) = 1
Δz_metric_component(::Type{<:Cartesian13Point}) = 4
Δz_metric_component(::Type{<:Cartesian123Point}) = 9
Δz_metric_component(::Type{<:XYZPoint}) = 9
Δz_metric_component(::Type{<:ZPoint}) = 1
Δz_metric_component(::Type{<:XZPoint}) = 4

end # module
