module Geometry

import LinearAlgebra
import UnrolledUtilities: unrolled_findfirst

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
include("axistensors.jl")
include("localgeometry.jl")
include("conversions.jl")
include("globalgeometry.jl")
include("mul_with_projection.jl")
include("auto_broadcaster_methods.jl")

"""
    Δz_metric_component(::Type{<:AbstractPoint})

The index of the z-component of an abstract point
in an `AxisTensor`.
"""
Δz_metric_component(::Type{<:LatLongZPoint}) = 9
Δz_metric_component(::Type{<:Cartesian3Point}) = 1
Δz_metric_component(::Type{<:Cartesian13Point}) = 4
Δz_metric_component(::Type{<:Cartesian123Point}) = 9
Δz_metric_component(::Type{<:XYZPoint}) = 9
Δz_metric_component(::Type{<:ZPoint}) = 1
Δz_metric_component(::Type{<:XZPoint}) = 4

end # module
