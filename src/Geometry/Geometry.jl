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
    recursively_find_dual_axes_for_projection(X)

The axes that the second operand of a multiplication must be projected onto for
entries of type `X` in the first operand, or `nothing` if no projection is
needed. For entries with multiple components that do not all share one axis, the
result is a `Tuple` of axes that pairs componentwise with the entry, with
`nothing` for the components that need no projection.

The result must be a compile-time constant: the eager finite difference GPU
kernel branches on `isnothing` of it (see `project_row2_for_mul` in
`ext/cuda/operators_fd_eager.jl`), and a runtime branch there makes the whole
projection dynamically dispatched, which fails to compile. Inference does not
reliably fold `_dual_axes_for_projection` — it stops refining the `map` over
`fieldtypes` after a few levels of nesting, so a deeply nested entry widens to a
non-constant `Union` — hence the generator, which evaluates it during
compilation by construction.

This is defined here, rather than next to `_dual_axes_for_projection`, because a
generator may only call methods that already exist when it is defined, and the
last `_dual_axes_for_projection` method is added in `auto_broadcaster_methods.jl`.
"""
@generated recursively_find_dual_axes_for_projection(::Type{X}) where {X} =
    QuoteNode(_dual_axes_for_projection(X))

include("deprecated.jl")

end # module
