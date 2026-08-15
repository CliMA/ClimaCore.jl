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
projection dynamically dispatched, which fails to compile. The result is a
nested structure of singletons (axes and `nothing`s), so it is fully determined
by its inferred type, and inference folds `_dual_axes_for_projection` to that
constant (verified for the entry types of the `test_non_scalar_*` matrix-field
broadcasts and for 5-level-deep `NamedTuple` nestings with mixed axes). If a new
entry type ever defeats this — the symptom is an `InvalidIRError` from a dynamic
`isnothing` in `project_row2_for_mul` — the fix is to make this a `@generated`
function again (`QuoteNode(_dual_axes_for_projection(X))`), at the cost of
freezing the method table: a generator cannot see `_dual_axes_for_projection`
methods defined after it, including any added by downstream packages.
"""
@inline recursively_find_dual_axes_for_projection(::Type{X}) where {X} =
    _dual_axes_for_projection(X)

include("deprecated.jl")

end # module
