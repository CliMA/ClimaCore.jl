import ..Utilities:
    PlusHalf,
    half,
    unionall_type,
    AutoBroadcaster,
    nested_broadcast,
    nested_broadcast_result_type,
    add_auto_broadcasters

const AllFiniteDifferenceSpace = Union{
    Spaces.FiniteDifferenceSpace,
    Spaces.ExtrudedFiniteDifferenceSpace,
    Spaces.MultiColumnFiniteDifferenceSpace,
}
const AllFaceFiniteDifferenceSpace = Union{
    Spaces.FaceFiniteDifferenceSpace,
    Spaces.FaceExtrudedFiniteDifferenceSpace,
    Spaces.FaceMultiColumnFiniteDifferenceSpace,
}
const AllCenterFiniteDifferenceSpace = Union{
    Spaces.CenterFiniteDifferenceSpace,
    Spaces.CenterExtrudedFiniteDifferenceSpace,
    Spaces.CenterMultiColumnFiniteDifferenceSpace,
}

Topologies.isperiodic(space::AllFiniteDifferenceSpace) =
    Topologies.isperiodic(Spaces.vertical_topology(space))


left_idx(space::AllCenterFiniteDifferenceSpace) =
    left_center_boundary_idx(space)
right_idx(space::AllCenterFiniteDifferenceSpace) =
    right_center_boundary_idx(space)
left_idx(space::AllFaceFiniteDifferenceSpace) = left_face_boundary_idx(space)
right_idx(space::AllFaceFiniteDifferenceSpace) = right_face_boundary_idx(space)

left_center_boundary_idx(space::AllFiniteDifferenceSpace) = 1
right_center_boundary_idx(space::AllFiniteDifferenceSpace) =
    num_levels(Spaces.grid(space), Grids.CellCenter())
left_face_boundary_idx(space::AllFiniteDifferenceSpace) = half
right_face_boundary_idx(space::AllFiniteDifferenceSpace) =
    num_levels(Spaces.grid(space), Grids.CellFace()) - half

# The local geometry data of a column slice (a ColumnGrid) is read through the
# data of its full grid, since building and reading a view of the column for
# every point would be much slower.
num_levels(grid, staggering) = size(Grids.local_geometry_data(grid, staggering), 1)
num_levels(grid::Grids.ColumnGrid, staggering) = num_levels(grid.full_grid, staggering)
Base.@propagate_inbounds column_local_geometry(grid, staggering, v) =
    Grids.local_geometry_data(grid, staggering)[v]
Base.@propagate_inbounds column_local_geometry(grid::Grids.ColumnGrid, staggering, v) =
    Grids.local_geometry_data(grid.full_grid, staggering)[CartesianIndex(
        v,
        grid.indices...,
    )]


left_face_boundary_idx(arg) = left_face_boundary_idx(axes(arg))
right_face_boundary_idx(arg) = right_face_boundary_idx(axes(arg))
left_center_boundary_idx(arg) = left_center_boundary_idx(axes(arg))
right_center_boundary_idx(arg) = right_center_boundary_idx(axes(arg))

# This can read the face local geometry of a center space, and vice versa.
Base.@propagate_inbounds Geometry.LocalGeometry(space::AllFiniteDifferenceSpace, idx) =
    column_local_geometry(
        Spaces.grid(space),
        idx isa PlusHalf ? Grids.CellFace() : Grids.CellCenter(),
        level_index(space, idx),
    )

"""
    VerticalBoundaryCondition <: AbstractBoundaryCondition

Supertype for the boundary conditions of the vertical (column)
[`FiniteDifferenceOperator`](@ref)s, e.g. [`SetValue`](@ref) and
[`Extrapolate`](@ref). Subtypes should define:

  - [`boundary_width`](@ref)
  - [`stencil_left_boundary`](@ref)
  - [`stencil_right_boundary`](@ref)
"""
abstract type VerticalBoundaryCondition <: AbstractBoundaryCondition end

Adapt.adapt_structure(to, bc::VerticalBoundaryCondition) =
    hasfield(typeof(bc), :val) ? unionall_type(typeof(bc))(Adapt.adapt(to, bc.val)) : bc

# Field-valued boundary conditions are sliced like the arguments of broadcasts.
Base.@propagate_inbounds column(bc::VerticalBoundaryCondition, inds...) =
    hasfield(typeof(bc), :val) && bc.val isa MaybeLazyField ?
    unionall_type(typeof(bc))(
        column(bc.val, Fields.arg_slice_indices(column, bc.val, inds)...),
    ) : bc

promote_bc(bc::VerticalBoundaryCondition, FT) =
    hasfield(typeof(bc), :val) ? unionall_type(typeof(bc))(promote_val(bc.val, FT)) : bc

promote_val(val, ::Type{FT}) where {FT} = val
promote_val(val::FT, ::Type{FT}) where {FT} = val
promote_val(val::Number, ::Type{FT}) where {FT} = FT(val)
promote_val(val::Geometry.Tensor, ::Type{FT}) where {FT} =
    Geometry.Tensor(similar_type(parent(val), FT)(parent(val)), axes(val))
promote_val(val::Geometry.Tensor{<:Any, FT}, ::Type{FT}) where {FT} = val

"""
    NullBoundaryCondition()

Placeholder boundary condition, used when no other boundary condition is given.

Wherever an operator needs a boundary row for this condition (that is, wherever
[`boundary_width`](@ref) is nonzero for it), the result produced there is `NaN`,
which flags the missing boundary condition. This holds on every evaluation path:
an operator that is rewritten into an operator matrix multiply gets a `NaN`
boundary row (see `MatrixFields`), and any other operator goes through
[`stencil_left_boundary`](@ref) / [`stencil_right_boundary`](@ref), which
produce `NaN` directly.

The advection operators never use this condition: when they are given no
boundary conditions, [`Extrapolate{0}`](@ref Extrapolate) is added to their
`bcs` by default, and a boundary whose name has no entry in `bcs` also falls
back to `Extrapolate{0}` (see [`AdvectionOperator`](@ref)).

Where `boundary_width` is zero the interior stencil applies instead, so the same
operator can give `NaN` at one boundary and an ordinary value at the other. To
obtain a meaningful boundary value, give the operator a boundary condition, or
overwrite the boundary afterwards with a [`SetBoundaryOperator`](@ref).
"""
struct NullBoundaryCondition <: VerticalBoundaryCondition end

"""
    SetValue(val)

Set the value at the boundary to be `val`. In the case of gradient operators,
this sets the input value from which the gradient is computed.
"""
struct SetValue{S} <: VerticalBoundaryCondition
    val::S
end

"""
    SetGradient(val)

Set the gradient at the boundary to be `val`. In the case of gradient operators
this sets the output value of the gradient.
"""
struct SetGradient{S} <: VerticalBoundaryCondition
    val::S
end

"""
    SetDivergence(val)

Set the divergence at the boundary to be `val`.
"""
struct SetDivergence{S} <: VerticalBoundaryCondition
    val::S
end

"""
    SetCurl(val)

Set the curl at the boundary to be `val`.
"""
struct SetCurl{S} <: VerticalBoundaryCondition
    val::S
end

"""
    Extrapolate{N}()
    Extrapolate(N = 0)

Evaluate the same stencil as the interior, but pad each ghost point the stencil
reaches with a value extrapolated (with an order-`N` polynomial) from the
`N + 1` closest interior points. Only `0 <= N <= 2` is supported.

If a stencil at a face `i` is a function of the values at
`x[i-3/2], x[i-1/2], x[i+1/2], x[i+3/2]`, then at the face `i = 3/2` the single
ghost point `x[0]` is padded with the weighted sum of the interior points
`x[1], x[2], x[3]`, with the following weights:

| N | x[1] | x[2] | x[3] |
|:- |:---- |:---- |:---- |
| 0 | 1    | 0    | 0    |
| 1 | 2    | -1   | 0    |
| 2 | 3    | -3   | 1    |

Only the interior points that the stencil can reach are available for the
extrapolation, and if a ghost point requires more interior points than are
available, `N` is reduced until the ghost point can be extrapolated with the
available interior points. For example, if `N = 2` and the stencil above is
evaluated at the boundary face `i = 1/2`, only the 2 interior points
`x[1], x[2]` are available, so both ghost points are padded with the `N = 1`
extrapolation:

```
x[-1] = x[0] = 2 * x[1] - x[2]
```

Every ghost point of a stencil is padded with the same extrapolated value: the
extrapolation continues the field along the third coordinate line
with a single boundary value, rather than evaluating the extrapolating
polynomial at each ghost point's own position.
"""
struct Extrapolate{N} <: VerticalBoundaryCondition
    function Extrapolate{N}() where {N}
        N isa Integer && 0 <= N <= 2 ||
            error("Extrapolate only supports orders 0 <= N <= 2; got N = $N")
        return new{N}()
    end
end
Extrapolate(N::Integer = 0) = Extrapolate{N}()

"""
    Outflow(; order = 0)

Construct the outflow (zero-normal-gradient family) boundary condition, which is
[`Extrapolate{order}()`](@ref Extrapolate), so it is accepted wherever
`Extrapolate` is. On the finite-difference advection operators it pads the
ghost points the interior stencil reaches with an order-`order` extrapolation
from the interior; near a boundary the order is reduced when fewer than
`order + 1` interior points are in range, as documented for
[`Extrapolate`](@ref). `Outflow()` is the zero-order (constant-value) closure.
"""
Outflow(; order = 0) = Extrapolate{order}()

"""
    extrapolate_weights(bc::Extrapolate{N}, navailable, FT)

Return the weights of the `navailable` closest interior points in the ghost-point
extrapolation of `bc`, as a tuple of 3 numbers of type `FT` ordered from the
closest interior point outwards (with trailing zeros when fewer than 3 points
are used). The extrapolation order is reduced to `navailable - 1` when fewer
than `N + 1` interior points are available.
"""
# The weights are floats chosen in each branch, rather than integers converted
# after the choice: LLVM narrows a run-time choice between the integers 0 and -1
# to a Bool converted with `sitofp i1`, which the NVPTX backend of LLVM 15
# (Julia 1.10) converts to 1.0 instead of -1.0, so that GPU kernels that fold
# ghost points into the rows of operator matrices extrapolated with the wrong
# sign whenever the compiler could bound the number of available points.
function extrapolate_weights(
    ::Extrapolate{N},
    navailable::Integer,
    ::Type{FT},
) where {N, FT}
    n = min(N, navailable - 1)
    return n == 0 ? (FT(1), FT(0), FT(0)) :
           n == 1 ? (FT(2), FT(-1), FT(0)) : (FT(3), FT(-3), FT(1))
end

# Callable ghost-point reconstruction interface (see AdvectionOperator): the
# arguments are the interior points available to the extrapolation, ordered
# from the one closest to the boundary outwards, and the result is the value
# shared by every ghost point the stencil reaches. The extrapolation order is
# reduced when fewer than N + 1 interior points are given; dispatching on the
# reduced order keeps the zero-weight terms out of the computation (see
# `extrapolate_weights` for the weights themselves). The reconstruction of a
# tuple-valued field applies componentwise, through the AutoBroadcaster
# arithmetic. The result must have the same type as the inputs (which are
# already AutoBroadcasters for tuple-valued fields, wrapped when broadcasted), so
# the wrappers are not dropped here: `advection_ghost_values` substitutes the
# result for a subset of the clamped stencil values, and a type mismatch
# between the two makes the stencil evaluation dynamically dispatched.
@inline (::Extrapolate{N})(x₁, x₂) where {N} =
    extrapolate_ghost_value(Val(min(N, 1)), x₁, x₂, x₂)
@inline (::Extrapolate{N})(x₁, x₂, x₃) where {N} =
    extrapolate_ghost_value(Val(min(N, 2)), x₁, x₂, x₃)
@inline extrapolate_ghost_value(::Val{0}, x₁, x₂, x₃) = x₁
@inline extrapolate_ghost_value(::Val{1}, x₁, x₂, x₃) =
    2 * add_auto_broadcasters(x₁) - add_auto_broadcasters(x₂)
# Grouped as 3(x₁ - x₂) + x₃ rather than 3x₁ - 3x₂ + x₃ so that constant data
# is reconstructed exactly
@inline extrapolate_ghost_value(::Val{2}, x₁, x₂, x₃) =
    3 * (add_auto_broadcasters(x₁) - add_auto_broadcasters(x₂)) +
    add_auto_broadcasters(x₃)

abstract type Location end
abstract type Boundary <: Location end
abstract type BoundaryWindow{name} <: Location end

struct Interior <: Location end
struct LeftBoundaryWindow{name} <: BoundaryWindow{name} end
struct RightBoundaryWindow{name} <: BoundaryWindow{name} end

@inline left_boundary_window(space) =
    LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
@inline right_boundary_window(space) =
    RightBoundaryWindow{Spaces.right_boundary_name(space)}()

"""
    FiniteDifferenceOperator

Supertype of the finite difference operators, which act along the vertical
(column) direction. Subtypes define:

  - [`return_eltype`](@ref)
  - [`return_space`](@ref)
  - [`stencil_interior_width`](@ref)
  - [`stencil_interior`](@ref)

See also [`VerticalBoundaryCondition`](@ref) for how to define the boundaries.
"""
abstract type FiniteDifferenceOperator <: AbstractOperator end

# Rebuild op with f applied to its boundary conditions (the first field of every
# operator that has them), keeping its other fields.
@inline function map_bcs(f::F, op) where {F}
    hasfield(typeof(op), :bcs) || return op
    @assert fieldname(typeof(op), 1) === :bcs
    return unionall_type(typeof(op))(
        f(op.bcs),
        ntuple(n -> getfield(op, n + 1), Val(fieldcount(typeof(op)) - 1))...,
    )
end

Adapt.adapt_structure(to, op::FiniteDifferenceOperator) =
    map_bcs(bcs -> Adapt.adapt(to, bcs), op)

Base.@propagate_inbounds column(op::FiniteDifferenceOperator, inds...) =
    map_bcs(bcs -> column(bcs, inds...), op)

get_boundary(op::FiniteDifferenceOperator, ::BoundaryWindow{name}) where {name} =
    hasfield(typeof(op.bcs), name) ? getfield(op.bcs, name) : NullBoundaryCondition()

@inline promote_bcs(op::FiniteDifferenceOperator, ::Type{FT}) where {FT} =
    map_bcs(bcs -> unrolled_map(Base.Fix2(promote_bc, FT), bcs), op)

"""
    boundary_width(op::FiniteDifferenceOperator, bc::VerticalBoundaryCondition, args...)

Return the width of a boundary condition `bc` on an operator `op`: the number
of locations at which a modified stencil is used.
"""
boundary_width(::FiniteDifferenceOperator, bc::NullBoundaryCondition, args...) = 0
boundary_width(::FiniteDifferenceOperator, bc::SetValue, args...) = 1
boundary_width(::FiniteDifferenceOperator, bc::SetGradient, args...) = 1
boundary_width(::FiniteDifferenceOperator, bc::SetDivergence, args...) = 1
boundary_width(::FiniteDifferenceOperator, bc::SetCurl, args...) = 1
boundary_width(::FiniteDifferenceOperator, bc::Extrapolate{N}, args...) where {N} = N + 1

struct StencilStyle <: OperatorStyle end

OperatorStyle(::FiniteDifferenceOperator) = StencilStyle()
slice_operator(::StencilStyle) = column

const StencilBroadcasted{F} = Broadcast.Broadcasted{StencilStyle, <:Any, F}
const StencilOperatorBroadcasted{Op <: FiniteDifferenceOperator} = StencilBroadcasted{Op}

"""
    stencil_interior_width(::Op, args...)

Return the width of the interior stencil for the operator `Op` with the given
arguments, as a tuple of 2-tuples: each 2-tuple holds the lower and upper bounds
of the index offsets of the stencil for the corresponding argument.

# Examples

```julia
stencil_interior_width(::Op, arg1, arg2) = ((-half, 1 + half), (0, 0))
```

implies that at index `i`, the stencil accesses `arg1` at `i - half`, `i + half`
and `i + 1 + half`, and `arg2` at index `i`.
"""
function stencil_interior_width end

"""
    stencil_interior(::Op, space, idx, args...)

Return the value of the interior stencil of the operator `Op` at vertical index
`idx` of the column `space`; `args` are the input arguments.
"""
function stencil_interior end

"""
    stencil_left_boundary(op, bc, space, idx, args...)

Return the result of the stencil operator `op` at vertical index `idx` of the
column `space` near the left boundary, with boundary condition
`bc`. For operators that cannot be evaluated without a boundary condition, a
`NullBoundaryCondition` generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this
method: their boundary rows come from `MatrixFields` instead, where a
`NullBoundaryCondition` row is filled with `NaN`s, so the boundary output is
`NaN` there as well.
"""
stencil_left_boundary(op, ::NullBoundaryCondition, space, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

"""
    stencil_right_boundary(op, bc, space, idx, args...)

Return the result of the stencil operator `op` at vertical index `idx` of the
column `space` near the right boundary, with boundary condition
`bc`. For operators that cannot be evaluated without a boundary condition, a
`NullBoundaryCondition` generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this
method: their boundary rows come from `MatrixFields` instead, where a
`NullBoundaryCondition` row is filled with `NaN`s, so the boundary output is
`NaN` there as well.
"""
stencil_right_boundary(op, ::NullBoundaryCondition, space, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

# Space with the staggering of the vertical index idx (Integer for centers,
# PlusHalf for faces).
@inline staggered_space(space, idx) =
    idx isa PlusHalf ? Spaces.face_space(space) : Spaces.center_space(space)

"""
    left_interior_idx(space, op, bc, args...)

Return the index of the left-most interior point of the operator `op` with
boundary `bc` when used with arguments `args...` and output space `space`. This is the first point that
is not within [`boundary_width`](@ref) of the boundary, and at which the
interior stencil does not read any values outside of the column (see
[`stencil_interior_width`](@ref)).
"""
@inline left_interior_idx(space, op, bc::VerticalBoundaryCondition, args...) =
    left_idx(space) + left_window_width(space, op, bc, args...)

"""
    right_interior_idx(space, op, bc, args...)

Return the index of the right-most interior point of the operator `op` with
boundary `bc` when used with arguments `args...` and output space `space`. This is the last point that
is not within [`boundary_width`](@ref) of the boundary, and at which the
interior stencil does not read any values outside of the column (see
[`stencil_interior_width`](@ref)).
"""
@inline right_interior_idx(space, op, bc::VerticalBoundaryCondition, args...) =
    right_idx(space) - right_window_width(space, op, bc, args...)

# Widths of the boundary windows (the numbers of points at the left and right
# ends of a column that are not interior points), which only depend on the
# types of the arguments: every column of a space ends as far from the end of
# the column with the other staggering as it starts from its start (a face
# column has one more point).
@inline function left_window_width(space, op, bc, args...)
    stencil_width = unrolled_maximum(stencil_interior_width(op, args...)) do (lo, _)
        first_index(space, left_idx(space) + lo) - lo - left_idx(space)
    end
    return max(boundary_width(op, bc, args...), stencil_width)
end
@inline function right_window_width(space, op, bc, args...)
    stencil_width = unrolled_maximum(stencil_interior_width(op, args...)) do (_, hi)
        hi + first_index(space, left_idx(space) + hi) - left_idx(space)
    end
    return max(boundary_width(op, bc, args...), stencil_width)
end

# Whether a point at the given distance from an end of the column lies in a
# window of the given width. The comparison is only made when the window is not
# empty, so that an operator without boundary points on one side (e.g., a
# SetBoundaryOperator with a condition on the other side only) never branches
# on the point's index, whose range the compiler does not always know, even
# though the window's bounds are constants.
@inline in_window(distance, width) = width > 0 && distance < width

# Whether the stencil at idx (an index of space) is a boundary stencil.
@inline function should_call_left_boundary(idx, space, op, args...)
    Topologies.isperiodic(space) && return false
    bc = get_boundary(op, left_boundary_window(space))
    return in_window(idx - left_idx(space), left_window_width(space, op, bc, args...))
end

@inline function should_call_right_boundary(idx, space, op, args...)
    Topologies.isperiodic(space) && return false
    bc = get_boundary(op, right_boundary_window(space))
    return in_window(right_idx(space) - idx, right_window_width(space, op, bc, args...))
end

# Level of a column space at vertical index idx (an Integer on centers or a
# PlusHalf on faces), wrapping around periodic columns.
@inline function level_index(space, idx)
    v = idx isa PlusHalf ? idx + half : idx
    return Topologies.isperiodic(space) ? mod1(v, right_center_boundary_idx(space)) : v
end

# First and last vertical indices of a column of space with the staggering of
# idx (an Integer on centers or a PlusHalf on faces).
@inline first_index(space, idx) =
    idx isa PlusHalf ? left_face_boundary_idx(space) : left_center_boundary_idx(space)
@inline last_index(space, idx) =
    idx isa PlusHalf ? right_face_boundary_idx(space) : right_center_boundary_idx(space)

# Whether a column of space has a point at the vertical index idx (of either
# staggering), and the closest index to idx that it has. Every index is in a
# periodic column, since its indices wrap around (see level_index).
@inline in_column(space, idx) =
    Topologies.isperiodic(space) ||
    first_index(space, idx) <= idx <= last_index(space, idx)
@inline column_index(space, idx) =
    Topologies.isperiodic(space) ? idx :
    clamp(idx, first_index(space, idx), last_index(space, idx))

# Numbers of the points of a stencil at the offsets ld:ud from the vertical index
# idx that lie beyond the left and right ends of a column of space. Stencils read
# their values at indices clamped to the column (see column_index), and the
# values or coefficients of these ghost points are then replaced using the
# boundary conditions. The numbers are computed from the distances of idx to the
# ends of the column, and they are zero at compile time on a side where the
# stencil never reaches beyond the column (see max_ghost_counts).
@inline function ghost_counts(space, idx, ld, ud)
    Topologies.isperiodic(space) && return (0, 0)
    (max_left, max_right) = max_ghost_counts(space, idx, ld, ud)
    return (
        max_left > 0 ? max(max_left - (idx - first_index(space, idx)), 0) : 0,
        max_right > 0 ? max(max_right - (last_index(space, idx) - idx), 0) : 0,
    )
end
# The numbers of ghost points at the ends of the column, which are the largest
# numbers that the stencil can reach (see right_window_width for the right end).
@inline function max_ghost_counts(space, idx, ld, ud)
    offset = first_index(space, idx + ld) - first_index(space, idx)
    return (offset - ld, ud + offset)
end

"""
    fuses_into_stencils(op)

Whether a [`FiniteDifferenceOperator`](@ref) can always be evaluated wherever
the stencil of another operator reads its result, because it only reads its own
arguments at the same point (or only reads values that are not computed in the
column, like local geometry). Other operators are only evaluated where they are
read when their arguments contain no stencils.
"""
fuses_into_stencils(::FiniteDifferenceOperator) = false

"""
    reads_neighbors(op, args...)

Return a tuple with one `Val(true)` or `Val(false)` for each argument of a
[`FiniteDifferenceOperator`](@ref), indicating whether its stencil reads that
argument at points other than the one it evaluates. By default, every argument
is read at neighboring points, unless the operator fuses into stencils (see
[`fuses_into_stencils`](@ref)).
"""
reads_neighbors(op, args...) = unrolled_map(Returns(Val(!fuses_into_stencils(op))), args)

"""
    reads_in_lockstep(op)

Whether the stencil of a [`FiniteDifferenceOperator`](@ref) reads its arguments
at neighboring points in the same way at every point of a column, without
branching on the point (e.g., by reading at indices clamped to the column, and
replacing values that lie outside of it after they are read). The threads that
evaluate such a stencil read its arguments at the same time (see
[`DataLayouts.update_points!`](@ref)), so its cached arguments can be kept in
registers on devices that support it (see [`cached_arg`](@ref)). By default,
this is `false`.
"""
reads_in_lockstep(_) = false

# How an operator reads each of its arguments: at the point it evaluates
# (Val(false)), at neighboring points (Val(true)), or at neighboring points in
# lockstep (Val(:lockstep); see reads_in_lockstep).
arg_reads(bc::StencilOperatorBroadcasted) =
    unrolled_map(reads_neighbors(bc.f, bc.args...)) do reads
        reads isa Val{true} && reads_in_lockstep(bc.f) ? Val(:lockstep) : reads
    end
arg_reads(bc) = unrolled_map(Returns(Val(false)), bc.args)
reads_neighbor(reads) = !(reads isa Val{false})

# Whether a broadcast contains operators that read their arguments at
# neighboring points.
has_stencils(arg) = false
has_stencils(bc::StencilBroadcasted) = unrolled_any(has_stencils, bc.args)
has_stencils(bc::StencilOperatorBroadcasted) =
    !fuses_into_stencils(bc.f) || unrolled_any(has_stencils, bc.args)

# Whether a broadcast evaluates each point from the value of its only argument
# at that point, either without any computation (like adjoint), or by replacing
# the values at the boundaries (which can differ in type from the others). Such
# a broadcast is read at neighboring points from a cache of its argument (see
# stencil_arg), so every cached value has the same type.
passes_through(_) = false
passes_through(bc::Broadcast.Broadcasted) =
    bc.f isa Union{typeof(adjoint), typeof(add_auto_broadcasters), SetBoundaryOperator}

"""
    recomputable(arg)

Whether an argument that an operator reads at neighboring points is cheaper to
evaluate at every point that reads it than to cache (see `stencil_arg`): a
pointwise expression of at most `MAX_RECOMPUTED_DEPTH` nested broadcasts over
constants and `Field`s that every thread can read, whose functions are all
[`recomputable_node`](@ref)s, or an application of a
[`recomputable_operator`](@ref). Caching such an expression would cost a buffer
that every thread writes and synchronizes on before any thread reads it, which
is more than the stencil's few evaluations of it.
"""
recomputable(arg) = recomputable(arg, Val(MAX_RECOMPUTED_DEPTH))
recomputable(_, _) = true
recomputable(field::Field, _) =
    !DataLayouts.stored_in_registers(Fields.field_values(field))
recomputable(::LazyField, _) = false
recomputable(bc::Fields.PointwiseBroadcasted, ::Val{depth}) where {depth} =
    depth > 0 &&
    recomputable_node(bc.f) &&
    unrolled_all(arg -> recomputable(arg, Val(depth - 1)), bc.args)
recomputable(bc::StencilOperatorBroadcasted, _) = recomputable_operator(bc.f)
const MAX_RECOMPUTED_DEPTH = 2

"""
    recomputable_operator(op)

Whether an application of the [`FiniteDifferenceOperator`](@ref) `op` can be
[`recomputable`](@ref), which holds when its value at each point does not read
any values (like the operator matrices of interpolations, whose rows only
depend on the float type). This is `false` by default.
"""
recomputable_operator(_) = false

"""
    recomputable_node(f)

Whether a pointwise broadcast of `f` can be [`recomputable`](@ref). This is
`false` for functions whose evaluation is expensive relative to a buffer read,
like projections that read the local geometry (see
`MatrixFields.ProjectForMul`), and `true` by default.
"""
recomputable_node(_) = true

# Whether an argument that is read at neighboring points is cached when buffers
# are shared (see stencil_arg), which holds for every argument that cached_arg
# evaluates into a buffer, unless it is recomputable, and for every broadcast
# that passes through the values of such an argument.
caches_values(arg) = is_cached(arg) && !recomputable(arg)
is_cached_arg(arg) = passes_through(arg) ? is_cached_arg(arg.args[1]) : caches_values(arg)

# Whether an operator caches any of its arguments.
caches_args(bc) =
    !has_private_buffers(bc) && unrolled_any(
        identity,
        unrolled_map(bc.args, arg_reads(bc)) do arg, reads
            reads_neighbor(reads) && is_cached_arg(arg)
        end,
    )

# Whether an operator is evaluated wherever another operator reads its result,
# which avoids evaluating it over a whole column. With private buffers, this
# happens when its arguments contain no stencils, so that no value is recomputed
# more than once for each point that reads it. Otherwise, this happens when it
# caches no arguments, and every operator that caches arguments is evaluated
# before any expression that contains it is cached or evaluated, so that the
# cached arguments of different applications are never live at the same time.
is_fused(bc) =
    has_private_buffers(bc) ?
    fuses_into_stencils(bc.f) || !unrolled_any(has_stencils, bc.args) : !caches_args(bc)

# A stencil expression with every argument replaced by its operator_arg.
fused_stencil(bc) =
    Broadcast.Broadcasted(bc.style, bc.f, unrolled_map(operator_arg, bc.args), bc.axes)

operator_arg(bc::StencilBroadcasted) =
    Broadcast.broadcasted(bc.f, unrolled_map(operator_arg, bc.args)...)
operator_arg(bc::StencilOperatorBroadcasted) =
    is_fused(bc) ? fused_stencil(bc) : apply_operators(bc)

# A pointwise expression over operators is evaluated in a single pass over each
# column, which evaluates every operator that the expression reads at the point
# being evaluated, unless it needs too much shared memory on GPUs, or it caches
# arguments while another operator in the pass also does. When buffers are
# shared, the pass also evaluates every operator that those operators read at
# the same point. Only other arguments may be evaluated over the whole column.
# At the top level of an expression, this pass writes directly into the
# destination.
pointwise_stencil(bc) = pointwise_stencil(bc, Val(num_caching_operators(bc) <= 1))
pointwise_stencil(bc, fuse_caching) = Broadcast.Broadcasted(
    bc.style,
    bc.f,
    unrolled_map(bc.args, in_pass(bc)) do arg, same_pass
        same_pass isa Val{true} ? pointwise_arg(arg, fuse_caching) : operator_arg(arg)
    end,
    bc.axes,
)
pointwise_arg(arg, _) = operator_arg(arg)
pointwise_arg(bc::StencilBroadcasted, fuse_caching) = pointwise_stencil(bc, fuse_caching)
pointwise_arg(bc::StencilOperatorBroadcasted, fuse_caching) =
    inlined_buffer_bytes(bc) <= MAX_INLINED_BUFFER_BYTES &&
    (fuse_caching isa Val{true} || !caches_args(bc)) ?
    pointwise_stencil(bc, fuse_caching) : apply_operators(bc)
in_pass(bc) = unrolled_map(Returns(Val(true)), bc.args)
in_pass(bc::StencilOperatorBroadcasted) =
    has_private_buffers(bc) ? unrolled_map(Returns(Val(false)), bc.args) :
    unrolled_map(reads -> Val(reads isa Val{false}), arg_reads(bc))
num_caching_operators(_) = 0
num_caching_operators(bc::StencilBroadcasted) =
    Int(caches_args(bc)) + unrolled_sum(
        unrolled_map(bc.args, in_pass(bc)) do arg, same_pass
            same_pass isa Val{true} ? num_caching_operators(arg) : 0
        end,
    )

apply_pointwise_operators(::StencilStyle, bc) = apply_stencil(pointwise_stencil(bc))
apply_operators!(dest, bc::StencilBroadcasted) =
    apply_stencil!(dest, pointwise_stencil(bc), Val(true))

# On GPUs, an operator allocates one shared buffer for every argument it caches,
# and so does every operator in its arguments, all in one inlined body.
inlined_buffer_bytes(bc::StencilOperatorBroadcasted) = published_bytes(bc, Val(false))
published_bytes(arg, reads) =
    reads_neighbor(reads) && passes_through(arg) ? published_bytes(arg.args[1], reads) :
    (reads_neighbor(reads) && caches_values(arg) ? sizeof(eltype(arg)) : 0) +
    nested_bytes(arg)
nested_bytes(_) = 0
nested_bytes(bc::StencilBroadcasted) =
    unrolled_sum(unrolled_map(published_bytes, bc.args, arg_reads(bc)))

@drop_recursion_limits has_stencils,
recomputable,
is_cached_arg,
operator_arg,
pointwise_stencil,
pointwise_arg,
num_caching_operators,
published_bytes,
nested_bytes

# Value of an operator argument or boundary value at vertical index idx of a
# column space, where values that are constant along the column (e.g., level
# fields) are indexed through Broadcast.newindex.
Base.@propagate_inbounds column_value(arg::MaybeLazyField, space, idx) =
    arg[Broadcast.newindex(arg, level_index(space, idx))]
# A generated function rather than an unrolled_map over a closure: each
# pointwise node of a stencil expression would otherwise add two method
# instances per expression type (the closure and its unrolled_map), each
# optimized with the node's whole subtree inlined into it (see also
# DataLayouts.slice_every_arg).
@generated function column_value(bc::StencilBroadcasted, space, idx)
    N = length(bc.parameters[4].parameters)
    return quote
        Base.@_propagate_inbounds_meta
        args = getfield(bc, :args)
        return getfield(bc, :f)(
            Base.Cartesian.@ntuple($N, n -> column_value(getfield(args, n), space, idx))...,
        )
    end
end
Base.@propagate_inbounds column_value(bc::StencilOperatorBroadcasted, space, idx) =
    stencil_value(bc.f, staggered_space(space, idx), idx, bc.args...)
@inline column_value(arg::Union{Ref, Broadcast.Broadcasted}, _, _) = arg[]
@inline column_value(arg::Tuple{Any}, _, _) = arg[1]
@inline column_value(arg, _, _) = add_auto_broadcasters(arg)

# Value of a boundary condition at the boundary index idx of space. A value over
# a whole column is read at its first level on the left (bottom) boundary and
# at its last level on the right (top) boundary, so that a value on the
# opposite staggering is read at its point that is closest to the boundary
# (e.g., the center adjacent to a boundary face).
Base.@propagate_inbounds boundary_value(val, space, idx) = column_value(val, space, idx)
Base.@propagate_inbounds boundary_value(val::MaybeLazyField, space, idx) =
    val[Broadcast.newindex(val, idx == left_idx(space) ? 1 : num_values(val))]
num_values(val) = iszero(ndims(val)) ? 1 : size(val)[1]

# The value of an operator application at vertical index idx of space, using
# the boundary stencils of its boundary windows. On a column too short to
# separate the two boundary windows, an index can lie in both windows at once;
# the left (bottom) boundary condition always takes precedence. In particular,
# when both boundary conditions prescribe the operator's output at such an index
# (e.g. a two-sided SetDivergence on a single-level DivergenceF2C), only the left
# one is applied.
Base.@propagate_inbounds stencil_value(op, space, idx, args...) =
    if should_call_left_boundary(idx, space, op, args...)
        bc = get_boundary(op, left_boundary_window(space))
        stencil_left_boundary(op, bc, space, idx, args...)
    elseif should_call_right_boundary(idx, space, op, args...)
        bc = get_boundary(op, right_boundary_window(space))
        stencil_right_boundary(op, bc, space, idx, args...)
    else
        stencil_interior(op, space, idx, args...)
    end

# Vertical index of the point at the given CartesianIndex in a column of space,
# where a column with a single point has a zero-dimensional index.
@inline function vertical_index(space, index)
    v = isempty(Tuple(index)) ? 1 : index[1]
    return Spaces.staggering(space) isa Spaces.CellFace ? v - half : v
end

# Field with the data shape of a column slice of space and the given scope, used
# as a template for allocating the result of an operator. Only the type and shape
# of the template's data are used.
@inline function space_template(space, scope)
    data = DataLayouts.column(
        Grids.local_geometry_data(Spaces.grid(space), Spaces.staggering(space)),
        1,
        1,
        1,
    )
    return Field(DataLayouts.reassign(data, scope), space)
end

# Arguments of an operator can be read at neighboring points (see
# reads_neighbors). When buffers are shared, each such argument is cached (see
# cached_arg): every thread evaluates it once at each point it owns, and
# publishes the values through a shared buffer that the operator reads instead,
# so that no value is recomputed for each point that reads it, and values in
# registers can be read by other threads. Pointwise functions and other
# arguments are read at the point being evaluated, so their own arguments are
# only cached when they are read at neighboring points. Since an operator that
# caches arguments is evaluated before any expression that contains it (see
# is_fused), a cached expression never caches arguments of its own.
@inline stencil_arg(arg, _) = arg
@inline stencil_arg(arg::MaybeLazyField, reads) =
    has_private_buffers(arg) ? arg :
    reads_neighbor(reads) ? neighbor_arg(arg, Val(reads isa Val{:lockstep})) :
    arg isa StencilBroadcasted ?
    Broadcast.Broadcasted(
        arg.style,
        arg.f,
        unrolled_map(stencil_arg, arg.args, arg_reads(arg)),
        arg.axes,
    ) : arg
@inline neighbor_arg(arg, lockstep) =
    !is_cached_arg(arg) ? arg :
    passes_through(arg) ?
    Broadcast.Broadcasted(
        StencilStyle(),
        arg.f,
        (neighbor_arg(arg.args[1], lockstep),),
        arg.axes,
    ) : cached_arg(arg, lockstep)

# A stencil expression is evaluated into a buffer or registers like any operator
# application.
@inline materialize_buffer(bc::StencilBroadcasted) = constant_field(
    apply_stencil!(
        buffer_similar(space_template(axes(bc), DataLayouts.DataScope(bc)), eltype(bc)),
        bc,
        Val(false),
    ),
)
@inline register_values(bc::StencilBroadcasted) = constant_field(
    apply_stencil!(
        register_similar(space_template(axes(bc), DataLayouts.DataScope(bc)), eltype(bc)),
        bc,
        Val(false),
    ),
)

# Evaluation of a stencil expression (an operator, or a pointwise expression
# over operators) over one column, with every point of the result computed by
# its own thread and written into dest. On devices where reading cached
# arguments can require lockstep, every thread of the column evaluates the
# stencil at the same time (see update_points!). The first synchronization makes
# the values that the pass reads visible to every thread, and the last one keeps
# them from being overwritten before every thread has read them. A cached
# argument skips the last one, since the operator that reads it synchronizes
# after it is evaluated.
#
# Only a pass that caches arguments reads values written by other threads, so a
# pass that caches none skips both synchronizations. Like the buffer of a cached
# pointwise argument (see materialize_buffer), its destination is written after
# every earlier pass has finished reading its own buffers: an operator that caches
# arguments is never evaluated into a cached argument (see is_fused), so every
# pass that reads buffers ends with a synchronization. This relies on the results
# of operators being in registers (see register_similar); in a scope where they
# cannot be, other threads read them directly, so every pass synchronizes.
@inline function apply_stencil!(dest, bc, finish)
    scope = DataLayouts.DataScope(bc)
    space = axes(bc)
    stencil_bc = stencil_arg(bc, Val(false))
    syncs =
        num_caching_operators(bc) > 0 ||
        !DataLayouts.has_scope_registers(
            Fields.field_values(space_template(space, scope)),
        )
    syncs && DataLayouts.synchronize(scope)
    dest_data = Fields.field_values(dest)
    if has_private_buffers(bc)
        # A single thread owns the whole column (as on CPUs), so the point loop
        # is entered directly with the point update, which is the loop that
        # update_points! runs on this scope. Entering it through update_points!
        # adds two method instances per expression (update_points! and its
        # update_point! closure), each optimized with the whole stencil inlined.
        DataLayouts.scoped_slice_loop(
            DataLayouts.ThisThread(),
            DataLayouts.ThisThread(),
            view,
            StencilPointUpdate(stencil_bc, space),
            DataLayouts.NoMask(),
            Val(true),
            dest_data,
        )
    else
        DataLayouts.update_points!(nothing, dest_data, view) do index
            @inbounds column_value(stencil_bc, space, vertical_index(space, index))
        end
    end
    syncs && finish isa Val{true} && DataLayouts.synchronize(scope)
    return dest
end

# Point update of apply_stencil!, as a callable struct rather than a closure so
# that it can be defined outside of apply_stencil! (see above).
struct StencilPointUpdate{B, S}
    stencil_bc::B
    space::S
end
@inline function (update::StencilPointUpdate)(index, point)
    (; stencil_bc, space) = update
    @inbounds point[] = column_value(stencil_bc, space, vertical_index(space, index))
    return nothing
end
@inline apply_stencil(bc) = constant_field(
    apply_stencil!(
        register_similar(space_template(axes(bc), DataLayouts.DataScope(bc)), eltype(bc)),
        bc,
        Val(true),
    ),
)

@inline apply_operator(op::FiniteDifferenceOperator, args...) =
    apply_stencil(
        Broadcast.Broadcasted(StencilStyle(), op, args, return_space(op, args...)),
    )

abstract type InterpolationOperator <: FiniteDifferenceOperator end

return_eltype(::InterpolationOperator, arg) = eltype(arg)

function assert_no_bcs(op, kwargs)
    length(kwargs) == 0 && return nothing
    error("$op does not accept boundary conditions.")
end

function assert_valid_bcs(op, kwargs, ::Type{ValidBCs}) where {ValidBCs}
    unrolled_foreach(values(kwargs)) do bc
        @assert bc isa ValidBCs "$op only supports boundary conditions:\n\n\t $ValidBCs.\n\n BCs given:\n\n\t $(values(kwargs))"
    end
    return nothing
end

# `GradientC2F`, `DivergenceC2F`, `CurlC2F` and `UpwindBiasedProductC2F` have no
# `SetValue` boundary stencil of their own; each is exactly expressible with the
# other operators and boundary conditions. When a `SetValue` is requested, those
# constructors return a `DirichletOperator` (defined with the `*_c2f_dirichlet`
# helpers below) instead of an operator of their own type.
has_setvalue_bc(kwargs) = unrolled_any(Base.Fix2(isa, SetValue), values(kwargs))

"""
    InterpolateF2C()

Interpolate from face to center mesh. No boundary conditions are required
(or supported).
"""
struct InterpolateF2C{BCS <: @NamedTuple{}} <: InterpolationOperator
    bcs::BCS
end
function InterpolateF2C(; kwargs...)
    assert_no_bcs("InterpolateF2C", kwargs)
    InterpolateF2C((NamedTuple()))
end

return_space(::InterpolateF2C, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::InterpolateF2C, arg) = ((-half, half),)

"""
    I = InterpolateC2F(;boundaries..)
    I.(x)

Interpolate a center-valued field `x` to faces, using the stencil

```math
I(x)[i] = \\frac{1}{2} (x[i+\\tfrac{1}{2}] + x[i-\\tfrac{1}{2}])
```

Supported boundary conditions are:

  - [`SetValue(x₀)`](@ref): set the value at the boundary face to be `x₀`. On the
    left boundary the stencil is

```math
I(x)[\\tfrac{1}{2}] = x₀
```

  - [`Extrapolate`](@ref): use the closest interior point as the boundary value.
    At the left boundary the stencil is

```math
I(x)[\\tfrac{1}{2}] = x[1]
```
"""
struct InterpolateC2F{BCS} <: InterpolationOperator
    bcs::BCS
    function InterpolateC2F(; kwargs...)
        assert_valid_bcs(
            "InterpolateC2F",
            kwargs,
            Union{SetValue, Extrapolate},
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    InterpolateC2F(bcs) = InterpolateC2F(; bcs...)
end

return_space(::InterpolateC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::InterpolateC2F, arg) = ((-half, half),)
# Only the boundary face reads a ghost point, and every order of extrapolation
# reduces to the closest interior point there (see MatrixFields.clip_row).
boundary_width(::InterpolateC2F, ::Extrapolate, args...) = 1

"""
    B = BottomBiasedC2F(;boundaries)
    B.(x)

Interpolate a center-valued field to a face-valued field from below.

```math
B(x)[i] = x[i-\\tfrac{1}{2}]
```

Only the bottom boundary condition can be set. The supported condition is:

  - [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.

```math
B(x)[\\tfrac{1}{2}] = x_0
```
"""
struct BottomBiasedC2F{BCS} <: InterpolationOperator
    bcs::BCS
    function BottomBiasedC2F(; kwargs...)
        assert_valid_bcs("BottomBiasedC2F", kwargs, SetValue)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    BottomBiasedC2F(bcs) = BottomBiasedC2F(; bcs...)
end

return_space(::BottomBiasedC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::BottomBiasedC2F, arg) = ((-half, -half),)

"""
    B = BottomBiasedF2C(;boundaries)
    B.(x)

Interpolate a face-valued field to a center-valued field from below.

```math
B(x)[i] = x[i-\\tfrac{1}{2}]
```

Only the bottom boundary condition can be set. The supported condition is:

  - [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.

```math
B(x)[1] = x_0
```
"""
struct BottomBiasedF2C{BCS} <: InterpolationOperator
    bcs::BCS
    function BottomBiasedF2C(; kwargs...)
        assert_valid_bcs("BottomBiasedF2C", kwargs, SetValue)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    BottomBiasedF2C(bcs) = BottomBiasedF2C(; bcs...)
end

return_space(::BottomBiasedF2C, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::BottomBiasedF2C, arg) = ((-half, -half),)

Base.@propagate_inbounds stencil_interior(
    ::BottomBiasedF2C,
    space,
    idx,
    arg,
) = column_value(arg, space, idx - half)
Base.@propagate_inbounds function stencil_left_boundary(
    ::BottomBiasedF2C,
    bc::SetValue,
    space,
    idx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    boundary_value(bc.val, space, idx)
end

"""
    T = TopBiasedC2F(;boundaries)
    T.(x)

Interpolate a center-valued field to a face-valued field from above.

```math
T(x)[i] = x[i+\\tfrac{1}{2}]
```

Only the top boundary condition can be set. The supported condition is:

  - [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.

```math
T(x)[n+\\tfrac{1}{2}] = x_0
```
"""
struct TopBiasedC2F{BCS} <: InterpolationOperator
    bcs::BCS
    function TopBiasedC2F(; kwargs...)
        assert_valid_bcs("TopBiasedC2F", kwargs, SetValue)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    TopBiasedC2F(bcs) = TopBiasedC2F(; bcs...)
end

return_space(::TopBiasedC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::TopBiasedC2F, arg) = ((half, half),)

"""
    T = TopBiasedF2C(;boundaries)
    T.(x)

Interpolate a face-valued field to a center-valued field from above.

```math
T(x)[i] = x[i+\\tfrac{1}{2}]
```

Only the top boundary condition can be set. The supported condition is:

  - [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.

```math
T(x)[n] = x_0
```
"""
struct TopBiasedF2C{BCS} <: InterpolationOperator
    bcs::BCS
    function TopBiasedF2C(; kwargs...)
        assert_valid_bcs("TopBiasedF2C", kwargs, SetValue)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    TopBiasedF2C(bcs) = TopBiasedF2C(; bcs...)
end

return_space(::TopBiasedF2C, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::TopBiasedF2C, arg) = ((half, half),)

abstract type WeightedInterpolationOperator <: InterpolationOperator end

return_eltype(::WeightedInterpolationOperator, weights, arg) =
    Utilities.return_type(*, Tuple{eltype(weights), eltype(arg)})

"""
    WI = WeightedInterpolateF2C(; boundaries)
    WI.(w, x)

Interpolate a face-valued field `x` to centers, weighted by a face-valued field
`w`, using the stencil

```math
WI(w, x)[i] = \\frac{
        w[i+\\tfrac{1}{2}] x[i+\\tfrac{1}{2}] +  w[i-\\tfrac{1}{2}] x[i-\\tfrac{1}{2}]
    }{
        w[i+\\tfrac{1}{2}] + w[i-\\tfrac{1}{2}]
    }
```

No boundary conditions are required (or supported).
"""
struct WeightedInterpolateF2C{BCS <: @NamedTuple{}} <:
       WeightedInterpolationOperator
    bcs::BCS
end

function WeightedInterpolateF2C(; kwargs...)
    assert_no_bcs("WeightedInterpolateF2C", kwargs)
    WeightedInterpolateF2C(NamedTuple(kwargs))
end

return_space(::WeightedInterpolateF2C, weight, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::WeightedInterpolateF2C, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateF2C,
    space,
    idx,
    weight,
    arg,
)
    w⁺ = column_value(weight, space, idx + half)
    w⁻ = column_value(weight, space, idx - half)
    a⁺ = column_value(arg, space, idx + half)
    a⁻ = column_value(arg, space, idx - half)
    (w⁺ * a⁺ + w⁻ * a⁻) / (w⁺ + w⁻)
end

"""
    WI = WeightedInterpolateC2F(; boundaries)
    WI.(w, x)

Interpolate a center-valued field `x` to faces, weighted by a center-valued field
`w`, using the stencil

```math
WI(w, x)[i] = \\frac{
    w[i+\\tfrac{1}{2}] x[i+\\tfrac{1}{2}] +  w[i-\\tfrac{1}{2}] x[i-\\tfrac{1}{2}]
}{
    w[i+\\tfrac{1}{2}] + w[i-\\tfrac{1}{2}]
}
```

Supported boundary conditions are:

  - [`SetValue(val)`](@ref): set the value at the boundary face to be `val`.
  - [`Extrapolate`](@ref): use the closest interior point as the boundary value.

These have the same stencil as in [`InterpolateC2F`](@ref).
"""
struct WeightedInterpolateC2F{BCS} <: WeightedInterpolationOperator
    bcs::BCS
    function WeightedInterpolateC2F(; kwargs...)
        assert_valid_bcs(
            "WeightedInterpolateC2F",
            kwargs,
            Union{SetValue, Extrapolate},
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    WeightedInterpolateC2F(bcs) = WeightedInterpolateC2F(; bcs...)
end

return_space(::WeightedInterpolateC2F, weight, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::WeightedInterpolateC2F, weight, arg) =
    ((-half, half), (-half, half))
# Only the boundary face reads a ghost point, and every order of extrapolation
# reduces to the closest interior point there (see MatrixFields.clip_row).
boundary_width(::WeightedInterpolateC2F, ::Extrapolate, args...) = 1
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateC2F,
    space,
    idx,
    weight,
    arg,
)
    w⁺ = column_value(weight, space, idx + half)
    w⁻ = column_value(weight, space, idx - half)
    a⁺ = column_value(arg, space, idx + half)
    a⁻ = column_value(arg, space, idx - half)
    (w⁺ * a⁺ + w⁻ * a⁻) / (w⁺ + w⁻)
end

# WeightedInterpolateC2F has no stencil_left_boundary/stencil_right_boundary methods:
# every broadcast over it is rewritten as an operator matrix multiply when it is
# instantiated, so its boundary values come from the operator matrix (for Extrapolate)
# or from a SetBoundaryOperator applied to the result (for SetValue); see
# MatrixFields/operator_matrices.jl.

"""
    AdvectionOperator

Supertype of the advection operators, e.g. [`UpwindBiasedProductC2F`](@ref) and
[`FCTZalesak`](@ref). Given a face-valued velocity field `v` and a
center-valued field `x`, for each face `i` an advection operator computes a
function of the form
`f(v[i-1], v[i], v[i+1], x[i-3/2], x[i-1/2], x[i+1/2], x[i+3/2])` or
`f(v[i], x[i-3/2], x[i-1/2], x[i+1/2], x[i+3/2])`
and returns a contravariant3 component. On non-periodic domains, all faces are
treated like interior faces, padding out-of-range stencil points with ghost
values (on periodic domains, indices wrap around instead):

  - The out-of-range values of the advected field are padded with the
    [`Extrapolate`](@ref) boundary condition for that boundary: every ghost
    point the stencil reaches takes the value extrapolated from the in-range
    interior points of the stencil (the extrapolation order is reduced at the
    boundary face itself, where fewer interior points are in range). The only
    supported boundary conditions are `Extrapolate{N}`
    ([`Outflow(; order = N)`](@ref Outflow) is the physically named
    constructor); when an advection operator is constructed with no boundary
    conditions, `Extrapolate{0}` is added to its `bcs`, and a boundary whose
    name has no entry in `bcs` also falls back to `Extrapolate{0}`.
  - The velocity field's out-of-range face indices are clamped to the domain.

An advection operator whose interior stencil is linear in the advected
argument (see `Operators.has_linear_stencil`) is rewritten as an
operator-matrix multiply when it is broadcasted (see
`MatrixFields.operator_matrix`), with the ghost-point extrapolations folded
into its matrix's boundary rows; every other advection operator is evaluated
pointwise, through the callable interface described below.

!!! note

    The ghost-point reconstruction continues the field along the third
    coordinate line. On a terrain-following grid, the boundary is the
    coordinate surface ``\\xi^3`` = const, and continuation along the wall
    would instead require horizontal derivatives, which a vertical stencil
    cannot compute (e.g. the closest-value padding continues the field with a
    zero derivative along the third coordinate line). The flux through the
    boundary surface is imposed by the enclosing operator instead.

By default, the operator is assumed to be a function of the velocity at the
current face only. An operator that is a function of the velocity at
neighboring faces defines

`Operators.advection_velocity_width(::SomeAdvectionOperator) = Val(:neighboring)`

and is then evaluated with the velocity at neighboring faces as well. The
default is `Val(:current)`.

The advected field is the broadcast argument following the velocity. An
operator that is a function of the stencils of multiple center-valued
quantities (e.g. [`FCTZalesak`](@ref)) should take a single center-valued field
whose elements are tuples of those quantities (e.g. `op.(v, tuple.(x, y))`);
each of the 4 stencil values passed to the operator is then such a tuple.

Subtypes of this abstract type that are evaluated pointwise should be
callable, with a method of the form:
`(::SomeAdvectionOperator)(v, x⁻⁻, x⁻, x⁺, x⁺⁺, extra_params...)`
or
`(::SomeAdvectionOperator)(v⁻, v, v⁺, x⁻⁻, x⁻, x⁺, x⁺⁺, extra_params...)`
if the operator is a function of the velocity at neighboring faces. All
velocity arguments are supplied as the contravariant3 component of the
face-valued velocity field, and `extra_params` are any broadcast arguments
beyond the velocity and advected field (e.g. `dt`), evaluated at the current
face and passed through as is. In particular, a vector-valued extra parameter
is not converted: an operator that needs one in contravariant form (e.g. a
velocity used only to determine the upwind direction, as in
[`TVDLimitedFluxC2F`](@ref)) should require its callers to supply it as
contravariant data. Subtypes that are instead rewritten as operator-matrix
multiplies define their matrix rows in `MatrixFields/operator_matrices.jl`.
"""
abstract type AdvectionOperator <: FiniteDifferenceOperator end

"""
    has_linear_stencil(op::AdvectionOperator)

Return whether `op` is a linear function of its advected argument, in the
interior and at the boundaries, so that it can be rewritten as an operator-matrix
multiply (see `MatrixFields.operator_matrix`). This requires an operator type
with a linear interior stencil (`has_linear_interior`) and a linear
ghost-point reconstruction (`is_linear_reconstruction`) at every boundary.
"""
has_linear_stencil(op::AdvectionOperator) =
    has_linear_interior(op) && unrolled_all(is_linear_reconstruction, op.bcs)
has_linear_interior(::AdvectionOperator) = false
is_linear_reconstruction(::Extrapolate) = true

# When no boundary conditions are supplied, advection operators default to
# Extrapolate{0} at both boundaries (`bottom` and `top` are the canonical
# vertical boundary names).
const default_advection_bcs = (; bottom = Extrapolate(), top = Extrapolate())
advection_bcs(kwargs) =
    isempty(kwargs) ? default_advection_bcs : NamedTuple(kwargs)

# Advection operators never use NullBoundaryCondition: a boundary whose name
# has no entry in `bcs` (e.g. when the vertical boundaries are not named
# `bottom` and `top`) gets the default Extrapolate{0} reconstruction.
get_boundary(
    op::AdvectionOperator,
    ::LeftBoundaryWindow{name},
) where {name} = get_advection_boundary(op.bcs, name)
get_boundary(
    op::AdvectionOperator,
    ::RightBoundaryWindow{name},
) where {name} = get_advection_boundary(op.bcs, name)
get_advection_boundary(bcs::NamedTuple, name::Symbol) =
    hasfield(typeof(bcs), name) ? getfield(bcs, name) : Extrapolate()

# A `bcs` entry whose name matches neither of the space's boundary names would
# silently be ignored in favor of the Extrapolate{0} fallback above, so catch
# mismatched names when the space is known (at broadcast instantiation, via
# `return_space`). Names that miss are only tolerated when `bcs` is exactly
# the default inserted by `advection_bcs`, whose reconstruction the fallback
# reproduces at any boundary name; on periodic spaces there are no boundaries
# and `bcs` is ignored altogether. Everything here is in the type domain, so
# the check folds away for valid broadcasts.
@inline function assert_valid_advection_bc_names(op::AdvectionOperator, space)
    op.bcs === default_advection_bcs && return nothing
    Topologies.isperiodic(space) && return nothing
    names =
        (Spaces.left_boundary_name(space), Spaces.right_boundary_name(space))
    unrolled_all(in(names), keys(op.bcs)) ||
        error(invalid_advection_bc_names_string(op, Val(names)))
    return nothing
end
@generated invalid_advection_bc_names_string(
    op,
    ::Val{space_names},
) where {
    space_names,
} = "Every boundary condition of $(op.name.name) must be named after a boundary \
     of the space ($(join(space_names, ", "))); \
     got ($(join(fieldnames(fieldtype(op, :bcs)), ", ")))"
# The stencil returns Contravariant3Vector(op(...)), where op's result combines
# the contravariant3 component of a velocity element with the advected field's
# stencil values using ordinary arithmetic. For tuple-valued fields, both of
# these are (nested) AutoBroadcasters of scalars, and the Contravariant3Vector
# constructor broadcasts over the nesting, so that e.g. an NTuple-valued
# advected field produces an NTuple of Contravariant3Vectors. The extra
# parameters do not affect the result type.

# Scalar structure of the contravariant3 component of a velocity element.
velocity_component_type(::Type{X}) where {X <: AutoBroadcaster} =
    nested_broadcast_result_type(velocity_component_type, X)
velocity_component_type(::Type{T}) where {T} = eltype(T)

# Scalar structure of an advected value in the stencil arithmetic. Operators
# whose advected field holds tuples of center-valued quantities (e.g.
# FCTZalesak) destructure each stencil value and combine the quantities
# componentwise, so they define this as the tuple's component type instead.
advected_component_type(op, ::Type{Tx}) where {Tx} = Tx

advection_eltype(::Type{X}) where {X <: AutoBroadcaster} =
    nested_broadcast_result_type(advection_eltype, X)
advection_eltype(::Type{T}) where {T} =
    Geometry.Contravariant3Vector{T}

function return_eltype(op::AdvectionOperator, V, arg, extra_params...)
    # eltype may be the inference-failure sentinel Union{} when this is called
    # while probing an expression with unsafe_eltype; propagate it instead of
    # dispatching on it (Union{} is a subtype of everything).
    (eltype(V) == Union{} || eltype(arg) == Union{}) && return Union{}
    return advection_eltype(
        Utilities.return_type(
            *,
            Tuple{
                velocity_component_type(eltype(V)),
                advected_component_type(op, add_auto_broadcasters(eltype(arg))),
            },
        ),
    )
end
function return_space(op::AdvectionOperator, V, arg, extra_params...)
    assert_valid_advection_bc_names(op, axes(V))
    return axes(V)
end
advection_velocity_width(::AdvectionOperator) = Val(:current)
velocity_stencil_width(::Val{:current}) = (0, 0)
velocity_stencil_width(::Val{:neighboring}) = (-1, 1)

# All faces are computed with the interior stencil, which applies the
# ghost-point extrapolations itself, so no face needs a boundary stencil. (The
# operators that are rewritten as matrix multiplies fold the ghost points into
# their matrix rows instead; see MatrixFields/operator_matrices.jl.)
Base.@propagate_inbounds stencil_value(op::AdvectionOperator, space, idx, args...) =
    stencil_interior(op, space, idx, args...)
reads_in_lockstep(::AdvectionOperator) = true

stencil_interior_width(
    op::AdvectionOperator,
    velocity,
    arg,
    extra_params...,
) = (
    velocity_stencil_width(advection_velocity_width(op)),
    (-half - 1, half + 1),
    map(Returns((0, 0)), extra_params)...,
)
reads_neighbors(op::AdvectionOperator, velocity, arg, extra_params...) = (
    Val(advection_velocity_width(op) isa Val{:neighboring}),
    Val(true),
    unrolled_map(Returns(Val(false)), extra_params)...,
)

Base.@propagate_inbounds function advection_velocities(
    ::Val{:current},
    space,
    idx,
    velocity,
)
    v = Geometry.contravariant3(
        column_value(velocity, space, idx),
        Geometry.LocalGeometry(space, idx),
    )
    return (v,)
end

Base.@propagate_inbounds function advection_velocities(
    ::Val{:neighboring},
    space,
    idx,
    velocity,
)
    idx⁻ = column_index(space, idx - 1)
    idx⁺ = column_index(space, idx + 1)
    v⁻ = Geometry.contravariant3(
        column_value(velocity, space, idx⁻),
        Geometry.LocalGeometry(space, idx⁻),
    )
    v = Geometry.contravariant3(
        column_value(velocity, space, idx),
        Geometry.LocalGeometry(space, idx),
    )
    v⁺ = Geometry.contravariant3(
        column_value(velocity, space, idx⁺),
        Geometry.LocalGeometry(space, idx⁺),
    )
    return (v⁻, v, v⁺)
end

"""
    advection_ghost_values(op, space, nghost_left, nghost_right, a⁻⁻, a⁻, a⁺, a⁺⁺)

Replace the values of the out-of-range centers of an [`AdvectionOperator`](@ref)
stencil, the first `nghost_left` and the last `nghost_right` of its 4 values
(see `ghost_counts`), with the extrapolation of `op`'s boundary condition for
that boundary from the in-range interior points of the stencil: at the boundary
face itself, both out-of-range centers share the extrapolation from the 2
in-range points, and at the face one in from the boundary the single
out-of-range center is extrapolated from the 3 in-range points (see
[`AdvectionOperator`](@ref)). The two boundaries are handled with independent
branches: on a 2-center column, the middle face is one in from both boundaries,
so both of its out-of-range centers need their ghost-point extrapolations, each
from the only 2 in-range points. Every other face is unaffected, and keeps the
closest-value padding of the caller's index clamping.

The operator matrices of the linear advection operators fold the coefficients
of the same ghost points into their in-range entries (see
`MatrixFields.clip_row`), so the linear and nonlinear operators share their
boundary semantics.
"""
@inline function advection_ghost_values(
    op::AdvectionOperator,
    space,
    nghost_left,
    nghost_right,
    a⁻⁻,
    a⁻,
    a⁺,
    a⁺⁺,
)
    if nghost_left == 2
        bc = get_boundary(op, left_boundary_window(space))
        a⁻⁻ = a⁻ = bc(a⁺, a⁺⁺)
    elseif nghost_left == 1
        bc = get_boundary(op, left_boundary_window(space))
        a⁻⁻ = nghost_right == 1 ? bc(a⁻, a⁺) : bc(a⁻, a⁺, a⁺⁺)
    end
    if nghost_right == 2
        bc = get_boundary(op, right_boundary_window(space))
        a⁺⁺ = a⁺ = bc(a⁻, a⁻⁻)
    elseif nghost_right == 1
        bc = get_boundary(op, right_boundary_window(space))
        a⁺⁺ = nghost_left == 1 ? bc(a⁺, a⁻) : bc(a⁺, a⁻, a⁻⁻)
    end
    return (a⁻⁻, a⁻, a⁺, a⁺⁺)
end

# we treat all faces like interior faces: stencil values are read at indices
# clamped to the column (indices wrap on periodic domains instead), and the
# values of the out-of-range points are then replaced by the boundary
# condition's ghost-point extrapolation from the in-range interior points of
# the stencil (a no-op with the default Extrapolate{0}, which matches the
# clamping). On terrain-following grids, the extrapolation is along the third
# coordinate line, not along the wall normal; see the note on AdvectionOperator.
Base.@propagate_inbounds function stencil_interior(
    op::AdvectionOperator,
    space,
    idx,
    velocity,
    arg,
    extra_params...,
)
    a⁻⁻ = column_value(arg, space, column_index(space, idx - half - 1))
    a⁻ = column_value(arg, space, column_index(space, idx - half))
    a⁺ = column_value(arg, space, column_index(space, idx + half))
    a⁺⁺ = column_value(arg, space, column_index(space, idx + half + 1))
    if !Topologies.isperiodic(space)
        (a⁻⁻, a⁻, a⁺, a⁺⁺) = advection_ghost_values(
            op,
            space,
            ghost_counts(space, idx, -half - 1, half + 1)...,
            a⁻⁻,
            a⁻,
            a⁺,
            a⁺⁺,
        )
    end
    vs = advection_velocities(advection_velocity_width(op), space, idx, velocity)
    params = extra_param_values(space, idx, extra_params...)
    return Geometry.Contravariant3Vector(
        op(vs..., a⁻⁻, a⁻, a⁺, a⁺⁺, params...),
    )
end

# The values of the extra parameters of an advection operator at idx, read like
# its other arguments (a map over a closure is not inlined on CPUs, where it is a
# call at every point, and it drops the caller's inbounds context).
Base.@propagate_inbounds extra_param_values(space, idx) = ()
Base.@propagate_inbounds extra_param_values(space, idx, param, params...) =
    (column_value(param, space, idx), extra_param_values(space, idx, params...)...)

"""
    U = UpwindBiasedProductC2F(;boundaries)
    U.(v, x)

Compute the product of the face-valued vector field `v` and a center-valued
field `x` at cell faces by upwinding `x` according to the direction of `v`.

More precisely, it is computed based on the sign of the 3rd contravariant
component, and it returns a `Contravariant3Vector`:

```math
U(\\boldsymbol{v},x)[i] = \\begin{cases}
  v^3[i] x[i-\\tfrac{1}{2}]\\boldsymbol{e}_3 \\textrm{, if } v^3[i] > 0 \\\\
  v^3[i] x[i+\\tfrac{1}{2}]\\boldsymbol{e}_3 \\textrm{, if } v^3[i] < 0
  \\end{cases}
```

where ``\\boldsymbol{e}_3`` is the 3rd covariant basis vector.

The only supported boundary condition is [`Extrapolate`](@ref)
([`Outflow`](@ref)), which is also added to `bcs` (as `Extrapolate{0}`) by
default when no boundary conditions are given: boundary faces are computed with
the interior stencil, padding the
ghost point it reaches with the boundary condition's extrapolation. The
stencil only reaches a ghost point at the boundary face itself, where a single
interior point is in range, so every extrapolation order reduces to the value
of the closest interior point: since the padded upwind and downwind values
then coincide, the boundary faces reduce to ``v^3[i] x_b \\boldsymbol{e}_3``,
where ``x_b`` is the value at the center closest to the boundary.

To prescribe the value of `x` used on the outside of a boundary instead, pass
a [`SetValue`](@ref): the constructor then returns a
[`DirichletOperator`](@ref) that applies
[`upwind_biased_product_c2f_dirichlet`](@ref), which reproduces the `SetValue`
boundary stencil exactly and fuses into an enclosing broadcast with lazy
boundary rows.
"""
struct UpwindBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
    function UpwindBiasedProductC2F(; kwargs...)
        has_setvalue_bc(kwargs) &&
            return DirichletOperator{UpwindBiasedProductC2F}(kwargs)
        assert_valid_bcs("UpwindBiasedProductC2F", kwargs, Extrapolate)
        bcs = advection_bcs(kwargs)
        new{typeof(bcs)}(bcs)
    end
    UpwindBiasedProductC2F(bcs) = UpwindBiasedProductC2F(; bcs...)
end
has_linear_interior(::UpwindBiasedProductC2F) = true

return_eltype(::UpwindBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

upwind_biased_product(v, a⁻, a⁺) = ((v + abs(v)) * a⁻ + (v - abs(v)) * a⁺) / 2

stencil_interior_width(::UpwindBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half, half))

"""
    LVL = LinVanLeerC2F(; constraint)
    LVL.(v, x, dt)

Compute the product of the face-valued vector field `v` and a center-valued
field `x` at cell faces using a slope-limited reconstruction of `x`, following
the van Leer class of limiters of [Lin1994](@cite). `dt` is the time step,
which enters the limiter through the local upwind CFL number. Four limiter
`constraint` options are provided:

  - `AlgebraicMean()`: Algebraic mean, which guarantees neither positivity nor
    monotonicity (eq. 2, `avg`).
  - `PositiveDefinite()`: Positive-definite with implicit diffusion based on
    local stencil extrema (eqs. 3b, 3c, 5a, 5b, `posd`).
  - `MonotoneHarmonic()`: Monotonicity-preserving harmonic mean, which implies a
    strong monotonicity constraint (eq. 4, `mono4`).
  - `MonotoneLocalExtrema()`: Monotonicity-preserving, with extrema bounded by
    the edge cells in the stencil (eq. 5, `mono5`).

The diffusion implied by these methods is proportional to the local upwind CFL
number. The mismatch Δ𝜙 = 0 returns the first-order upwind method. Special
cases discussed in [Lin1994](@cite), such as setting 𝜙_min = 0 or 𝜙_max to the
saturation mixing ratio for water vapor, are not considered here in favour of
the generalized local extrema in eqs. (5a, 5b).

As for all [`AdvectionOperator`](@ref)s, boundary faces are computed with the
interior stencil, padding ghost points with the [`Extrapolate`](@ref)
([`Outflow`](@ref)) boundary condition's extrapolation (`Extrapolate{0}` is
added to `bcs` by default when no boundary conditions are given).
"""
struct LinVanLeerC2F{BCS, C} <: AdvectionOperator
    bcs::BCS
    constraint::C
end
function LinVanLeerC2F(; constraint, kwargs...)
    assert_valid_bcs("LinVanLeerC2F", kwargs, Extrapolate)
    LinVanLeerC2F(advection_bcs(kwargs), constraint)
end

@inline (op::LinVanLeerC2F)(v, a⁻⁻, a⁻, a⁺, a⁺⁺, dt) =
    slope_limited_product(v, a⁻, a⁻⁻, a⁺, a⁺⁺, dt, op.constraint)

"""
    LimiterConstraint

Supertype of the `constraint` options of [`LinVanLeerC2F`](@ref), which select how
the slope of the reconstructed field is limited: [`AlgebraicMean`](@ref),
[`PositiveDefinite`](@ref), [`MonotoneHarmonic`](@ref), [`MonotoneLocalExtrema`](@ref).
"""
abstract type LimiterConstraint end

"""
    AlgebraicMean()

[`LimiterConstraint`](@ref) for [`LinVanLeerC2F`](@ref): the slope is the algebraic
mean of the two one-sided differences, scaled by `1 - |CFL|`. It guarantees neither
positivity nor monotonicity (eq. 2, `avg`, of [Lin1994](@cite)).
"""
struct AlgebraicMean <: LimiterConstraint end

"""
    PositiveDefinite()

[`LimiterConstraint`](@ref) for [`LinVanLeerC2F`](@ref): the mean slope is bounded by
twice the distance from the topmost stencil value to the local minimum and maximum,
which keeps the reconstruction positive with implicit diffusion (eqs. 3b, 3c, 5a, 5b,
`posd`, of [Lin1994](@cite)).
"""
struct PositiveDefinite <: LimiterConstraint end

"""
    MonotoneHarmonic()

[`LimiterConstraint`](@ref) for [`LinVanLeerC2F`](@ref): the slope is the harmonic
mean of the two one-sided differences when they have the same sign and zero
otherwise, a strong monotonicity constraint (eq. 4, `mono4`, of [Lin1994](@cite)).
"""
struct MonotoneHarmonic <: LimiterConstraint end

"""
    MonotoneLocalExtrema()

[`LimiterConstraint`](@ref) for [`LinVanLeerC2F`](@ref): the mean slope is bounded so
that the reconstructed values stay within the minimum and maximum of the three-cell
stencil, preserving monotonicity (eq. 5, `mono5`, of [Lin1994](@cite)).
"""
struct MonotoneLocalExtrema <: LimiterConstraint end

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::MonotoneLocalExtrema)
    Δ𝜙_avg = ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2
    min𝜙 = min(a⁻, a⁰, a⁺)
    max𝜙 = max(a⁻, a⁰, a⁺)
    𝛼 = min(abs(Δ𝜙_avg), 2 * (a⁰ - min𝜙), 2 * (max𝜙 - a⁰))
    return sign(Δ𝜙_avg) * 𝛼 * (1 - sign(v) * v * dt)
end

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::MonotoneHarmonic)
    Δ𝜙_avg = ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2
    c = sign(v) * v * dt
    if sign(a⁰ - a⁻) == sign(a⁺ - a⁰) && Δ𝜙_avg != 0
        return ((a⁰ - a⁻) * (a⁺ - a⁰)) / (Δ𝜙_avg) * (1 - c)
    else
        return zero(v)
    end
end

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::PositiveDefinite)
    Δ𝜙_avg = ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2
    min𝜙 = min(a⁻, a⁰, a⁺)
    max𝜙 = max(a⁻, a⁰, a⁺)
    return sign(Δ𝜙_avg) *
           min(abs(Δ𝜙_avg), 2 * max(a⁺ - min𝜙, zero(a⁺)), 2 * max(max𝜙 - a⁺, zero(a⁺))) *
           (1 - sign(v) * v * dt)
end

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::AlgebraicMean)
    return ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2 * (1 - sign(v) * v * dt)
end

function slope_limited_product(v, a⁻, a⁻⁻, a⁺, a⁺⁺, dt, constraint)
    # Following Lin et al. (1994)
    # https://doi.org/10.1175/1520-0493(1994)122<1575:ACOTVL>2.0.CO;2
    if v >= 0
        # Eqn (2,5a,5b,5c)
        Δ𝛼 = compute_Δ𝛼_linvanleer(a⁻⁻, a⁻, a⁺, v, dt, constraint)
        return v * (a⁻ + Δ𝛼 / 2)
    else
        # Eqn (2,5a,5b,5c)
        Δ𝛼 = compute_Δ𝛼_linvanleer(a⁻, a⁺, a⁺⁺, v, dt, constraint)
        return v * (a⁺ - Δ𝛼 / 2)
    end
end

"""
    U = Upwind3rdOrderBiasedProductC2F(;boundaries)
    U.(v, x)

Compute the product of a face-valued vector field `v` and a center-valued field
`x` at cell faces by upwinding `x`, to third order of accuracy, according to `v`:

```math
U(v,x)[i] = \\begin{cases}
  v[i] \\left(-2 x[i-\\tfrac{3}{2}] + 10 x[i-\\tfrac{1}{2}] + 4 x[i+\\tfrac{1}{2}] \\right) / 12  \\textrm{, if } v[i] > 0 \\\\
  v[i] \\left(4 x[i-\\tfrac{1}{2}] + 10 x[i+\\tfrac{1}{2}] -2 x[i+\\tfrac{3}{2}]  \\right) / 12  \\textrm{, if } v[i] < 0
  \\end{cases}
```

This stencil is based on [WickerSkamarock2002](@cite), eq. 4(a).

The only supported boundary condition is [`Extrapolate`](@ref)
([`Outflow`](@ref)): boundary faces are computed with the interior stencil,
padding each ghost point it reaches with the condition's extrapolation from the
in-range interior points
(the extrapolation order is reduced at the boundary face itself, where only 2
interior points are in range). When no boundary conditions are given,
`Extrapolate{0}` is added to `bcs` by default. The extrapolations are taken
along the third coordinate line; on a terrain-following grid that is not the
wall-normal direction (see the note on [`AdvectionOperator`](@ref)).
The flux through the boundary itself is not set by this padding: it is
imposed by the enclosing operator, e.g. a [`DivergenceF2C`](@ref) operator
with a [`SetValue`](@ref) boundary.
"""
struct Upwind3rdOrderBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
    function Upwind3rdOrderBiasedProductC2F(; kwargs...)
        assert_valid_bcs(
            "Upwind3rdOrderBiasedProductC2F",
            kwargs,
            Extrapolate,
        )
        bcs = advection_bcs(kwargs)
        new{typeof(bcs)}(bcs)
    end
    Upwind3rdOrderBiasedProductC2F(bcs) =
        Upwind3rdOrderBiasedProductC2F(; bcs...)
end
has_linear_interior(::Upwind3rdOrderBiasedProductC2F) = true

return_eltype(::Upwind3rdOrderBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

stencil_interior_width(::Upwind3rdOrderBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half - 1, half + 1))

"""
    U = FCTBorisBook()
    U.(v, x)

Correct the flux using the flux-corrected transport formulation by Boris and
Book [BorisBook1973](@cite).

# Arguments

  - `v`: A face-valued vector field.
  - `x`: A center-valued field.

```math
Ac(v,x)[i] =
  s[i] \\max \\left\\{0, \\min \\left[ |v[i] |, s[i] \\left( x[i+\\tfrac{3}{2}] - x[i+\\tfrac{1}{2}]  \\right) ,  s[i] \\left( x[i-\\tfrac{1}{2}] - x[i-\\tfrac{3}{2}]  \\right) \\right] \\right\\},
```

where ``s[i] = +1`` if ``v[i] \\geq 0`` and ``s[i] = -1`` if ``v[i] \\leq 0``,
and ``Ac`` represents the resulting corrected antidiffusive flux. This
formulation is based on [BorisBook1973](@cite), as reported in
[durran2010](@cite) section 5.4.1.

As for all [`AdvectionOperator`](@ref)s, boundary faces are computed with the
interior stencil, padding ghost points with the [`Extrapolate`](@ref)
([`Outflow`](@ref)) boundary condition's extrapolation (`Extrapolate{0}` is
added to `bcs` by default when no boundary conditions are given). With the
default, the padded values make
the one-sided difference of `x` on the boundary side vanish at the two faces
nearest each boundary, and that difference bounds the corrected antidiffusive
flux, so the flux is zero there.
"""
struct FCTBorisBook{BCS} <: AdvectionOperator
    bcs::BCS
end
function FCTBorisBook(; kwargs...)
    assert_valid_bcs("FCTBorisBook", kwargs, Extrapolate)
    FCTBorisBook(advection_bcs(kwargs))
end

fct_boris_book(v, a⁻⁻, a⁻, a⁺, a⁺⁺) =
    ifelse(
        iszero(v),
        max(v, min(v, a⁺⁺ - a⁺, a⁻ - a⁻⁻)),
        sign(v) *
        max(zero(v), min(abs(v), sign(v) * (a⁺⁺ - a⁺), sign(v) * (a⁻ - a⁻⁻))),
    )

@inline (op::FCTBorisBook)(v, a⁻⁻, a⁻, a⁺, a⁺⁺) =
    fct_boris_book(v, a⁻⁻, a⁻, a⁺, a⁺⁺)

"""
    U = FCTZalesak()
    U.(A, tuple.(Φ, Φᵗᵈ))

Correct the flux using the flux-corrected transport formulation by Zalesak
[zalesak1979fully](@cite).

# Arguments

  - `A`: A face-valued vector field, the antidiffusive flux.
  - `tuple.(Φ, Φᵗᵈ)`: A center-valued field whose elements are 2-tuples of the
    field `Φ` and its transported-diffused value `Φᵗᵈ`.

```math
Φ_j^{n+1} = Φ_j^{td} - (C_{j+\\frac{1}{2}}A_{j+\\frac{1}{2}} - C_{j-\\frac{1}{2}}A_{j-\\frac{1}{2}})
```

This stencil is based on [zalesak1979fully](@cite), as reported in
[durran2010](@cite) section 5.4.2, where ``C`` denotes the corrected
antidiffusive flux.

As for all [`AdvectionOperator`](@ref)s, boundary faces are computed with the
interior stencil, padding ghost points with the [`Extrapolate`](@ref)
([`Outflow`](@ref)) boundary condition's extrapolation (`Extrapolate{0}` is
added to `bcs` by default when no boundary conditions are given); the
extrapolation of a tuple-valued field
applies to each of `Φ` and `Φᵗᵈ`. No value is imposed at the faces nearest
each boundary: the corrected antidiffusive flux there is whatever the padded
stencil gives.
"""
struct FCTZalesak{BCS} <: AdvectionOperator
    bcs::BCS
end
function FCTZalesak(; kwargs...)
    assert_valid_bcs("FCTZalesak", kwargs, Extrapolate)
    FCTZalesak(advection_bcs(kwargs))
end

advection_velocity_width(::FCTZalesak) = Val(:neighboring)

# each advected stencil value is a 2-tuple of (Φ, Φᵗᵈ), which the operator
# destructures and combines componentwise
advected_component_type(::FCTZalesak, ::Type{Tx}) where {Tx} = eltype(Tx)

@inline function (op::FCTZalesak)(A₋₁, A, A₊₁, x₋₃₂, x₋₁₂, x₊₁₂, x₊₃₂)
    # each stencil value is a 2-tuple of (Φ, Φᵗᵈ)
    (ϕ₋₃₂, ϕ₋₃₂ᵗᵈ) = x₋₃₂
    (ϕ₋₁₂, ϕ₋₁₂ᵗᵈ) = x₋₁₂
    (ϕ₊₁₂, ϕ₊₁₂ᵗᵈ) = x₊₁₂
    (ϕ₊₃₂, ϕ₊₃₂ᵗᵈ) = x₊₃₂
    # 1/dt is in ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂, ϕ₋₃₂ᵗᵈ, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ, ϕ₊₃₂ᵗᵈ

    # 𝒮5.4.2 (1)  Durran (5.32)  Zalesak's cosmetic correction
    # which is usually omitted but used in Durran's textbook
    # implementation of the flux corrected transport method.
    # (Textbook suggests mixed results in 3 reported scenarios)
    A = ifelse(
        max(
            A * (ϕ₊₁₂ᵗᵈ - ϕ₋₁₂ᵗᵈ),
            min(A * (ϕ₊₃₂ᵗᵈ - ϕ₊₁₂ᵗᵈ), A * (ϕ₋₁₂ᵗᵈ - ϕ₋₃₂ᵗᵈ)),
        ) >= 0,
        A,
        zero(A),
    )

    P₋₁₂⁻ = max(0, A) - min(0, A₋₁)
    P₋₁₂⁺ = max(0, A₋₁) - min(0, A)
    P₊₁₂⁻ = max(0, A₊₁) - min(0, A)
    P₊₁₂⁺ = max(0, A) - min(0, A₊₁)

    # 𝒮5.4.2 (2)
    # If flow is nondivergent, ϕᵗᵈ are not needed in the formulae below
    ϕ₋₁₂ᵐᵃˣ = max(ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₋₃₂ᵗᵈ, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ)
    ϕ₋₁₂ᵐⁱⁿ = min(ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₋₃₂ᵗᵈ, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ)
    ϕ₊₁₂ᵐᵃˣ = max(ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ, ϕ₊₃₂ᵗᵈ)
    ϕ₊₁₂ᵐⁱⁿ = min(ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ, ϕ₊₃₂ᵗᵈ)

    # Zalesak also requires, in equation (5.33) Δx/Δt, which for the
    # reference element we may assume Δζ = 1 between interfaces
    R₋₁₂⁻ = ifelse(P₋₁₂⁻ > 0, min(1, (ϕ₋₁₂ᵗᵈ - ϕ₋₁₂ᵐⁱⁿ) / P₋₁₂⁻), zero(A))
    R₋₁₂⁺ = ifelse(P₋₁₂⁺ > 0, min(1, (ϕ₋₁₂ᵐᵃˣ - ϕ₋₁₂ᵗᵈ) / P₋₁₂⁺), zero(A))
    R₊₁₂⁻ = ifelse(P₊₁₂⁻ > 0, min(1, (ϕ₊₁₂ᵗᵈ - ϕ₊₁₂ᵐⁱⁿ) / P₊₁₂⁻), zero(A))
    R₊₁₂⁺ = ifelse(P₊₁₂⁺ > 0, min(1, (ϕ₊₁₂ᵐᵃˣ - ϕ₊₁₂ᵗᵈ) / P₊₁₂⁺), zero(A))

    A_fct = ifelse(A >= 0, min(R₊₁₂⁺, R₋₁₂⁻), min(R₋₁₂⁺, R₊₁₂⁻)) * A
    return A_fct
end

"""
    AbstractTVDSlopeLimiter

Supertype of the TVD slope limiters used by [`TVDLimitedFluxC2F`](@ref), which
documents the general formulation. Each subtype defines the multiplicative
limiter `C(r)` of the slope ratio `r`. Subtypes: `RZeroLimiter`,
`RHalfLimiter`, `RMaxLimiter`, `MinModLimiter`, `KorenLimiter`,
`SuperbeeLimiter`, and `MonotonizedCentralLimiter`.
"""
abstract type AbstractTVDSlopeLimiter end


"""
    RZeroLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = 0`, which returns the low-order
flux.
"""
struct RZeroLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RZeroLimiter) = zero(r)

"""
    RHalfLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = 1/2`.
"""
struct RHalfLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RHalfLimiter) = one(r) / 2

"""
    RMaxLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = 1`, which returns the high-order
flux.
"""
struct RMaxLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RMaxLimiter) = one(r)

"""
    MinModLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = max(0, min(1, r))`.
"""
struct MinModLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::MinModLimiter) = max(0, min(1, r))

"""
    KorenLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = max(0, min(2r, (1 + 2r) / 3, 2))`.
"""
struct KorenLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::KorenLimiter) = max(0, min(2r, (1 + 2r) / 3, 2))

"""
    SuperbeeLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = max(0, min(1, r), min(2, r))`.
"""
struct SuperbeeLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::SuperbeeLimiter) = max(0, min(1, r), min(2, r))

"""
    MonotonizedCentralLimiter()

[`AbstractTVDSlopeLimiter`](@ref) with `C(r) = max(0, min(2r, (1 + r) / 2, 2))`.
"""
struct MonotonizedCentralLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::MonotonizedCentralLimiter) = max(0, min(2r, (1 + r) / 2, 2))

"""
    U = TVDLimitedFluxC2F(; method)
    U.(𝒜, Φ, 𝓊)

Limit the face-valued antidiffusive flux `𝒜` with a TVD slope limiter `method`,
using the center-valued field `Φ` to compute the slope ratio and the face-valued
velocity `𝓊` to determine the upwind direction.

Following the notation of [durran2010](@cite), `𝒜 = ℱʰ - ℱˡ` is the
antidiffusive flux, where the superscripts h and l denote the high- and
low-order (monotone) fluxes. The TVD limiter adjusts the flux to

```math
F_{j+1/2} = F^{l}_{j+1/2} + C_{j+1/2} (F^{h}_{j+1/2} - F^{l}_{j+1/2}),
```

where ``C_{j+1/2}`` is the multiplicative limiter, a function of the ratio `r`
of the upwind slope of `Φ` to the slope across the cell interface. `C = 1`
recovers the high-order flux and `C = 0` the low-order flux. The operator
returns ``C_{j+1/2} 𝒜_{j+1/2}``.

The supported `method`s are the subtypes of [`AbstractTVDSlopeLimiter`](@ref):
`RZeroLimiter()` (returns the low-order flux), `RHalfLimiter()` (flux
multiplier `1/2`), `RMaxLimiter()` (returns the high-order flux),
`MinModLimiter()`, `KorenLimiter()`, `SuperbeeLimiter()`, and
`MonotonizedCentralLimiter()`.

The face-valued velocity `𝓊` is only used to determine the upwind direction,
and must be supplied as contravariant data: either a `Contravariant3Vector`
field, or a scalar field holding the contravariant3 component (e.g.
`Geometry.contravariant3.(u, Fields.local_geometry_field(face_space))` for a
velocity field `u` in another basis).

As for all [`AdvectionOperator`](@ref)s, boundary faces are computed with the
interior stencil, padding ghost points with the [`Extrapolate`](@ref)
([`Outflow`](@ref)) boundary condition's extrapolation (`Extrapolate{0}` is
added to `bcs` by default when no boundary conditions are given). No value is
imposed at the faces nearest each boundary: the limited flux there is whatever
the padded stencil gives.
"""
struct TVDLimitedFluxC2F{BCS, M} <: AdvectionOperator
    bcs::BCS
    method::M
end
function TVDLimitedFluxC2F(; method, kwargs...)
    assert_valid_bcs("TVDLimitedFluxC2F", kwargs, Extrapolate)
    TVDLimitedFluxC2F(advection_bcs(kwargs), method)
end

@inline (op::TVDLimitedFluxC2F)(
    A,
    ϕ₋₃₂,
    ϕ₋₁₂,
    ϕ₊₁₂,
    ϕ₊₃₂,
    𝓊::Geometry.Contravariant3Vector,
) = op(A, ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂, 𝓊.u³)
@inline function (op::TVDLimitedFluxC2F)(A, ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂, 𝓊)
    Δϕ = ϕ₊₁₂ - ϕ₋₁₂ + eps(typeof(ϕ₋₁₂))
    Δϕ_upwind = ifelse(𝓊 >= 0, ϕ₋₁₂ - ϕ₋₃₂, ϕ₊₃₂ - ϕ₊₁₂)
    # a zero upwind slope always gives r = 0, even when Δϕ is also zero (the
    # added eps does not prevent that: ϕ₊₁₂ - ϕ₋₁₂ can be exactly -eps in
    # regions where ϕ is flat up to roundoff, and 0 / 0 would produce NaN);
    # ghost-cell padding also makes the upwind slope exactly zero at a boundary
    # face whose velocity points into the domain
    r = ifelse(Δϕ_upwind == 0, zero(Δϕ_upwind), Δϕ_upwind / Δϕ)
    return limiter_coeff(r, op.method) * A
end

abstract type BoundaryOperator <: FiniteDifferenceOperator end

"""
    SetBoundaryOperator(;boundaries...)

Return the argument unchanged in the interior, and replace the value at each
boundary for which a condition is given. The operator preserves the space of its
argument, so it modifies the boundary faces of a face field or the boundary
center cells of a center field. A side with no condition is left untouched.

The following boundary conditions are supported:

  - [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
  - [`SetGradient(val)`](@ref): set the value to be `val` on the boundary,
    projected onto the `Covariant3` axis.
  - [`SetCurl(val)`](@ref): set the value to be `val` on the boundary, projected
    onto the `Contravariant12` axis (the axis of [`CurlC2F`](@ref)'s output).
  - [`SetDivergence(val)`](@ref): set the value to be `val` on the boundary.

The projecting conditions exist so that this operator can reapply the boundary
conditions of the operator it was derived from when a broadcast is rewritten as
an operator matrix multiply; see `MatrixFields.modifies_output`.
"""
struct SetBoundaryOperator{BCS} <: BoundaryOperator
    bcs::BCS
    function SetBoundaryOperator(; kwargs...)
        assert_valid_bcs(
            "SetBoundaryOperator",
            kwargs,
            Union{SetValue, SetGradient, SetCurl, SetDivergence},
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    SetBoundaryOperator(bcs) = SetBoundaryOperator(; bcs...)
end

return_eltype(::SetBoundaryOperator, arg) = eltype(arg)
fuses_into_stencils(::SetBoundaryOperator) = true
return_space(::SetBoundaryOperator, arg) = axes(arg)

# Whether an expression contains a SetBoundaryOperator, whose values at the
# boundary can have other types than its eltype (e.g., a Contravariant3Vector
# imposed on a Covariant3Vector flux).
has_set_boundary_operator(_) = false
has_set_boundary_operator(bc::Broadcast.Broadcasted) =
    unrolled_any(has_set_boundary_operator, bc.args)
has_set_boundary_operator(::StencilOperatorBroadcasted{<:SetBoundaryOperator}) = true
@drop_recursion_limits has_set_boundary_operator

# The type of the plain value that a SetValue or SetDivergence condition imposes
# (see imposed_boundary_value): that of its value, or of the values of its field.
imposed_value_type(bc::Union{SetValue, SetDivergence}) =
    bc.val isa MaybeLazyField ? eltype(bc.val) : typeof(add_auto_broadcasters(bc.val))

# The value that a boundary condition imposes in place of a value of type T (and
# its type, for a condition value of type V). When T is tuple-valued (an
# AutoBroadcaster, like the values of a NamedTuple-valued field in an operator),
# the condition applies to each component, as in AutoBroadcaster arithmetic: a
# value that is not tuple-valued (e.g., a Contravariant3Vector imposed on a
# NamedTuple of fluxes) is copied into every component, and a tuple-valued one
# is paired with the components. The imposed value then has the type of the
# values it replaces, up to the axes of its tensors (which
# MatrixFields.projected_operand unifies), as every value of a cached argument
# must (see cached_arg).
@inline imposed_value(::Type, value) = value
@inline imposed_value(::Type{T}, value) where {T <: AutoBroadcaster} =
    nested_broadcast((_, component) -> component, new(T), add_auto_broadcasters(value))
imposed_type(::Type, ::Type{V}) where {V} = V
imposed_type(::Type{T}, ::Type{V}) where {T <: AutoBroadcaster, V} =
    typeof(imposed_value(T, new(V)))

# The metric (see Geometry.projected_metric) that projecting the values imposed
# by every SetBoundaryOperator in arg onto axes reads, combined over all of their
# conditions (see Geometry.combine_projected_metrics): nothing when no imposed
# value needs a metric (or arg has no such operator), and all of lg for a value
# that is projected onto the operator's axis before it is imposed (SetGradient
# and SetCurl). Together with the metric for the values of arg's eltype, this is
# the metric for every value that arg can have (see
# MatrixFields.projected_operand).
imposed_values_metric(axes, _, lg) = nothing
imposed_values_metric(axes, bc::Broadcast.Broadcasted, lg) = unrolled_mapreduce(
    arg -> imposed_values_metric(axes, arg, lg),
    (metric1, metric2) -> Geometry.combine_projected_metrics(metric1, metric2, lg),
    bc.args;
    init = nothing,
)
imposed_values_metric(
    axes,
    bc::StencilOperatorBroadcasted{<:SetBoundaryOperator},
    lg,
) = Geometry.combine_projected_metrics(
    unrolled_mapreduce(
        boundary_condition -> imposed_value_metric(axes, boundary_condition, lg),
        (metric1, metric2) -> Geometry.combine_projected_metrics(metric1, metric2, lg),
        values(bc.f.bcs);
        init = nothing,
    ),
    imposed_values_metric(axes, bc.args[1], lg),
    lg,
)
imposed_value_metric(axes, bc::Union{SetValue, SetDivergence}, lg) =
    Geometry.projected_metric(axes, Val(imposed_value_type(bc)), lg)
imposed_value_metric(_, _, lg) = Some(lg)

# Whether every value that a SetBoundaryOperator in arg imposes is a plain value
# of type T (see imposes_plain_value), the type of the values it replaces, so
# that arg only has values of type T.
imposes_only(_, ::Type) = true
imposes_only(bc::Broadcast.Broadcasted, ::Type{T}) where {T} =
    unrolled_all(arg -> imposes_only(arg, T), bc.args)
imposes_only(bc::StencilOperatorBroadcasted{<:SetBoundaryOperator}, ::Type{T}) where {T} =
    unrolled_all(
        boundary_condition -> imposes_plain_value(boundary_condition, T),
        values(bc.f.bcs),
    ) && imposes_only(bc.args[1], T)
@drop_recursion_limits imposed_values_metric, imposes_only

stencil_interior_width(::SetBoundaryOperator, arg) = ((0, 0),)
Base.@propagate_inbounds stencil_interior(
    ::SetBoundaryOperator,
    space,
    idx,
    arg,
) = column_value(arg, space, idx)

# An argument that requires lockstep reads (see requires_lockstep) is read at
# every point, including the boundaries where its value is replaced, so that an
# operator that reads this operator's result in lockstep (see reads_in_lockstep)
# also reads the argument in lockstep. So is an argument without stencils (a
# field or a pointwise expression over fields) whose buffers are not private to
# a thread (see has_private_buffers), as on GPUs, where the threads of a column
# run the same instructions anyway: reading it before branching on the point
# lets an operator issue the reads of every point in its band together, since
# the compiler does not know the range of the point index and cannot fold the
# branch, and only the two boundary points of each column read a value that
# they discard. A stencil argument keeps the branch, which skips its evaluation
# at the boundaries and lets the compiler fold the stencil's own checks.
Base.@propagate_inbounds function stencil_value(
    op::SetBoundaryOperator,
    space,
    idx,
    arg,
)
    # A periodic column has no boundaries (and no boundary windows to look up).
    Topologies.isperiodic(space) && return stencil_interior(op, space, idx, arg)
    left_bc = get_boundary(op, left_boundary_window(space))
    right_bc = get_boundary(op, right_boundary_window(space))
    reads_first =
        requires_lockstep(arg) || (
            !has_private_buffers(arg) &&
            !has_stencils(arg) &&
            imposes_plain_value(left_bc, eltype(arg)) &&
            imposes_plain_value(right_bc, eltype(arg))
        )
    if reads_first
        value = stencil_interior(op, space, idx, arg)
        # The left (bottom) condition is selected last, so that it takes
        # precedence on a column too short to separate the two windows.
        value = boundary_select(
            right_bc,
            space,
            right_idx(space),
            should_call_right_boundary(idx, space, op, arg),
            value,
        )
        return boundary_select(
            left_bc,
            space,
            left_idx(space),
            should_call_left_boundary(idx, space, op, arg),
            value,
        )
    end
    if should_call_left_boundary(idx, space, op, arg)
        return stencil_left_boundary(op, left_bc, space, idx, arg)
    elseif should_call_right_boundary(idx, space, op, arg)
        return stencil_right_boundary(op, right_bc, space, idx, arg)
    end
    return stencil_interior(op, space, idx, arg)
end

# Whether bc imposes a plain value (SetValue or SetDivergence) whose type is T,
# the type of the values it replaces, or no value at all. Such a value can be
# selected with ifelse (see boundary_select); a value that has to be projected
# (SetGradient or SetCurl), or that has another type than the values it
# replaces, is applied with a branch, which is only worth taking after reading
# the argument when the reads require lockstep.
@inline imposes_plain_value(::NullBoundaryCondition, ::Type) = true
@inline imposes_plain_value(_, ::Type) = false
@inline imposes_plain_value(bc::Union{SetValue, SetDivergence}, ::Type{T}) where {T} =
    imposed_type(T, imposed_value_type(bc)) == T

# The value of a SetBoundaryOperator at a point whose argument has already been
# read (see stencil_value): the value that bc imposes at the boundary with
# index boundary_idx replaces the argument's value where is_boundary holds. A
# plain imposed value (SetValue or SetDivergence) of the same type as the
# argument's values is selected with ifelse, so that the compiler does not sink
# the read of the argument into a branch; it is computed at the boundary index,
# where it is a constant or a value at the boundary level. A value that has to
# be projected (SetGradient or SetCurl) or that has another type is selected
# with a branch instead, and a missing condition imposes nothing.
@inline boundary_select(::NullBoundaryCondition, _, _, _, value) = value
Base.@propagate_inbounds boundary_select(bc, space, boundary_idx, is_boundary, value) =
    is_boundary ? imposed_boundary_value(bc, space, boundary_idx, typeof(value)) : value
Base.@propagate_inbounds function boundary_select(
    bc::Union{SetValue, SetDivergence},
    space,
    boundary_idx,
    is_boundary,
    value,
)
    imposed = imposed_boundary_value(bc, space, boundary_idx, typeof(value))
    return imposed isa typeof(value) ? ifelse(is_boundary, imposed, value) :
           is_boundary ? imposed : value
end

# The value a `SetBoundaryOperator` imposes at a boundary. `SetGradient` and `SetCurl`
# hold values in the axis the operator they were taken from writes its output in, so they
# are projected onto that axis; `SetValue` and `SetDivergence` are imposed as given.
# The value may be a constant, a field on the boundary level, or a field or lazy
# broadcast over the whole space (see boundary_value). A lazy value must be
# pointwise (fields combined with pointwise functions): a stencil operator
# inside it would be evaluated at the boundary index of the wrong staggering,
# so its boundary handling breaks down (an operator without boundary conditions
# yields its `NullBoundaryCondition` `NaN`s there).
Base.@propagate_inbounds imposed_boundary_value(
    bc::Union{SetValue, SetDivergence},
    space,
    idx,
) = boundary_value(bc.val, space, idx)
Base.@propagate_inbounds imposed_boundary_value(
    bc::SetGradient,
    space,
    idx,
) = Geometry.project(
    Geometry.Covariant3Axis(),
    boundary_value(bc.val, space, idx),
    Geometry.LocalGeometry(space, idx),
)
# Project onto Contravariant12, not Contravariant123: CurlC2F's operator
# matrix produces Contravariant12Vector entries (see `op_matrix_row_type` for
# CurlFiniteDifferenceOperator), so a wider boundary value would make the
# result a Union of the two vector types across the column.
Base.@propagate_inbounds imposed_boundary_value(bc::SetCurl, space, idx) =
    Geometry.project(
        Geometry.Contravariant12Axis(),
        boundary_value(bc.val, space, idx),
        Geometry.LocalGeometry(space, idx),
    )
# The value imposed in place of a value of type T (see imposed_value).
Base.@propagate_inbounds imposed_boundary_value(bc, space, idx, ::Type{T}) where {T} =
    imposed_value(T, imposed_boundary_value(bc, space, idx))

Base.@propagate_inbounds function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::Union{SetValue, SetGradient, SetCurl, SetDivergence},
    space,
    idx,
    arg,
)
    @boundscheck @assert idx == left_idx(space)
    return imposed_boundary_value(bc, space, idx, eltype(arg))
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::Union{SetValue, SetGradient, SetCurl, SetDivergence},
    space,
    idx,
    arg,
)
    @boundscheck @assert idx == right_idx(space)
    return imposed_boundary_value(bc, space, idx, eltype(arg))
end

abstract type GradientOperator <: FiniteDifferenceOperator end

return_eltype(::GradientOperator, arg) =
    Geometry.gradient_result_type(Val((3,)), eltype(arg))

"""
    G = GradientF2C(;boundaryname=boundarycondition...)
    G.(x)

Compute the gradient of a face-valued field `x`, returning a center-valued
`Covariant3` vector field, using the stencil:

```math
G(x)[i]^3 = x[i+\\tfrac{1}{2}] - x[i-\\tfrac{1}{2}]
```

The usual division factor ``1 / \\Delta z`` of a first-order finite difference
operator is accounted for in the `LocalVector` basis. Hence, users must cast the
output of `GradientF2C` to a `UVector`, `VVector` or `WVector`, according to
the type of domain on which the operator is defined.

The following boundary conditions are supported:

  - By default (no boundary condition), the value of `x` at the boundary face
    is used.
  - [`SetValue(x₀)`](@ref): calculate the gradient assuming the value at the
    boundary is `x₀`. For the left boundary, this becomes:

```math
G(x)[1]³ = x[1+\\tfrac{1}{2}] - x₀
```

  - [`SetGradient(v₀)`](@ref): set the value of the gradient at the center
    closest to the boundary to be `v₀`. For the left boundary, this becomes:

```math
G(x)[1] = v₀
```

As with [`GradientC2F`](@ref), `v₀` is projected onto the covariant 3 axis.
"""
struct GradientF2C{BCS} <: GradientOperator
    bcs::BCS
    function GradientF2C(; kwargs...)
        assert_valid_bcs("GradientF2C", kwargs, Union{SetValue, SetGradient})
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    GradientF2C(bcs) = GradientF2C(; bcs...)
end

return_space(::GradientF2C, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::GradientF2C, arg) = ((-half, half),)

"""
    G = GradientC2F(;boundaryname=boundarycondition...)
    G.(x)

Compute the gradient of a center-valued field `x`, returning a face-valued
`Covariant3` vector field, using the stencil:

```math
G(x)[i]^3 = x[i+\\tfrac{1}{2}] - x[i-\\tfrac{1}{2}]
```

The following boundary conditions are supported:

  - [`SetGradient(v₀)`](@ref): set the value of the gradient at the boundary to be
    `v₀`. For the left boundary, this becomes:
    ```math
    G(x)[\\tfrac{1}{2}] = v₀
    ```

!!! note

    `v₀` is projected onto the covariant 3 axis, so it prescribes
    ``\\partial x / \\partial \\xi^3``, the derivative along the third
    coordinate line. On a terrain-following grid the boundary is the coordinate
    surface ``\\xi^3`` = const, whose normal derivative is the contravariant 3
    component ``g^{31} \\partial_1 x + g^{32} \\partial_2 x + g^{33} \\partial_3 x``.
    The two differ wherever ``g^{31}`` or ``g^{32}`` is nonzero, so
    `SetGradient(Covariant3Vector(0))` is a zero normal derivative only where
    the boundary is flat; elsewhere the value that gives one is
    ``-(g^{31} \\partial_1 x + g^{32} \\partial_2 x) / g^{33}``.

To prescribe the boundary value of `x` instead, pass a [`SetValue`](@ref):
the constructor then returns a [`DirichletOperator`](@ref) that applies
[`gradient_c2f_dirichlet`](@ref), which reproduces the `SetValue` boundary
stencil exactly and fuses into an enclosing broadcast with lazy boundary rows.
"""
struct GradientC2F{BC} <: GradientOperator
    bcs::BC
    function GradientC2F(; kwargs...)
        has_setvalue_bc(kwargs) &&
            return DirichletOperator{GradientC2F}(kwargs)
        assert_valid_bcs("GradientC2F", kwargs, SetGradient)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    GradientC2F(bcs) = GradientC2F(; bcs...)
end

return_space(::GradientC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::GradientC2F, arg) = ((-half, half),)

abstract type DivergenceOperator <: FiniteDifferenceOperator end

return_eltype(::DivergenceOperator, arg) =
    Geometry.divergence_result_type(eltype(arg))

"""
    D = DivergenceF2C(;boundaryname=boundarycondition...)
    D.(v)

Compute the vertical contribution to the divergence of a face-valued field
vector `v`, returning a center-valued scalar field, using the stencil

```math
D(v)[i] = (Jv³[i+\\tfrac{1}{2}] - Jv³[i-\\tfrac{1}{2}]) / J[i]
```

where `Jv³` is the Jacobian multiplied by the third contravariant component of
`v`.

The following boundary conditions are supported:

  - By default (no boundary condition), the value of `v` at the boundary face
    is used.
  - [`SetValue(v₀)`](@ref): calculate the divergence assuming the value at the
    boundary is `v₀`. For the left boundary, this becomes:

```math
D(v)[1] = (Jv³[1+\\tfrac{1}{2}] - Jv³₀) / J[1]
```

  - [`SetDivergence(d₀)`](@ref SetDivergence): set the divergence at the cell
    center closest to the boundary to be `d₀`. For the left boundary, this
    becomes:

```math
D(v)[1] = d₀
```

  - [`Extrapolate()`](@ref Extrapolate), equivalently [`Outflow()`](@ref):
    set the value at the center closest to the boundary to be the same as the
    neighbouring interior value. For the left boundary, this becomes:

```math
D(v)[1] = D(v)[2]
```
"""
struct DivergenceF2C{BCS} <: DivergenceOperator
    bcs::BCS
    function DivergenceF2C(; kwargs...)
        assert_valid_bcs(
            "DivergenceF2C",
            kwargs,
            Union{SetValue, SetDivergence, Extrapolate},
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    DivergenceF2C(bcs) = DivergenceF2C(; bcs...)
end

return_space(::DivergenceF2C, arg) = Spaces.center_space(axes(arg))

stencil_interior_width(::DivergenceF2C, arg) = ((-half, half),)
# Every order of extrapolation replicates the closest interior output at a
# boundary center (see MatrixFields.op_matrix_row).
boundary_width(::DivergenceF2C, ::Extrapolate, args...) = 1

"""
    D = DivergenceC2F(;boundaryname=boundarycondition...)
    D.(v)

Compute the vertical contribution to the divergence of a center-valued field
vector `v`, returning a face-valued scalar field, using the stencil

```math
D(v)[i] = (Jv³[i+\\tfrac{1}{2}] - Jv³[i-\\tfrac{1}{2}]) / J[i]
```

where `Jv³` is the Jacobian multiplied by the third contravariant component of
`v`.

The following boundary conditions are supported:

  - [`SetDivergence(x)`](@ref): set the value of the divergence at the boundary to be `x`.
    ```math
    D(v)[\\tfrac{1}{2}] = x
    ```

To prescribe the boundary value of `v` instead, pass a [`SetValue`](@ref):
the constructor then returns a [`DirichletOperator`](@ref) that applies
[`divergence_c2f_dirichlet`](@ref), which reproduces the `SetValue` boundary
stencil exactly and fuses into an enclosing broadcast with lazy boundary rows.
"""
struct DivergenceC2F{BC} <: DivergenceOperator
    bcs::BC
    function DivergenceC2F(; kwargs...)
        has_setvalue_bc(kwargs) &&
            return DirichletOperator{DivergenceC2F}(kwargs)
        assert_valid_bcs("DivergenceC2F", kwargs, SetDivergence)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    DivergenceC2F(bcs) = DivergenceC2F(; bcs...)
end

return_space(::DivergenceC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::DivergenceC2F, arg) = ((-half, half),)

abstract type CurlFiniteDifferenceOperator <: FiniteDifferenceOperator end

return_eltype(::CurlFiniteDifferenceOperator, arg) =
    Geometry.curl_result_type(Val((3,)), eltype(arg))

"""
    C = CurlC2F(;boundaryname=boundarycondition...)
    C.(v)

Compute the vertical-derivative contribution to the curl of a center-valued
covariant vector field `v`. It acts on the horizontal covariant components of
`v` (that is, it only depends on ``v₁`` and ``v₂``), and returns a face-valued
horizontal contravariant vector field (that is, ``C(v)³ = 0``).

Specifically it approximates:

```math
\\begin{align*}
C(v)^1 &= -\\frac{1}{J} \\frac{\\partial v_2}{\\partial \\xi^3}  \\\\
C(v)^2 &= \\frac{1}{J} \\frac{\\partial v_1}{\\partial \\xi^3} \\\\
\\end{align*}
```

using the stencils

```math
\\begin{align*}
C(v)[i]^1 &= - \\frac{1}{J[i]} (v₂[i+\\tfrac{1}{2}] - v₂[i-\\tfrac{1}{2}]) \\\\
C(v)[i]^2 &= \\frac{1}{J[i]}  (v₁[i+\\tfrac{1}{2}] - v₁[i-\\tfrac{1}{2}])
\\end{align*}
```

where ``v₁`` and ``v₂`` are the 1st and 2nd covariant components of ``v``, and
``J`` is the Jacobian determinant.

The following boundary conditions are supported:

  - [`SetCurl(v⁰)`](@ref): enforce the curl operator output at the boundary to be
    the contravariant vector `v⁰`.

To prescribe the boundary value of `v` instead, pass a [`SetValue`](@ref):
the constructor then returns a [`DirichletOperator`](@ref) that applies
[`curl_c2f_dirichlet`](@ref), which reproduces the `SetValue` boundary stencil
exactly and fuses into an enclosing broadcast with lazy boundary rows.
"""
struct CurlC2F{BC} <: CurlFiniteDifferenceOperator
    bcs::BC
    function CurlC2F(; kwargs...)
        has_setvalue_bc(kwargs) && return DirichletOperator{CurlC2F}(kwargs)
        assert_valid_bcs("CurlC2F", kwargs, SetCurl)
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    CurlC2F(bcs) = CurlC2F(; bcs...)
end

return_space(::CurlC2F, arg) = Spaces.face_space(axes(arg))

stencil_interior_width(::CurlC2F, arg) = ((-half, half),)

# Dirichlet (`SetValue`) replacements for the center-to-face operators.
#
# `GradientC2F`, `DivergenceC2F`, `CurlC2F` and `UpwindBiasedProductC2F` have no
# `SetValue` boundary stencil of their own; each is exactly expressible with the
# other operators and boundary conditions (the
# "Boundary values and advection built from the primitive operators" testset
# in `test/Operators/finitedifference/unit_column.jl` pins the replacement
# expressions against the stencils they reproduce). The helpers below build
# those replacements, and requesting a `SetValue` from one of those operators'
# constructors returns a `DirichletOperator` that applies the matching helper.
# Each helper has a lazy `*_broadcasted` form, which returns the replacement
# as an unmaterialized stencil broadcast that fuses into an enclosing
# broadcast like any other operator application; the public helpers
# materialize that broadcast. The boundary rows are lazy as well: each row is a
# pointwise function of the values adjacent to the boundary face, stored as an
# unmaterialized broadcast over the full argument fields that is only read at
# the boundary index (see `imposed_boundary_value` and `boundary_adjacent_arg`),
# so applying a helper allocates nothing.

"""
    DirichletOperator{Op}(bcs)

The operator returned by the constructor of `Op` (one of [`GradientC2F`](@ref),
[`DivergenceC2F`](@ref), [`CurlC2F`](@ref) or [`UpwindBiasedProductC2F`](@ref))
when one of the requested boundary conditions is a [`SetValue`](@ref), which
those operators do not support directly. Applying it with `.` calls the
corresponding Dirichlet helper ([`gradient_c2f_dirichlet`](@ref),
[`divergence_c2f_dirichlet`](@ref), [`curl_c2f_dirichlet`](@ref) or
[`upwind_biased_product_c2f_dirichlet`](@ref)) with each `SetValue(x₀)`
unwrapped to its value `x₀` and every other boundary condition passed through
as given. The helper's result is a lazy stencil broadcast with lazy boundary
rows, so it fuses into an enclosing broadcast like a true operator application
and allocates nothing.

# Fields

  - `bcs`: `NamedTuple` of boundary values and conditions, keyed by boundary name.
"""
struct DirichletOperator{Op, BCS}
    bcs::BCS
    DirichletOperator{Op}(kwargs) where {Op} = (
        bcs = map(dirichlet_bc_value, NamedTuple(kwargs));
        new{Op, typeof(bcs)}(bcs)
    )
end

dirichlet_bc_value(bc::SetValue) = bc.val
dirichlet_bc_value(bc) = bc

dirichlet_helper_broadcasted(::DirichletOperator{GradientC2F}) =
    gradient_c2f_dirichlet_broadcasted
dirichlet_helper_broadcasted(::DirichletOperator{DivergenceC2F}) =
    divergence_c2f_dirichlet_broadcasted
dirichlet_helper_broadcasted(::DirichletOperator{CurlC2F}) =
    curl_c2f_dirichlet_broadcasted
dirichlet_helper_broadcasted(::DirichletOperator{UpwindBiasedProductC2F}) =
    upwind_biased_product_c2f_dirichlet_broadcasted

# Applying a DirichletOperator returns the helper's lazy stencil broadcast,
# which fuses into any enclosing broadcast; lazy arguments are passed through
# unmaterialized.
Base.Broadcast.broadcasted(op::DirichletOperator, args...) =
    dirichlet_helper_broadcasted(op)(args...; op.bcs...)

# Wrap a boundary value for use in a lazy boundary row: numbers and axis
# tensors broadcast as scalars (as 1-tuples rather than `Ref`s, which would
# heap-allocate on every application), while fields and lazy broadcasts (over
# the boundary level or over the whole space) broadcast as themselves.
dirichlet_value(val) = (val,)
dirichlet_value(val::MaybeLazyField) = val

# Replace the center-staggered arguments of a boundary row (the boundary the row
# belongs to is known when the row is built) with their levels adjacent to the
# boundary face (bottom face 1/2 -> center 1, top face n+1/2 -> center n), which
# are constant along each column; everything else -- face-staggered fields,
# boundary-level fields, and scalars -- is read at the boundary face index as
# given.
boundary_adjacent_arg(::Val, arg) = arg
boundary_adjacent_arg(bname::Val, arg::MaybeLazyField) =
    _boundary_adjacent_arg(bname, axes(arg), arg)
_boundary_adjacent_arg(
    ::Val{S},
    space::AllCenterFiniteDifferenceSpace,
    arg,
) where {S} = boundary_adjacent_level(
    arg,
    S === Spaces.left_boundary_name(space) ? left_idx(space) : right_idx(space),
)
boundary_adjacent_level(arg, idx) = Fields.level(arg, idx)
# A level of an expression with stencils cannot be read lazily (each of its
# values depends on neighboring levels), so such an argument's values are
# computed first. This only happens for a stencil nested in the argument of a
# Dirichlet operator, and it allocates the argument's field.
boundary_adjacent_level(arg::StencilBroadcasted, idx) =
    Fields.level(Base.Broadcast.materialize(arg), idx)
_boundary_adjacent_arg(::Val, space, arg) = arg

# The lazy broadcast holding a Dirichlet boundary row, anchored to the face
# space whose boundary it is read at. The arguments may mix center- and
# face-staggered fields (center-staggered ones are replaced by their levels
# adjacent to the boundary face), so their axes cannot be combined by
# `instantiate`; the `Broadcasted` is constructed with the face space instead.
function dirichlet_row_broadcasted(f::F, bname::Val, face_space, args...) where {F}
    row_args = map(arg -> boundary_adjacent_arg(bname, arg), args)
    style = Base.Broadcast.combine_styles(row_args...)
    return Base.Broadcast.Broadcasted(style, f, row_args, face_space)
end

# The Dirichlet boundary rows as pointwise functions of the values adjacent to
# the boundary face. `_upper`/`_lower` follow the vertical direction: at the
# bottom boundary the prescribed value is the lower argument and the first
# center level is the upper one, and at the top boundary the roles are
# reversed, so each helper builds both of its boundary rows from one function.
dirichlet_gradient_row(x_upper, x_lower) =
    Geometry.Covariant3Vector(2 * (x_upper - x_lower))
dirichlet_divergence_row(v_upper, lg_upper, v_lower, lg_lower, face_lg) =
    (
        Geometry.Jcontravariant3(v_upper, lg_upper) -
        Geometry.Jcontravariant3(v_lower, lg_lower)
    ) * 2 / face_lg.J
function dirichlet_curl_row(u_upper, u_lower, face_lg)
    Δu = u_upper - u_lower
    return Geometry.Contravariant12Vector(
        -2 * Δu.components.data.:2 / face_lg.J,
        2 * Δu.components.data.:1 / face_lg.J,
    )
end
dirichlet_upwind_row(v, x_lower, x_upper, face_lg) =
    Geometry.Contravariant3Vector(
        upwind_biased_product(
            Geometry.contravariant3(v, face_lg),
            x_lower,
            x_upper,
        ),
    )

# Split the user-provided boundary values into the values at the space's left
# (bottom) and right (top) boundaries, validating the boundary names.
# The boundary names are part of the vertical topology's type, so the check
# folds away for valid names, and the error message is built at compile time.
@inline function dirichlet_boundary_values(f, space, boundary_values)
    names =
        (Spaces.left_boundary_name(space), Spaces.right_boundary_name(space))
    unrolled_all(in(names), keys(boundary_values)) ||
        error(invalid_dirichlet_names_string(f, Val(keys(boundary_values)), Val(names)))
    (names..., get(boundary_values, names[1], nothing),
        get(boundary_values, names[2], nothing))
end
@generated invalid_dirichlet_names_string(
    f,
    ::Val{bc_names},
    ::Val{space_names},
) where {bc_names, space_names} = "$(nameof(f.instance)): every boundary value \
    must be named after a boundary of the space ($(join(space_names, ", "))); \
    got $(join(bc_names, ", "))"

"""
    gradient_c2f_dirichlet(x; <boundary_name> = x₀...)

Return the vertical gradient of the center-valued field `x` at faces, with the
value of `x` prescribed to be `x₀` at each named boundary face: the Dirichlet
form of [`GradientC2F`](@ref), equivalent to
`GradientC2F(<boundary_name> = SetValue(x₀)).(x)` and built (for `bottom` and
`top` boundaries) as

```julia
GradientC2F(
    bottom = SetGradient(Geometry.Covariant3Vector.(2 .* (Fields.level(x, 1) .- x₀))),
    top = SetGradient(Geometry.Covariant3Vector.(2 .* (x₀ .- Fields.level(x, nlevels)))),
).(
    x,
)
```

`x` must have a scalar eltype. Each boundary value may be a number, a `Field`
(on the corresponding boundary level of `x`'s space, or on a whole space, of
which only the level adjacent to the boundary is read), or an unmaterialized
lazy broadcast of such fields; a boundary value that is already a
[`VerticalBoundaryCondition`](@ref) (e.g. a `SetGradient`) is instead applied
as given, so a Dirichlet value on one boundary can be combined with an
explicit condition on the other; and a boundary without a prescribed value is
computed as by `GradientC2F` without a boundary condition there. The result is
materialized on the face space.
"""
gradient_c2f_dirichlet(x; boundary_values...) = Base.Broadcast.materialize(
    gradient_c2f_dirichlet_broadcasted(x; boundary_values...),
)

# The lazy form of `gradient_c2f_dirichlet`: the same replacement, returned as
# an unmaterialized stencil broadcast with lazy boundary rows.
function gradient_c2f_dirichlet_broadcasted(x; boundary_values...)
    space = axes(x)
    face_space = Spaces.face_space(space)
    (lname, rname, x_bot, x_top) =
        dirichlet_boundary_values(
            gradient_c2f_dirichlet,
            space,
            NamedTuple(boundary_values),
        )
    bcs = (;)
    if x_bot !== nothing
        bc = if x_bot isa VerticalBoundaryCondition
            x_bot
        else
            # G(x)[1/2] = 2 (x[1] - x₀)
            SetGradient(
                dirichlet_row_broadcasted(
                    dirichlet_gradient_row,
                    Val(lname),
                    face_space,
                    x,
                    dirichlet_value(x_bot),
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(lname,)}((bc,)))
    end
    if x_top !== nothing
        bc = if x_top isa VerticalBoundaryCondition
            x_top
        else
            # G(x)[n+1/2] = 2 (x₀ - x[n])
            SetGradient(
                dirichlet_row_broadcasted(
                    dirichlet_gradient_row,
                    Val(rname),
                    face_space,
                    dirichlet_value(x_top),
                    x,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(rname,)}((bc,)))
    end
    return Base.Broadcast.broadcasted(GradientC2F(; bcs...), x)
end

"""
    divergence_c2f_dirichlet(v; <boundary_name> = v₀...)

Return the vertical contribution to the divergence of the center-valued vector
field `v` at faces, with the value of `v` prescribed to be `v₀` at each named
boundary face: the Dirichlet form of [`DivergenceC2F`](@ref), equivalent to
`DivergenceC2F(<boundary_name> = SetValue(v₀)).(v)` and built by wrapping a
plain `DivergenceC2F` in a [`SetBoundaryOperator`](@ref) that overrides each
prescribed boundary face with the Dirichlet stencil's value,

```math
D(v)[\\tfrac{1}{2}] = (Jv³[1] - Jv³₀) \\frac{2}{J[\\tfrac{1}{2}]}
```

(and its mirror image at the top), where `Jv³₀` is computed from `v₀` and the
boundary face's local geometry.

Each boundary value may be an axis tensor such as `Geometry.WVector(0.0)` (a
number is not meaningful here), a `Field` of such values (on the corresponding
boundary level, or on a whole space, of which only the level adjacent to the
boundary is read), or an unmaterialized lazy broadcast of such fields. A
boundary value that is already a [`VerticalBoundaryCondition`](@ref) (one
accepted by `SetBoundaryOperator`, e.g. a `SetValue` or `SetDivergence` of the
operator's output) is instead imposed as given on the wrapping
`SetBoundaryOperator`, and a boundary without a prescribed value is computed
as by `DivergenceC2F` without a boundary condition there. The result is
materialized on the face space.
"""
divergence_c2f_dirichlet(v; boundary_values...) = Base.Broadcast.materialize(
    divergence_c2f_dirichlet_broadcasted(v; boundary_values...),
)

# The lazy form of `divergence_c2f_dirichlet` (see
# `gradient_c2f_dirichlet_broadcasted`).
function divergence_c2f_dirichlet_broadcasted(v; boundary_values...)
    space = axes(v)
    face_space = Spaces.face_space(space)
    (lname, rname, v_bot, v_top) =
        dirichlet_boundary_values(
            divergence_c2f_dirichlet,
            space,
            NamedTuple(boundary_values),
        )
    face_lg = Fields.local_geometry_field(face_space)
    center_lg = Fields.local_geometry_field(space)
    bcs = (;)
    if v_bot !== nothing
        bc = if v_bot isa VerticalBoundaryCondition
            v_bot
        else
            # D(v)[1/2] = (Jv³[1] - Jv³₀) 2 / J[1/2], with Jv³₀ computed from
            # the prescribed value and the boundary face's local geometry
            SetValue(
                dirichlet_row_broadcasted(
                    dirichlet_divergence_row,
                    Val(lname),
                    face_space,
                    v,
                    center_lg,
                    dirichlet_value(v_bot),
                    face_lg,
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(lname,)}((bc,)))
    end
    if v_top !== nothing
        bc = if v_top isa VerticalBoundaryCondition
            v_top
        else
            # D(v)[n+1/2] = (Jv³₀ - Jv³[n]) 2 / J[n+1/2]
            SetValue(
                dirichlet_row_broadcasted(
                    dirichlet_divergence_row,
                    Val(rname),
                    face_space,
                    dirichlet_value(v_top),
                    face_lg,
                    v,
                    center_lg,
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(rname,)}((bc,)))
    end
    return Base.Broadcast.broadcasted(
        SetBoundaryOperator(; bcs...),
        Base.Broadcast.broadcasted(DivergenceC2F(), v),
    )
end

"""
    curl_c2f_dirichlet(u; <boundary_name> = u₀...)

Return the vertical-derivative contribution to the curl of the center-valued
covariant vector field `u` at faces, with the value of `u` prescribed to be
`u₀` at each named boundary face: the Dirichlet form of [`CurlC2F`](@ref),
equivalent to `CurlC2F(<boundary_name> = SetValue(u₀)).(u)` and built by
supplying the Dirichlet stencil's boundary rows,

```math
C(u)[\\tfrac{1}{2}]^1 = -(u_2[1] - u_{2,0}) \\frac{2}{J[\\tfrac{1}{2}]}, \\quad
C(u)[\\tfrac{1}{2}]^2 = (u_1[1] - u_{1,0}) \\frac{2}{J[\\tfrac{1}{2}]}
```

(and their mirror images at the top), as [`SetCurl`](@ref) boundary
conditions.

Each boundary value must have the covariant 1 and 2 components of `eltype(u)`
(e.g. a `Geometry.Covariant12Vector`), and may be an axis tensor, a `Field` of
such values (on the corresponding boundary level, or on a whole space, of
which only the level adjacent to the boundary is read), or an unmaterialized
lazy broadcast of such fields. A boundary value that is already a
[`VerticalBoundaryCondition`](@ref) (e.g. a `SetCurl`) is instead applied as
given, and a boundary without a prescribed value is computed as by `CurlC2F`
without a boundary condition there. The result is materialized on the face
space.
"""
curl_c2f_dirichlet(u; boundary_values...) = Base.Broadcast.materialize(
    curl_c2f_dirichlet_broadcasted(u; boundary_values...),
)

# The lazy form of `curl_c2f_dirichlet` (see
# `gradient_c2f_dirichlet_broadcasted`).
function curl_c2f_dirichlet_broadcasted(u; boundary_values...)
    space = axes(u)
    face_space = Spaces.face_space(space)
    (lname, rname, u_bot, u_top) =
        dirichlet_boundary_values(curl_c2f_dirichlet, space, NamedTuple(boundary_values))
    face_lg = Fields.local_geometry_field(face_space)
    bcs = (;)
    if u_bot !== nothing
        bc = if u_bot isa VerticalBoundaryCondition
            u_bot
        else
            # C(u)[1/2] from Δu = u[1] - u₀
            SetCurl(
                dirichlet_row_broadcasted(
                    dirichlet_curl_row,
                    Val(lname),
                    face_space,
                    u,
                    dirichlet_value(u_bot),
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(lname,)}((bc,)))
    end
    if u_top !== nothing
        bc = if u_top isa VerticalBoundaryCondition
            u_top
        else
            # C(u)[n+1/2] from Δu = u₀ - u[n]
            SetCurl(
                dirichlet_row_broadcasted(
                    dirichlet_curl_row,
                    Val(rname),
                    face_space,
                    dirichlet_value(u_top),
                    u,
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(rname,)}((bc,)))
    end
    return Base.Broadcast.broadcasted(CurlC2F(; bcs...), u)
end

"""
    upwind_biased_product_c2f_dirichlet(v, x; <boundary_name> = x₀...)

Return the first-order upwind product of the face-valued vector field `v` and
the center-valued field `x`, with the value of `x` on the outside of each named
boundary prescribed to be `x₀`: the Dirichlet form of
[`UpwindBiasedProductC2F`](@ref), equivalent to
`UpwindBiasedProductC2F(<boundary_name> = SetValue(x₀)).(v, x)` and built by
wrapping a plain `UpwindBiasedProductC2F` in a [`SetBoundaryOperator`](@ref)
that overrides each prescribed boundary face with the Dirichlet stencil's
value, the upwind product of `v³` there with `x₀` on the boundary side and the
closest center value of `x` on the interior side.

Each boundary value may be a number, a `Field` (on the corresponding boundary
level of `x`'s space, or on a whole space, of which only the level adjacent to
the boundary is read), or an unmaterialized lazy broadcast of such fields; a
boundary value that is already a [`VerticalBoundaryCondition`](@ref) (one
accepted by `SetBoundaryOperator`, e.g. a `SetValue` of the flux) is instead
imposed as given on the wrapping `SetBoundaryOperator`; and a boundary without
a prescribed value is computed as by `UpwindBiasedProductC2F` without a
boundary condition there. The result is materialized on the face space.
"""
upwind_biased_product_c2f_dirichlet(v, x; boundary_values...) =
    Base.Broadcast.materialize(
        upwind_biased_product_c2f_dirichlet_broadcasted(
            v,
            x;
            boundary_values...,
        ),
    )

# The lazy form of `upwind_biased_product_c2f_dirichlet` (see
# `gradient_c2f_dirichlet_broadcasted`).
function upwind_biased_product_c2f_dirichlet_broadcasted(
    v,
    x;
    boundary_values...,
)
    center_space = axes(x)
    face_space = Spaces.face_space(center_space)
    (lname, rname, x_bot, x_top) = dirichlet_boundary_values(
        upwind_biased_product_c2f_dirichlet,
        center_space,
        NamedTuple(boundary_values),
    )
    face_lg = Fields.local_geometry_field(face_space)
    bcs = (;)
    if x_bot !== nothing
        bc = if x_bot isa VerticalBoundaryCondition
            x_bot
        else
            # U(v, x)[1/2] = upwind product of v³[1/2] with x₀ below and x[1]
            # above
            SetValue(
                dirichlet_row_broadcasted(
                    dirichlet_upwind_row,
                    Val(lname),
                    face_space,
                    v,
                    dirichlet_value(x_bot),
                    x,
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(lname,)}((bc,)))
    end
    if x_top !== nothing
        bc = if x_top isa VerticalBoundaryCondition
            x_top
        else
            # U(v, x)[n+1/2] = upwind product of v³[n+1/2] with x[n] below and
            # x₀ above
            SetValue(
                dirichlet_row_broadcasted(
                    dirichlet_upwind_row,
                    Val(rname),
                    face_space,
                    v,
                    x,
                    dirichlet_value(x_top),
                    face_lg,
                ),
            )
        end
        bcs = merge(bcs, NamedTuple{(rname,)}((bc,)))
    end
    return Base.Broadcast.broadcasted(
        SetBoundaryOperator(; bcs...),
        Base.Broadcast.broadcasted(UpwindBiasedProductC2F(), v, x),
    )
end

# Evaluating a stencil expression recurses through these functions once for each
# operator in the expression, alternating between the center and face spaces of
# a column, which the default recursion limit would widen to abstract types.
@drop_recursion_limits column_value,
stencil_value,
stencil_interior,
stencil_left_boundary,
stencil_right_boundary,
stencil_arg,
neighbor_arg,
cached_arg,
materialize_buffer,
register_values,
apply_stencil!

# Arguments of the functions that build and evaluate a stencil expression are
# often partially constant structs, but constant propagation never improves on
# the inferred result; it only re-infers every function in the expression.
@drop_constprop apply_operators,
inline_apply_operators,
noinline_apply_operators,
apply_operators!,
operator_arg,
pointwise_arg,
apply_operator,
apply_stencil,
apply_stencil!,
materialize_buffer,
column_value,
stencil_value
