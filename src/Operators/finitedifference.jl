import ..Utilities:
    PlusHalf,
    half,
    unionall_type,
    AutoBroadcaster,
    nested_broadcast_result_type,
    add_auto_broadcasters
import ..DebugOnly: allow_mismatched_spaces_unsafe
import UnrolledUtilities: unrolled_map

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
right_center_boundary_idx(space::AllFiniteDifferenceSpace) = size(
    Spaces.local_geometry_data(Spaces.space(space, Spaces.CellCenter())),
    1,
)
left_face_boundary_idx(space::AllFiniteDifferenceSpace) = half
right_face_boundary_idx(space::AllFiniteDifferenceSpace) =
    size(
        Spaces.local_geometry_data(Spaces.space(space, Spaces.CellFace())),
        1,
    ) - half


left_face_boundary_idx(arg) = left_face_boundary_idx(axes(arg))
right_face_boundary_idx(arg) = right_face_boundary_idx(axes(arg))
left_center_boundary_idx(arg) = left_center_boundary_idx(axes(arg))
right_center_boundary_idx(arg) = right_center_boundary_idx(axes(arg))

# Unlike getidx, this allows extracting the face local geometry from the center space,
# and vice versa.
Base.@propagate_inbounds function Geometry.LocalGeometry(
    space::AllFiniteDifferenceSpace,
    idx::Integer,
    hidx,
)
    v = idx
    if Topologies.isperiodic(space)
        v = mod1(v, Spaces.nlevels(space))
    end
    i, j, h = hindices(space, hidx)
    local_geom =
        Grids.local_geometry_data(Spaces.grid(space), Grids.CellCenter())
    return @inbounds local_geom[v, i, j, h]
end
Base.@propagate_inbounds function Geometry.LocalGeometry(
    space::AllFiniteDifferenceSpace,
    idx::PlusHalf,
    hidx,
)
    v = idx + half
    if Topologies.isperiodic(space)
        v = mod1(v, Spaces.nlevels(space))
    end
    i, j, h = hindices(space, hidx)
    local_geom = Grids.local_geometry_data(Spaces.grid(space), Grids.CellFace())
    return @inbounds local_geom[v, i, j, h]
end

strip_space(bc::AbstractBoundaryCondition, parent_space) =
    hasproperty(bc, :val) ?
    unionall_type(typeof(bc))(strip_space(bc.val, parent_space)) : bc

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
    extrapolate_weights(bc::Extrapolate{N}, navailable)

Return the weights of the `navailable` closest interior points in the ghost-point
extrapolation of `bc`, as a tuple of 3 integers ordered from the closest
interior point outwards (with trailing zeros when fewer than 3 points are
used). The extrapolation order is reduced to `navailable - 1` when fewer than
`N + 1` interior points are available.
"""
function extrapolate_weights(::Extrapolate{N}, navailable::Integer) where {N}
    n = min(N, navailable - 1)
    return n == 0 ? (1, 0, 0) : n == 1 ? (2, -1, 0) : (3, -3, 1)
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
# already AutoBroadcasters for tuple-valued fields, wrapped by `getidx`), so
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

# Deprecated aliases for the one-sided reconstruction boundary conditions that
# Extrapolate replaces. The aliases differ numerically from the conditions they
# replace: those conditions replaced the whole stencil with fixed one-sided
# reconstructions at the two faces nearest each boundary, while Extrapolate keeps
# the interior stencil's upwinding and only pads its ghost points. At the face one
# in from a boundary the two coincide exactly when the velocity at that face points
# toward the boundary (the downwind-biased reconstruction is then also the upwind
# choice) and differ when it points into the domain (see NEWS.md for the stencils).
# At the boundary face itself the replaced reconstructions reached one center
# beyond the boundary, so they required an enclosing operator that overrides that
# face (e.g. DivergenceF2C with SetValue); Extrapolate's ghost-point padding is
# well-defined there.
Base.@deprecate_binding FirstOrderOneSided Extrapolate{0} false
Base.@deprecate_binding ThirdOrderOneSided Extrapolate{1} false

abstract type Location end
abstract type Boundary <: Location end
abstract type BoundaryWindow <: Location end

struct Interior <: Location end
struct LeftBoundaryWindow{name} <: BoundaryWindow end
struct RightBoundaryWindow{name} <: BoundaryWindow end

"""
    FiniteDifferenceOperator

Supertype of the finite difference operators, which act along the vertical
(column) direction. Subtypes define:

  - [`return_eltype`](@ref)
  - [`return_space`](@ref)
  - [`stencil_interior_width`](@ref)
  - [`stencil_interior`](@ref)

See also [`AbstractBoundaryCondition`](@ref) for how to define the boundaries.
"""
abstract type FiniteDifferenceOperator <: AbstractOperator end

return_eltype(::FiniteDifferenceOperator, arg) = eltype(arg)

# boundary width error fallback
@noinline invalid_boundary_condition_error(op_type::Type, bc_type::Type) =
    error("Boundary `$bc_type` is not supported for operator `$op_type`")

boundary_width(
    op::FiniteDifferenceOperator,
    bc::AbstractBoundaryCondition,
    args...,
) = invalid_boundary_condition_error(typeof(op), typeof(bc))

@inline left_boundary_window(space) =
    LeftBoundaryWindow{Spaces.left_boundary_name(space)}()

@inline right_boundary_window(space) =
    RightBoundaryWindow{Spaces.right_boundary_name(space)}()

get_boundary(bcs::NamedTuple, name::Symbol) =
    hasfield(typeof(bcs), name) ? getfield(bcs, name) : NullBoundaryCondition()

get_boundary(bcs::@NamedTuple{}, name::Symbol) = NullBoundaryCondition()

get_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} = get_boundary(op.bcs, name)

get_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} = get_boundary(op.bcs, name)

strip_space(op::FiniteDifferenceOperator, parent_space) =
    unionall_type(typeof(op))(
        NamedTuple{keys(op.bcs)}(
            unrolled_tuple_map(Base.Fix2(strip_space, parent_space), values(op.bcs)),
        ),
    )

abstract type AbstractStencilStyle <: Fields.AbstractFieldStyle end

struct ColumnStencilStyle <: AbstractStencilStyle end

AbstractStencilStyle(bc, ::ClimaComms.AbstractCPUDevice) = ColumnStencilStyle

"""
    StencilBroadcasted{Style}(op, args[,axes[, work]])

Broadcast node holding a [`FiniteDifferenceOperator`](@ref) `op` applied to
`args`, analogous to a `Base.Broadcast.Broadcasted` object. Returned by
`Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)`.

# Fields

  - `op`: The operator.
  - `args`: The broadcast arguments.
  - `axes`: The result space, or `nothing` before instantiation.
  - `work`: Optional work storage, `nothing` by default.
"""
struct StencilBroadcasted{Style, Op, Args, Axes, Work} <:
       OperatorBroadcasted{Style}
    op::Op
    args::Args
    axes::Axes
    work::Work
end
StencilBroadcasted{Style}(
    op::Op,
    args::Args,
    axes::Axes = nothing,
    work::Work = nothing,
) where {Style, Op, Args, Axes, Work} =
    StencilBroadcasted{Style, Op, Args, Axes, Work}(op, args, axes, work)

Adapt.adapt_structure(to, sbc::StencilBroadcasted{Style}) where {Style} =
    StencilBroadcasted{Style}(
        Adapt.adapt(to, sbc.op),
        Adapt.adapt(to, sbc.args),
        Adapt.adapt(to, sbc.axes),
    )

function Base.Broadcast.instantiate(sbc::StencilBroadcasted)
    op = sbc.op
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(sbc.args)
    # axes: same logic as Broadcasted
    if sbc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.axes(sbc)
    else
        axes = sbc.axes
        if axes !== Base.axes(sbc)
            Base.Broadcast.check_broadcast_axes(axes, args...)
        end
    end
    Style = AbstractStencilStyle(sbc, ClimaComms.device(axes))
    return StencilBroadcasted{Style}(op, args, axes)
end
function Base.Broadcast.instantiate(
    bc::Base.Broadcast.Broadcasted{<:AbstractStencilStyle},
)
    # recursively instantiate the arguments to allocate intermediate work arrays
    args = instantiate_args(bc.args)
    # axes: same logic as Broadcasted
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(args...)
    else
        axes = bc.axes
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    Style = AbstractStencilStyle(bc, ClimaComms.device(axes))
    return Base.Broadcast.Broadcasted{Style}(bc.f, args, axes)
end

function strip_space(sbc::StencilBroadcasted{Style}, parent_space) where {Style}
    current_space = axes(sbc)
    new_space = placeholder_space(current_space, parent_space)
    return StencilBroadcasted{Style}(
        strip_space(sbc.op, current_space),
        unrolled_tuple_map(Base.Fix2(strip_space, current_space), sbc.args),
        new_space,
    )
end

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
    stencil_interior(::Op, space, idx, hidx, args...)

Return the value of the interior stencil of the operator `Op` at vertical index
`idx` and horizontal index `hidx` of `space`; `args` are the input arguments.
"""
function stencil_interior end

"""
    boundary_width(::Op, ::BC, args...)

Return the width of a boundary condition `BC` on an operator `Op`: the number
of locations at which a modified stencil is used. Either this function, or
[`left_interior_idx`](@ref) and [`right_interior_idx`](@ref), must be defined
for a specific `Op`/`BC` combination.
"""
function boundary_width end

"""
    stencil_left_boundary(op, bc, space, idx, hidx, args...)

Return the result of the stencil operator `op` at horizontal index `hidx` and
vertical index `idx` of `space` near the left boundary, with boundary condition
`bc`. For operators that cannot be evaluated without a boundary condition, a
`NullBoundaryCondition` generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this
method: their boundary rows come from `MatrixFields` instead, where a
`NullBoundaryCondition` row is filled with `NaN`s, so the boundary output is
`NaN` there as well.
"""
stencil_left_boundary(op, ::NullBoundaryCondition, space, _, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

"""
    stencil_right_boundary(op, bc, space, idx, hidx, args...)

Return the result of the stencil operator `op` at horizontal index `hidx` and
vertical index `idx` of `space` near the right boundary, with boundary condition
`bc`. For operators that cannot be evaluated without a boundary condition, a
`NullBoundaryCondition` generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this
method: their boundary rows come from `MatrixFields` instead, where a
`NullBoundaryCondition` row is filled with `NaN`s, so the boundary output is
`NaN` there as well.
"""
stencil_right_boundary(op, ::NullBoundaryCondition, space, _, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

abstract type InterpolationOperator <: FiniteDifferenceOperator end

function assert_no_bcs(op, kwargs)
    length(kwargs) == 0 && return nothing
    error("$op does not accept boundary conditions.")
end

import UnrolledUtilities as UU


function assert_valid_bcs(op, kwargs, ::Type{ValidBCs}) where {ValidBCs}
    UU.unrolled_foreach(values(values(kwargs))) do bc
        @assert bc isa ValidBCs "$op only supports boundary conditions:\n\n\t $ValidBCs.\n\n BCs given:\n\n\t $(values(values(kwargs)))"
    end
    return nothing
end

# `GradientC2F`, `DivergenceC2F`, `CurlC2F` and `UpwindBiasedProductC2F` have no
# `SetValue` boundary stencil of their own; each is exactly expressible with the
# other operators and boundary conditions. When a `SetValue` is requested, those
# constructors return a `DirichletOperator` (defined with the `*_c2f_dirichlet`
# helpers below) instead of an operator of their own type.
has_setvalue_bc(kwargs) =
    UU.unrolled_any(bc -> bc isa SetValue, values(values(kwargs)))

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

return_space(::InterpolateF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::InterpolateF2C, arg) = ((-half, half),)

boundary_width(::InterpolateF2C, ::VerticalBoundaryCondition) = 0


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

return_space(::InterpolateC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::InterpolateC2F, arg) = ((-half, half),)

boundary_width(::InterpolateC2F, ::VerticalBoundaryCondition) = 1


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

return_space(::BottomBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::BottomBiasedC2F, arg) = ((-half, -half),)

left_interior_idx(
    space::AbstractSpace,
    ::BottomBiasedC2F,
    ::VerticalBoundaryCondition,
    arg,
) = left_idx(space) + 1
right_interior_idx(
    space::AbstractSpace,
    ::BottomBiasedC2F,
    ::VerticalBoundaryCondition,
    arg,
) = right_idx(space)

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

return_space(::BottomBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::BottomBiasedF2C, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::BottomBiasedF2C,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, idx - half, hidx)
left_interior_idx(
    space::AbstractSpace,
    ::BottomBiasedF2C,
    ::VerticalBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::BottomBiasedF2C,
    ::VerticalBoundaryCondition,
    arg,
) = right_idx(space)

left_interior_idx(space::AbstractSpace, ::BottomBiasedF2C, ::SetValue, arg) =
    left_idx(space) + 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::BottomBiasedF2C,
    bc::SetValue,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    getidx(space, bc.val, idx, hidx)
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

return_space(::TopBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::TopBiasedC2F, arg) = ((half, half),)

left_interior_idx(
    space::AbstractSpace,
    ::TopBiasedC2F,
    ::VerticalBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::TopBiasedC2F,
    ::VerticalBoundaryCondition,
    arg,
) = right_idx(space) - 1

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

return_space(::TopBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::TopBiasedF2C, arg) = ((half, half),)

left_interior_idx(
    space::AbstractSpace,
    ::TopBiasedF2C,
    ::VerticalBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::TopBiasedF2C,
    ::VerticalBoundaryCondition,
    arg,
) = right_idx(space)

right_interior_idx(space::AbstractSpace, ::TopBiasedF2C, ::SetValue, arg) =
    right_idx(space) - 1

abstract type WeightedInterpolationOperator <: InterpolationOperator end
# TODO: this is not in general correct and the return type
# should be based on the component operator types (/, *) but we don't have a good way
# of creating ex. one(field_type) for complex fields for inference
return_eltype(::WeightedInterpolationOperator, weights, arg) = eltype(arg)

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

return_space(
    ::WeightedInterpolateF2C,
    weight_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllFaceFiniteDifferenceSpace,
) = Spaces.space(arg_space, Spaces.CellCenter())

stencil_interior_width(::WeightedInterpolateF2C, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateF2C,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(space, weight, idx + half, hidx)
    w⁻ = getidx(space, weight, idx - half, hidx)
    a⁺ = getidx(space, arg, idx + half, hidx)
    a⁻ = getidx(space, arg, idx - half, hidx)
    (w⁺ * a⁺ + w⁻ * a⁻) / (w⁺ + w⁻)
end

boundary_width(::WeightedInterpolateF2C, ::VerticalBoundaryCondition) = 0

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

return_space(
    ::WeightedInterpolateC2F,
    weight_space::AllCenterFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = Spaces.space(arg_space, Spaces.CellFace())

stencil_interior_width(::WeightedInterpolateC2F, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateC2F,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(space, weight, idx + half, hidx)
    w⁻ = getidx(space, weight, idx - half, hidx)
    a⁺ = getidx(space, arg, idx + half, hidx)
    a⁻ = getidx(space, arg, idx - half, hidx)
    (w⁺ * a⁺ + w⁻ * a⁻) / (w⁺ + w⁻)
end

boundary_width(::WeightedInterpolateC2F, ::VerticalBoundaryCondition) = 1

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
    has_linear_interior(op) &&
    UU.unrolled_all(is_linear_reconstruction, values(op.bcs))
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
    UU.unrolled_all(name -> name in names, keys(op.bcs)) ||
        invalid_advection_bc_names_error(typeof(op), keys(op.bcs), names)
    return nothing
end
@noinline invalid_advection_bc_names_error(op_type, bc_names, space_names) =
    error(
        "Every boundary condition of $(op_type.name.name) must be named \
         after a boundary of the space ($(join(space_names, ", "))); \
         got ($(join(bc_names, ", ")))",
    )
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
        Geometry.mul_return_type(
            velocity_component_type(eltype(V)),
            advected_component_type(op, add_auto_broadcasters(eltype(arg))),
        ),
    )
end
function return_space(
    op::AdvectionOperator,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
    extra_param_spaces...,
)
    assert_valid_advection_bc_names(op, velocity_space)
    return velocity_space
end
advection_velocity_width(::AdvectionOperator) = Val(:current)
velocity_stencil_width(::Val{:current}) = (0, 0)
velocity_stencil_width(::Val{:neighboring}) = (-1, 1)
# All faces are computed with the interior stencil (which applies the
# ghost-point extrapolations itself), so no boundary window is needed. The
# operators that are rewritten as matrix multiplies override this: their
# matrices need explicit boundary rows.
boundary_width(::AdvectionOperator, ::VerticalBoundaryCondition) = 0

# Never reached at runtime for operators whose boundary_width is 0 (no face is
# treated as a boundary window), but getidx's boundary branch still needs
# statically resolvable methods when the operator carries boundary conditions;
# evaluating the interior stencil keeps the semantics identical either way.
# (Operators rewritten as matrix multiplies never evaluate these raw-operator
# methods at all: their boundary rows come through FDOperatorMatrix.)
Base.@propagate_inbounds stencil_left_boundary(
    op::AdvectionOperator,
    bc,
    space,
    idx,
    hidx,
    args...,
) = stencil_interior(op, space, idx, hidx, args...)
Base.@propagate_inbounds stencil_right_boundary(
    op::AdvectionOperator,
    bc,
    space,
    idx,
    hidx,
    args...,
) = stencil_interior(op, space, idx, hidx, args...)
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

Base.@propagate_inbounds function advection_velocities(
    ::Val{:current},
    space,
    idx,
    hidx,
    velocity,
)
    v = Geometry.contravariant3(
        getidx(space, velocity, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return (v,)
end

Base.@propagate_inbounds function advection_velocities(
    ::Val{:neighboring},
    space,
    idx,
    hidx,
    velocity,
)
    idx⁻, idx⁺ = if Topologies.isperiodic(space)
        (idx - 1, idx + 1) # getidx and LocalGeometry wrap periodic indices
    else
        lf = left_face_boundary_idx(space)
        rf = right_face_boundary_idx(space)
        (max(idx - 1, lf), min(idx + 1, rf))
    end
    v⁻ = Geometry.contravariant3(
        getidx(space, velocity, idx⁻, hidx),
        Geometry.LocalGeometry(space, idx⁻, hidx),
    )
    v = Geometry.contravariant3(
        getidx(space, velocity, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    v⁺ = Geometry.contravariant3(
        getidx(space, velocity, idx⁺, hidx),
        Geometry.LocalGeometry(space, idx⁺, hidx),
    )
    return (v⁻, v, v⁺)
end

"""
    advection_ghost_values(op, space, dist_left, dist_right, a⁻⁻, a⁻, a⁺, a⁺⁺)

Replace the values of the out-of-range centers of an [`AdvectionOperator`](@ref)
stencil at the face `dist_left` faces in from the left boundary and
`dist_right` faces in from the right one with the extrapolation of `op`'s
boundary condition for that boundary from the in-range interior points of the
stencil: at the boundary face itself, both out-of-range centers share the
extrapolation from the 2 in-range points, and at the face one in from the
boundary the single out-of-range center is extrapolated from the 3 in-range
points (see [`AdvectionOperator`](@ref)). The two boundaries are handled with
independent branches: on a 2-center column, the middle face is one in from
both boundaries, so both of its out-of-range centers need their ghost-point
extrapolations, each from the only 2 in-range points. Every other face is
unaffected, and keeps the closest-value padding of the caller's index
clamping.

The stencil values are passed in already clamped, so this is shared by the
pointwise `stencil_interior` method below (the CPU and lazy-GPU path) and the
eager GPU kernel's `advection_gather`, keeping the boundary semantics of the
two paths identical by construction.
"""
@inline function advection_ghost_values(
    op::AdvectionOperator,
    space,
    dist_left,
    dist_right,
    a⁻⁻,
    a⁻,
    a⁺,
    a⁺⁺,
)
    if dist_left == 0
        bc = get_boundary(op, left_boundary_window(space))
        a⁻⁻ = a⁻ = bc(a⁺, a⁺⁺)
    elseif dist_left == 1
        bc = get_boundary(op, left_boundary_window(space))
        a⁻⁻ = dist_right == 1 ? bc(a⁻, a⁺) : bc(a⁻, a⁺, a⁺⁺)
    end
    if dist_right == 0
        bc = get_boundary(op, right_boundary_window(space))
        a⁺⁺ = a⁺ = bc(a⁻, a⁻⁻)
    elseif dist_right == 1
        bc = get_boundary(op, right_boundary_window(space))
        a⁺⁺ = dist_left == 1 ? bc(a⁺, a⁻) : bc(a⁺, a⁻, a⁻⁻)
    end
    return (a⁻⁻, a⁻, a⁺, a⁺⁺)
end

# we treat all faces like interior faces: out-of-range stencil indices are
# clamped to the domain (indices wrap on periodic domains instead), and the
# clamped values are then replaced by the boundary condition's ghost-point
# extrapolation from the in-range interior points of the stencil (a no-op
# with the default Extrapolate{0}, which matches the clamping). On
# terrain-following grids, the extrapolation is along the third coordinate
# line, not along the wall normal; see the note on AdvectionOperator.
Base.@propagate_inbounds function stencil_interior(
    op::AdvectionOperator,
    space,
    idx,
    hidx,
    velocity,
    arg,
    extra_params...,
)
    a⁻⁻, a⁻, a⁺, a⁺⁺ = if Topologies.isperiodic(space)
        (
            getidx(space, arg, idx - half - 1, hidx),
            getidx(space, arg, idx - half, hidx),
            getidx(space, arg, idx + half, hidx),
            getidx(space, arg, idx + half + 1, hidx),
        )
    else
        lc = left_center_boundary_idx(space)
        rc = right_center_boundary_idx(space)
        a⁻⁻ = getidx(space, arg, clamp(idx - half - 1, lc, rc), hidx)
        a⁻ = getidx(space, arg, clamp(idx - half, lc, rc), hidx)
        a⁺ = getidx(space, arg, clamp(idx + half, lc, rc), hidx)
        a⁺⁺ = getidx(space, arg, clamp(idx + half + 1, lc, rc), hidx)
        lf = left_face_boundary_idx(space)
        rf = right_face_boundary_idx(space)
        advection_ghost_values(op, space, idx - lf, rf - idx, a⁻⁻, a⁻, a⁺, a⁺⁺)
    end
    vs = advection_velocities(
        advection_velocity_width(op),
        space,
        idx,
        hidx,
        velocity,
    )
    params = map(param -> getidx(space, param, idx, hidx), extra_params)
    return Geometry.Contravariant3Vector(
        op(vs..., a⁻⁻, a⁻, a⁺, a⁺⁺, params...),
    )
end
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

# Boundary faces are computed with the interior stencil, padding ghost points
# with the Extrapolate boundary condition's extrapolation from the in-range
# interior points (see AdvectionOperator). Unlike a pointwise-evaluated
# AdvectionOperator, this operator is evaluated through its operator matrix,
# whose multiply clips out-of-range band entries instead of extrapolating
# their values, so the faces whose interior row reaches a ghost point need
# explicit boundary rows that fold the ghost coefficients into the in-range
# interior columns; see MatrixFields/operator_matrices.jl.
boundary_width(::UpwindBiasedProductC2F, ::VerticalBoundaryCondition) = 1

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

abstract type LimiterConstraint end
struct AlgebraicMean <: LimiterConstraint end
struct PositiveDefinite <: LimiterConstraint end
struct MonotoneHarmonic <: LimiterConstraint end
struct MonotoneLocalExtrema <: LimiterConstraint end


strip_space(op::LinVanLeerC2F, parent_space) = LinVanLeerC2F(
    NamedTuple{keys(op.bcs)}(
        unrolled_tuple_map(Base.Fix2(strip_space, parent_space), values(op.bcs)),
    ),
    op.constraint,
)

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::MonotoneLocalExtrema)
    Δ𝜙_avg = ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2
    min𝜙 = min(a⁻, a⁰, a⁺)
    max𝜙 = max(a⁻, a⁰, a⁺)
    𝛼 = min(abs(Δ𝜙_avg), 2 * (a⁰ - min𝜙), 2 * (max𝜙 - a⁰))
    Δ𝛼 = sign(Δ𝜙_avg) * 𝛼 * (1 - sign(v) * v * dt)
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

posdiff(x, y) = ifelse(x - y ≥ 0, x - y, zero(x))

function compute_Δ𝛼_linvanleer(a⁻, a⁰, a⁺, v, dt, ::PositiveDefinite)
    Δ𝜙_avg = ((a⁰ - a⁻) + (a⁺ - a⁰)) / 2
    min𝜙 = min(a⁻, a⁰, a⁺)
    max𝜙 = max(a⁻, a⁰, a⁺)
    return sign(Δ𝜙_avg) *
           min(abs(Δ𝜙_avg), 2 * posdiff(a⁺, min𝜙), 2 * posdiff(max𝜙, a⁺)) *
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


# As for UpwindBiasedProductC2F, the boundary rows implement the interior
# stencil with extrapolated ghost points; see
# MatrixFields/operator_matrices.jl. The interior stencil reaches one center
# beyond its face on each side, so the two faces nearest each boundary need
# boundary rows.
boundary_width(::Upwind3rdOrderBiasedProductC2F, ::VerticalBoundaryCondition) =
    2

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

strip_space(op::TVDLimitedFluxC2F, parent_space) = TVDLimitedFluxC2F(
    NamedTuple{keys(op.bcs)}(
        unrolled_tuple_map(Base.Fix2(strip_space, parent_space), values(op.bcs)),
    ),
    op.method,
)

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

return_space(::SetBoundaryOperator, space) = space

stencil_interior_width(::SetBoundaryOperator, arg) = ((0, 0),)
Base.@propagate_inbounds stencil_interior(
    ::SetBoundaryOperator,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, idx, hidx)

# A side with no boundary condition has no boundary row: the operator is the
# identity there. Returning 0 (rather than gating on NullBoundaryCondition in the
# generic windowing code) keeps one-sided SetBoundaryOperators out of the boundary
# window entirely, so they never reach the NaN-producing NullBoundaryCondition
# stencil methods.
boundary_width(::SetBoundaryOperator, ::VerticalBoundaryCondition) = 0
boundary_width(
    ::SetBoundaryOperator,
    ::Union{SetValue, SetGradient, SetCurl, SetDivergence},
) = 1
# The value a `SetBoundaryOperator` imposes at a boundary. `SetGradient` and `SetCurl`
# hold values in the axis the operator they were taken from writes its output in, so they
# are projected onto that axis; `SetValue` and `SetDivergence` are imposed as given.
# The value is read at the boundary index, so it may be a constant, a field on
# the boundary level, or a field or lazy broadcast over the whole space. A
# full-space value on the space's own staggering is read at the boundary index
# itself; one on the opposite staggering must be wrapped in
# `BoundaryAdjacentCenter` (as the Dirichlet helpers' boundary rows do), which
# reads it at the center level adjacent to the boundary face.
# A lazy value must be pointwise (fields combined with pointwise functions): a
# stencil operator inside it would be evaluated at the boundary index of the
# wrong staggering, so its boundary handling breaks down (an operator without
# boundary conditions yields its `NullBoundaryCondition` `NaN`s there).
Base.@propagate_inbounds imposed_boundary_value(
    bc::Union{SetValue, SetDivergence},
    space,
    idx,
    hidx,
) = getidx(space, bc.val, idx, hidx)
Base.@propagate_inbounds imposed_boundary_value(
    bc::SetGradient,
    space,
    idx,
    hidx,
) = Geometry.project(
    Geometry.Covariant3Axis(),
    getidx(space, bc.val, idx, hidx),
    Geometry.LocalGeometry(space, idx, hidx),
)
# Project onto Contravariant12, not Contravariant123: CurlC2F's operator
# matrix produces Contravariant12Vector entries (see `op_matrix_row_type` for
# CurlFiniteDifferenceOperator), so a wider boundary value would make getidx
# return a Union of the two vector types across the column.
Base.@propagate_inbounds imposed_boundary_value(bc::SetCurl, space, idx, hidx) =
    Geometry.project(
        Geometry.Contravariant12Axis(),
        getidx(space, bc.val, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )

Base.@propagate_inbounds function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::Union{SetValue, SetGradient, SetCurl, SetDivergence},
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_idx(space)
    return imposed_boundary_value(bc, space, idx, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::Union{SetValue, SetGradient, SetCurl, SetDivergence},
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_idx(space)
    return imposed_boundary_value(bc, space, idx, hidx)
end


abstract type GradientOperator <: FiniteDifferenceOperator end
# TODO: make the axis the operator works over part of the operator type.
# similar to the spectral operators, hardcoded to vertical only `(3,)` for now
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

return_space(::GradientF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::GradientF2C, arg) = ((-half, half),)

boundary_width(::GradientF2C, ::VerticalBoundaryCondition) = 0

boundary_width(::GradientF2C, ::SetValue) = 1
boundary_width(::GradientF2C, ::SetGradient) = 1
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

return_space(::GradientC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::GradientC2F, arg) = ((-half, half),)
boundary_width(::GradientC2F, ::VerticalBoundaryCondition) = 1

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

return_space(::DivergenceF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::DivergenceF2C, arg) = ((-half, half),)
boundary_width(::DivergenceF2C, ::VerticalBoundaryCondition) = 0
boundary_width(::DivergenceF2C, ::SetValue) = 1
boundary_width(::DivergenceF2C, ::SetDivergence) = 1
# Without this, `left_interior_idx`/`right_interior_idx` place the boundary
# centers in the interior window, so `op_matrix_first_row`/`op_matrix_last_row`
# for `Extrapolate` are never reached and the condition is silently ignored.
boundary_width(::DivergenceF2C, ::Extrapolate) = 1

# Extend `adapt_structure` for all boundary conditions containing a `val` field.
function Adapt.adapt_structure(to, bc::AbstractBoundaryCondition)
    if hasfield(typeof(bc), :val)
        return unionall_type(typeof(bc))(Adapt.adapt_structure(to, bc.val))
    else
        return bc
    end
end

# Extend `adapt_structure` for all operator types with boundary conditions.
Adapt.adapt_structure(to, op::FiniteDifferenceOperator) =
    hasfield(typeof(op), :bcs) ? adapt_fd_operator(to, op, op.bcs) : op

@inline adapt_fd_operator(to, op::LinVanLeerC2F, bcs) =
    LinVanLeerC2F(adapt_bcs(to, bcs), Adapt.adapt_structure(to, op.constraint))

@inline adapt_fd_operator(to, op::TVDLimitedFluxC2F, bcs) =
    TVDLimitedFluxC2F(adapt_bcs(to, bcs), Adapt.adapt_structure(to, op.method))

@inline adapt_fd_operator(to, op, bcs) =
    unionall_type(typeof(op))(; adapt_bcs(to, bcs)...)

@inline adapt_bcs(to, bcs) = NamedTuple{keys(bcs)}(
    unrolled_tuple_map(bc -> Adapt.adapt_structure(to, bc), values(bcs)),
)

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

return_space(::DivergenceC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::DivergenceC2F, arg) = ((-half, half),)

boundary_width(::DivergenceC2F, ::VerticalBoundaryCondition) = 1


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

return_space(::CurlC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())


stencil_interior_width(::CurlC2F, arg) = ((-half, half),)

boundary_width(::CurlC2F, ::VerticalBoundaryCondition) = 1


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
# the boundary index (see `imposed_boundary_value` and
# `BoundaryAdjacentCenter`), so applying a helper allocates nothing.

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
dirichlet_value(val::Union{Fields.Field, Base.AbstractBroadcasted}) = val

# Wraps a center-staggered argument of a lazy Dirichlet boundary row (`S` is
# the name of the boundary the row belongs to, as validated by
# `dirichlet_boundary_values`). The row is read at a boundary face index, but
# its center-staggered arguments are only meaningful at the center level
# adjacent to that face, so `getidx` on the wrapper translates the face index
# (bottom face 1/2 -> center 1, top face n+1/2 -> center n) before recursing
# into the wrapped argument.
struct BoundaryAdjacentCenter{S, A}
    arg::A
end
BoundaryAdjacentCenter{S}(arg::A) where {S, A} =
    BoundaryAdjacentCenter{S, A}(arg)

Adapt.adapt_structure(to, w::BoundaryAdjacentCenter{S}) where {S} =
    BoundaryAdjacentCenter{S}(Adapt.adapt(to, w.arg))

strip_space(w::BoundaryAdjacentCenter{S}, parent_space) where {S} =
    BoundaryAdjacentCenter{S}(strip_space(w.arg, parent_space))

# The wrapper broadcasts like its wrapped argument, so `combine_styles` over a
# boundary row's arguments still resolves to a field style.
Base.Broadcast.BroadcastStyle(
    ::Type{BoundaryAdjacentCenter{S, A}},
) where {S, A} = Base.Broadcast.BroadcastStyle(A)

Base.@propagate_inbounds function getidx(
    parent_space,
    w::BoundaryAdjacentCenter{S},
    idx::PlusHalf,
    hidx,
) where {S}
    # At the left (bottom) boundary the adjacent center level is above the
    # face, and at the right (top) boundary it is below. The boundary names
    # are part of the vertical topology's type, so the comparison
    # constant-folds.
    shift = S === Spaces.left_boundary_name(parent_space) ? half : -half
    return getidx(parent_space, w.arg, idx + shift, hidx)
end
# Keep a stray wrapper read at a non-face index from reaching the generic
# scalar `getidx` fallback, which would silently return the wrapper itself.
@inline getidx(parent_space, w::BoundaryAdjacentCenter, idx, hidx) =
    error("a BoundaryAdjacentCenter can only be read at a boundary face index")

# Wrap the center-staggered arguments of a boundary row (the boundary the row
# belongs to is known when the row is built); everything else -- face-staggered
# fields, boundary-level fields, and scalars -- is read at the boundary face
# index as given.
boundary_adjacent_arg(::Val, arg) = arg
boundary_adjacent_arg(
    bname::Val,
    arg::Union{Fields.Field, Base.AbstractBroadcasted},
) = _boundary_adjacent_arg(bname, axes(arg), arg)
_boundary_adjacent_arg(
    ::Val{S},
    ::AllCenterFiniteDifferenceSpace,
    arg,
) where {S} = BoundaryAdjacentCenter{S}(arg)
_boundary_adjacent_arg(::Val, space, arg) = arg

# The lazy broadcast holding a Dirichlet boundary row, anchored to the face
# space whose boundary it is read at. The arguments may mix center- and
# face-staggered fields (center-staggered ones are wrapped in
# `BoundaryAdjacentCenter`, which reads them at the level adjacent to the
# boundary face), so their axes cannot be combined by `instantiate`; the
# `Broadcasted` is constructed with the face space as its axes instead.
function dirichlet_row_broadcasted(
    f::F,
    bname::Val,
    face_space,
    args...,
) where {F}
    row_args = map(arg -> boundary_adjacent_arg(bname, arg), args)
    return Base.Broadcast.Broadcasted{
        typeof(Base.Broadcast.combine_styles(row_args...)),
    }(
        f,
        row_args,
        face_space,
    )
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
function dirichlet_boundary_values(fname, space, boundary_values)
    names =
        (Spaces.left_boundary_name(space), Spaces.right_boundary_name(space))
    foreach(keys(boundary_values)) do name
        name in names || error(
            "$fname: every boundary value must be named after a boundary of \
             the space ($(join(names, ", "))); got $name",
        )
    end
    (names..., get(boundary_values, names[1], nothing),
        get(boundary_values, names[2], nothing))
end

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
lazy broadcast of such fields; a boundary value that is already an
[`AbstractBoundaryCondition`](@ref) (e.g. a `SetGradient`) is instead applied
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
    fname = "gradient_c2f_dirichlet"
    (lname, rname, x_bot, x_top) =
        dirichlet_boundary_values(fname, space, NamedTuple(boundary_values))
    bcs = (;)
    if x_bot !== nothing
        bc = if x_bot isa AbstractBoundaryCondition
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
        bc = if x_top isa AbstractBoundaryCondition
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
boundary value that is already an [`AbstractBoundaryCondition`](@ref) (one
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
    fname = "divergence_c2f_dirichlet"
    (lname, rname, v_bot, v_top) =
        dirichlet_boundary_values(fname, space, NamedTuple(boundary_values))
    face_lg = Fields.local_geometry_field(face_space)
    center_lg = Fields.local_geometry_field(space)
    bcs = (;)
    if v_bot !== nothing
        bc = if v_bot isa AbstractBoundaryCondition
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
        bc = if v_top isa AbstractBoundaryCondition
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
lazy broadcast of such fields. A boundary value that is already an
[`AbstractBoundaryCondition`](@ref) (e.g. a `SetCurl`) is instead applied as
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
    fname = "curl_c2f_dirichlet"
    (lname, rname, u_bot, u_top) =
        dirichlet_boundary_values(fname, space, NamedTuple(boundary_values))
    face_lg = Fields.local_geometry_field(face_space)
    bcs = (;)
    if u_bot !== nothing
        bc = if u_bot isa AbstractBoundaryCondition
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
        bc = if u_top isa AbstractBoundaryCondition
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
boundary value that is already an [`AbstractBoundaryCondition`](@ref) (one
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
    fname = "upwind_biased_product_c2f_dirichlet"
    (lname, rname, x_bot, x_top) = dirichlet_boundary_values(
        fname,
        center_space,
        NamedTuple(boundary_values),
    )
    face_lg = Fields.local_geometry_field(face_space)
    bcs = (;)
    if x_bot !== nothing
        bc = if x_bot isa AbstractBoundaryCondition
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
        bc = if x_top isa AbstractBoundaryCondition
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


# code for figuring out boundary widths
# TODO: should move this to `instantiate` and store this in the StencilBroadcasted object?

_stencil_interior_width(bc::StencilBroadcasted) =
    stencil_interior_width(bc.op, bc.args...)

"""
    left_interior_idx(space, op::FiniteDifferenceOperator, bc::VerticalBoundaryCondition, args...)

Return the index of the left-most interior point of the operator `op` with
boundary `bc` when used with arguments `args...`. By default, this is

```julia
left_idx(space) + boundary_width(op, bc)
```

but it can be overridden for specific stencil types (e.g. if the stencil is
asymmetric).
"""
@inline function left_interior_idx(
    space::AbstractSpace,
    op::FiniteDifferenceOperator,
    bc::VerticalBoundaryCondition,
    args...,
)
    left_idx(space) + boundary_width(op, bc)
end

"""
    right_interior_idx(space, op::FiniteDifferenceOperator, bc::VerticalBoundaryCondition, args...)

Return the index of the right-most interior point of the operator `op` with
boundary `bc` when used with arguments `args...`. By default, this is

```julia
right_idx(space) - boundary_width(op, bc)
```

but it can be overridden for specific stencil types (e.g. if the stencil is
asymmetric).
"""
@inline function right_interior_idx(
    space::AbstractSpace,
    op::FiniteDifferenceOperator,
    bc::VerticalBoundaryCondition,
    args...,
)
    right_idx(space) - boundary_width(op, bc)
end


@inline _left_interior_window_idx_args(args::Tuple, space, loc) =
    unrolled_tuple_map(args) do arg
        left_interior_window_idx(arg, space, loc)
    end

"""
    left_interior_window_idx(arg, space, loc)

Compute the index of the leftmost point of `arg` (a stencil broadcast, field,
or scalar) on `space` that uses only the interior stencil; `loc` is the left
boundary window.
"""
@inline function left_interior_window_idx(
    bc::StencilBroadcasted,
    parent_space,
    loc::LeftBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    widths = _stencil_interior_width(bc)
    args_idx = _left_interior_window_idx_args(bc.args, space, loc)
    args_idx_widths = map((arg, width) -> arg - width[1], args_idx, widths)
    return max(
        max(args_idx_widths...),
        left_interior_idx(space, bc.op, get_boundary(bc.op, loc), bc.args...),
    )
end
@inline function left_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{<:AbstractStencilStyle},
    parent_space,
    loc::LeftBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    arg_idxs = _left_interior_window_idx_args(bc.args, space, loc)
    maximum(arg_idxs)
end
@inline function left_interior_window_idx(
    field::Union{
        Field,
        Base.Broadcast.Broadcasted{<:Fields.AbstractFieldStyle},
    },
    parent_space,
    loc::LeftBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    left_idx(space)
end
@inline function left_interior_window_idx(_, space, loc::LeftBoundaryWindow)
    left_idx(space)
end

@inline _right_interior_window_idx_args(args::Tuple, space, loc) =
    unrolled_tuple_map(args) do arg
        right_interior_window_idx(arg, space, loc)
    end

@inline function right_interior_window_idx(
    bc::StencilBroadcasted,
    parent_space,
    loc::RightBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    widths = _stencil_interior_width(bc)
    args_idx = _right_interior_window_idx_args(bc.args, space, loc)
    args_widths = map((arg, width) -> arg - width[2], args_idx, widths)
    return min(
        min(args_widths...),
        right_interior_idx(space, bc.op, get_boundary(bc.op, loc), bc.args...),
    )
end

@inline function right_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{<:AbstractStencilStyle},
    parent_space,
    loc::RightBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    arg_idxs = _right_interior_window_idx_args(bc.args, space, loc)
    minimum(arg_idxs)
end

@inline function right_interior_window_idx(
    field::Union{
        Field,
        Base.Broadcast.Broadcasted{<:Fields.AbstractFieldStyle},
    },
    parent_space,
    loc::RightBoundaryWindow,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    right_idx(space)
end
@inline function right_interior_window_idx(_, space, loc::RightBoundaryWindow)
    right_idx(space)
end

@inline function should_call_left_boundary(idx, space, op, args...)
    Topologies.isperiodic(space) && return false
    loc = left_boundary_window(space)
    boundary_condition = get_boundary(op, loc)
    return idx < left_interior_idx(
        space,
        op,
        boundary_condition,
        args...,
    )
end

@inline function should_call_right_boundary(idx, space, op, args...)
    Topologies.isperiodic(space) && return false
    loc = right_boundary_window(space)
    boundary_condition = get_boundary(op, loc)
    return idx > right_interior_idx(
        space,
        op,
        boundary_condition,
        args...,
    )
end

# When bounds checks are forced with check-bounds=yes, avoid inlining stencil
# nodes of a broadcast expression through @propagate_inbounds. If each stencil
# node inlines its interior and boundary subexpressions, the size of the
# @propagate_inbounds expression grows exponentially with operator depth. With a
# bounds check in every array access, LLVM can take tens of minutes to compile
# flux-corrected transport examples. The check_bounds flag is constant and
# precompilation caches are keyed on it, so each variant gets its own cache. If
# bounds checks aren't forced, @propagate_inbounds improves runtime performance.
macro maybe_propagate_inbounds(expr)
    esc(isone(Base.JLOptions().check_bounds) ? expr : :(Base.@propagate_inbounds $expr))
end

@maybe_propagate_inbounds function getidx(
    parent_space,
    bc::Union{StencilBroadcasted, Base.Broadcast.Broadcasted{<:Fields.AbstractFieldStyle}},
    idx,
    hidx,
)
    # Use Union-splitting here (x isa X) instead of dispatch
    # for improved latency.
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    if bc isa Base.Broadcast.Broadcasted
        # Manually call bc.f for small tuples (improved latency)
        (; args) = bc
        N = length(bc.args)
        if N == 1
            return bc.f(getidx(space, args[1], idx, hidx))
        elseif N == 2
            return bc.f(
                getidx(space, args[1], idx, hidx),
                getidx(space, args[2], idx, hidx),
            )
        elseif N == 3
            return bc.f(
                getidx(space, args[1], idx, hidx),
                getidx(space, args[2], idx, hidx),
                getidx(space, args[3], idx, hidx),
            )
        end
        return call_bc_f(bc.f, space, idx, hidx, args...)
    end
    op = bc.op
    # On a column too short to separate the two boundary windows
    # (`window_bounds` clamps the overlap), an index can lie in both windows at
    # once; the left (bottom) boundary condition always takes precedence. In
    # particular, when both boundary conditions prescribe the operator's output
    # at such an index (e.g. a two-sided SetDivergence on a single-level
    # DivergenceF2C), only the left one is applied.
    if should_call_left_boundary(idx, space, bc.op, bc.args...)
        stencil_left_boundary(
            op,
            get_boundary(op, left_boundary_window(space)),
            space,
            idx,
            hidx,
            bc.args...,
        )
    elseif should_call_right_boundary(idx, space, bc.op, bc.args...)
        stencil_right_boundary(
            op,
            get_boundary(op, right_boundary_window(space)),
            space,
            idx,
            hidx,
            bc.args...,
        )
    else
        stencil_interior(bc.op, space, idx, hidx, bc.args...)
    end
end

# broadcasting a ColumnStencilStyle gives the StencilBroadcasted's style
Base.Broadcast.BroadcastStyle(
    ::Type{<:StencilBroadcasted{Style}},
) where {Style} = Style()

Base.Broadcast.BroadcastStyle(
    style::AbstractStencilStyle,
    ::Fields.AbstractFieldStyle,
) = style

Base.eltype(bc::StencilBroadcasted) = return_eltype(bc.op, bc.args...)

vidx(space::AllFaceFiniteDifferenceSpace, idx::Union{Nothing, PlusHalf}) =
    isnothing(idx) ? 1 :
    Topologies.isperiodic(space) ? mod1(idx + half, Spaces.nlevels(space)) : idx + half
vidx(space::AllCenterFiniteDifferenceSpace, idx::Union{Nothing, Integer}) =
    isnothing(idx) ? 1 :
    Topologies.isperiodic(space) ? mod1(idx, Spaces.nlevels(space)) : idx
vidx(space::AbstractSpace, idx) = 1

# Fields on a column space only have data at a single horizontal index, so the
# horizontal indices from the broadcast expression do not apply to them.
@inline hindices(::Spaces.FiniteDifferenceSpace, hidx) = (1, 1, 1)
@inline hindices(space, hidx) = hidx

Base.@propagate_inbounds function getidx(parent_space, bc::Fields.Field, idx)
    field_data = Fields.field_values(bc)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    v = vidx(space, idx)
    return @inbounds field_data[v]
end
Base.@propagate_inbounds function getidx(
    parent_space,
    bc::Fields.Field,
    idx,
    hidx,
)
    field_data = Fields.field_values(bc)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    v = vidx(space, idx)
    i, j, h = hindices(space, hidx)
    return @inbounds field_data[v, i, j, h]
end

# unwap boxed scalars
@inline getidx(parent_space, scalar::Tuple{T}, idx, hidx) where {T} = scalar[1]
@inline getidx(parent_space, scalar::Ref, idx, hidx) = scalar[]
@inline getidx(parent_space, field::Fields.PointField, idx, hidx) = field[]
@inline getidx(parent_space, field::Fields.PointField, idx) = field[]
@inline getidx(
    parent_space,
    bc::BC,
    idx,
    hidx,
) where {
    BC <: Base.Broadcast.Broadcasted,
} = bc[]

# enable automatic nested broadcasting over single-valued boundary conditions
@inline getidx(parent_space, scalar, idx, hidx) = add_auto_broadcasters(scalar)

# getidx error fallbacks
@noinline inferred_getidx_error(idx_type::Type, space_type::Type) =
    error("Invalid index type `$idx_type` for field on space `$space_type`")

# Recursively unwrap getidx broadcast arguments in a way that the optimizer can reduce
# statically.
@generated function call_bc_f(f::F, space, idx, hidx, args...) where {F}
    N = length(args)
    return quote
        Base.@_propagate_inbounds_meta
        Base.Cartesian.@ncall $N f i -> getidx(space, args[i], idx, hidx)
    end
end

@drop_recursion_limits call_bc_f, getidx

# setidx! methods for copyto!
Base.@propagate_inbounds function setidx!(
    parent_space,
    field::Fields.Field,
    idx,
    hidx,
    val,
)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    v = vidx(space, idx)
    field_data = Fields.field_values(field)
    i, j, h = hidx
    @inbounds field_data[v, i, j, h] = val
    val
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    args′ = map(Base.Broadcast.broadcastable, args)
    style = Base.Broadcast.result_style(
        ColumnStencilStyle(),
        Base.Broadcast.combine_styles(args′...),
    )
    Base.Broadcast.broadcasted(style, op, args′...)
end

function Base.Broadcast.broadcasted(
    ::Style,
    op::FiniteDifferenceOperator,
    args...,
) where {Style <: AbstractStencilStyle}
    # Promote boundary conditions to float type
    # so that we can use integer-input boundary
    # condition values.
    # TODO: we should probably disallow this, as it
    # may help with latency.
    FT = Spaces.undertype(axes(StencilBroadcasted{Style}(op, args)))
    args′ =
        unrolled_tuple_map(args) do arg
            is_auto_broadcastable(eltype(arg)) ?
            Base.Broadcast.broadcasted(add_auto_broadcasters, arg) : arg
        end
    return StencilBroadcasted{Style}(promote_bcs(op, FT), args′)
end

# check that inferred output field space is equal to dest field space
@noinline inferred_stencil_spaces_error(
    dest_space_type::Type,
    result_space_type::Type,
) = error(
    "dest space `$dest_space_type` is not the same instance as the inferred broadcasted result space `$result_space_type`",
)

function Base.Broadcast.materialize!(
    ::DataLayouts.DataStyle,
    dest::Fields.Field,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractStencilStyle}
    dest_space, result_space = axes(dest), axes(bc)
    if result_space !== dest_space && !allow_mismatched_spaces_unsafe()
        # TODO: we pass the types here to avoid stack copying data
        # but this could lead to a confusing error message (same space type but
        # different instances)
        inferred_stencil_spaces_error(typeof(dest_space), typeof(result_space))
    end
    # The default Base behavior is to instantiate a Broadcasted object with the same axes
    # as the destination.
    return copyto!(
        dest,
        Base.Broadcast.instantiate(
            Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, dest_space),
        ),
    )
end

# A boundary condition holds values for one boundary point per column, so it
# passes through slicing unchanged; stated explicitly because level/slab/column
# deliberately have no generic identity fallback (see src/interface.jl).
for slice_op in (:level, :slab, :column)
    @eval $slice_op(bc::AbstractBoundaryCondition, inds...) = bc
end

Base.@propagate_inbounds column(op::FiniteDifferenceOperator, inds...) =
    unionall_type(typeof(op))(column(op.bcs, inds...))
Base.@propagate_inbounds column(sbc::StencilBroadcasted{S}, inds...) where {S} =
    StencilBroadcasted{S}(
        column(sbc.op, inds...),
        column(sbc.args, inds...),
        column(sbc.axes, inds...),
    )

#TODO: the optimizer dies with column broadcast expressions over a certain complexity
@drop_recursion_limits column

function _serial_copyto!(field_out::Field, bc, Ni::Int, Nj::Int, Nh::Int)
    space = axes(field_out)
    bounds = window_bounds(space, bc)
    bcs = bc # strip_space(bc, space)
    mask = Spaces.get_mask(axes(field_out))
    @inbounds for h in 1:Nh, j in 1:Nj, i in 1:Ni
        DataLayouts.should_compute(mask, CartesianIndex(1, i, j, h)) ||
            continue
        apply_stencil!(space, field_out, bcs, (i, j, h), bounds)
    end
    call_post_op_callback() &&
        post_op_callback(field_out, field_out, bc, Ni, Nj, Nh)
    return field_out
end

function _threaded_copyto!(field_out::Field, bc, Ni::Int, Nj::Int, Nh::Int)
    space = axes(field_out)
    bounds = window_bounds(space, bc)
    bcs = bc # strip_space(bc, space)
    mask = Spaces.get_mask(axes(field_out))
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for j in 1:Nj, i in 1:Ni
                DataLayouts.should_compute(
                    mask,
                    CartesianIndex(1, i, j, h),
                ) || continue
                apply_stencil!(space, field_out, bcs, (i, j, h), bounds)
            end
        end
    end
    call_post_op_callback() &&
        post_op_callback(field_out, field_out, bc, Ni, Nj, Nh)
    return field_out
end

function Base.copyto!(
    field_out::Field,
    bc::Union{
        StencilBroadcasted{ColumnStencilStyle},
        Broadcasted{ColumnStencilStyle},
    };
    mask = DataLayouts.NoMask(),
)
    space = axes(bc)
    local_geometry = Spaces.local_geometry_data(space)
    (_, Ni, Nj, Nh) = size(local_geometry)
    context = ClimaComms.context(axes(field_out))
    device = ClimaComms.device(context)
    if (device isa ClimaComms.CPUMultiThreaded) && Nh > 1
        return _threaded_copyto!(field_out, bc, Ni, Nj, Nh)
    end
    return _serial_copyto!(field_out, bc, Ni, Nj, Nh)
end

function window_bounds(space, bc)
    if Topologies.isperiodic(space)
        li = lw = left_idx(space)
        ri = rw = right_idx(space)
    else
        lbw = left_boundary_window(space)
        rbw = right_boundary_window(space)
        li = left_idx(space)
        lw = left_interior_window_idx(bc, space, lbw)::typeof(li)
        ri = right_idx(space)
        rw = right_interior_window_idx(bc, space, rbw)::typeof(ri)
        # On a short column the two boundary windows can overlap (e.g. a
        # 4-wide advection stencil on a 2-center column, whose middle face is
        # within a stencil width of both boundaries), crossing `lw` past `rw`.
        # Boundary handling is dispatched per index (`should_call_left_boundary`
        # takes precedence over the right), so the window split only needs to
        # cover each index exactly once: clamp the crossed bounds into an
        # empty interior window, with the overlap assigned to the left window.
        lw = min(lw, ri + 1)
        rw = max(rw, lw - 1)
    end
    @assert li <= lw <= rw + 1 && rw <= ri
    return (li, lw, rw, ri)
end

Base.@propagate_inbounds function apply_stencil!(
    space,
    field_out,
    bc,
    hidx,
    (li, lw, rw, ri) = window_bounds(space, bc),
)
    IP = Topologies.isperiodic(space)
    L = !IP ? li : lw
    R = !IP ? ri : rw
    @inbounds for idx in L:R
        val = getidx(space, bc, idx, hidx)
        setidx!(space, field_out, idx, hidx, val)
    end
    return field_out
end

"""
    fd_shmem_is_supported(bc::Base.Broadcast.AbstractBroadcasted)

Return whether the broadcasted object `bc` supports shared memory, so that the
GPU extension can dispatch to an optimized kernel.

This function and its dispatch become unnecessary once all operators support
shared memory.
"""
function fd_shmem_is_supported end

"""
    any_fd_shmem_supported(::Base.Broadcast.AbstractBroadcasted)

Return whether any operator in the broadcasted object supports finite-difference
shared memory (see [`fd_shmem_is_supported`](@ref)).
"""
function any_fd_shmem_supported end

"""
    promote_bcs(op::FiniteDifferenceOperator, FT)
    promote_bcs(bcs::NamedTuple, FT)

Promote the boundary conditions of `op` (or the boundary-condition `NamedTuple`
`bcs`) with `promote_bc`, so that integer-valued boundary conditions take the
space's float type `FT` and `getidx` is type-stable throughout the broadcast
expression. Internal.
"""
@inline function promote_bcs(
    op::FiniteDifferenceOperator,
    ::Type{FT},
) where {FT}
    if hasfield(typeof(op), :bcs)
        unionall_type(typeof(op))(promote_bcs(op.bcs, FT))
    else
        op
    end
end

@inline function promote_bcs(op::LinVanLeerC2F, ::Type{FT}) where {FT}
    if hasfield(typeof(op), :bcs)
        unionall_type(typeof(op))(promote_bcs(op.bcs, FT), op.constraint)
    else
        op
    end
end

@inline function promote_bcs(op::TVDLimitedFluxC2F, ::Type{FT}) where {FT}
    if hasfield(typeof(op), :bcs)
        unionall_type(typeof(op))(promote_bcs(op.bcs, FT), op.method)
    else
        op
    end
end

@inline promote_bcs(x::Fields.Field, ::Type{FT}) where {FT} = x
@inline promote_bcs(bcs::@NamedTuple{}, ::Type{FT}) where {FT} = NamedTuple()
@inline promote_bcs(bcs::NamedTuple{N, V}, ::Type{FT}) where {FT} where {N, V} =
    NamedTuple{N}(map(x -> promote_bc(x, FT), values(bcs)))

"""
    promote_bc(bc::AbstractBoundaryCondition, FT)

Promote the value of a `SetValue`, `SetGradient`, `SetDivergence`, or `SetCurl`
boundary condition to the space's float type `FT` when it is an `Integer` or an
axis tensor of another element type, so that `getidx` is type-stable throughout
the broadcast expression; other boundary conditions are returned unchanged.
Internal.
"""
promote_bc(bc::SetValue, FT) = bc
promote_bc(bc::SetGradient, FT) = bc
promote_bc(bc::SetDivergence, FT) = bc
promote_bc(bc::SetCurl, FT) = bc
promote_bc(bc::AbstractBoundaryCondition, FT) = bc

promote_bc(bc::SetValue{<:Integer}, ::Type{FT}) where {FT} =
    SetValue(FT(bc.val))
promote_bc(bc::SetGradient{<:Integer}, ::Type{FT}) where {FT} =
    SetGradient(FT(bc.val))
promote_bc(bc::SetDivergence{<:Integer}, ::Type{FT}) where {FT} =
    SetDivergence(FT(bc.val))
promote_bc(bc::SetCurl{<:Integer}, ::Type{FT}) where {FT} = SetCurl(FT(bc.val))

sconvert(::Type{T}, x::SArray{S}) where {T, S} = SArray{S, T}(x...)

function promote_axis_tensor(
    at::Geometry.Tensor,
    ::Type{FT},
) where {FT}
    fc = sconvert(FT, parent(at))
    return Geometry.Tensor(fc, axes(at))
end

promote_axis_tensor(at::Geometry.Tensor{<:Any, FT}, ::Type{FT}) where {FT} = at

promote_bc(bc::SetValue{<:Geometry.AbstractTensor}, ::Type{FT}) where {FT} =
    SetValue(promote_axis_tensor(bc.val, FT))
promote_bc(bc::SetGradient{<:Geometry.AbstractTensor}, ::Type{FT}) where {FT} =
    SetGradient(promote_axis_tensor(bc.val, FT))
promote_bc(bc::SetDivergence{<:Geometry.AbstractTensor}, ::Type{FT}) where {FT} =
    SetDivergence(promote_axis_tensor(bc.val, FT))
promote_bc(bc::SetCurl{<:Geometry.AbstractTensor}, ::Type{FT}) where {FT} =
    SetCurl(promote_axis_tensor(bc.val, FT))

"""
    use_fd_shmem()

Return whether finite-difference shared memory is enabled for the operators that
support it; `false` by default. Users can enable it from global scope by
redefining the method.

# Examples

```julia
Operators.use_fd_shmem() = true
```

# Notes

A ~30% slowdown was observed with shared memory enabled in ClimaCore 0.14.31
aquaplanet benchmarks.
"""
use_fd_shmem() = false
