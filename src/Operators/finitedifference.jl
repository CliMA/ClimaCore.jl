import ..Utilities: PlusHalf, half, unionall_type
import ..DebugOnly: allow_mismatched_spaces_unsafe
import UnrolledUtilities: unrolled_map

const AllFiniteDifferenceSpace =
    Union{Spaces.FiniteDifferenceSpace, Spaces.ExtrudedFiniteDifferenceSpace}
const AllFaceFiniteDifferenceSpace = Union{
    Spaces.FaceFiniteDifferenceSpace,
    Spaces.FaceExtrudedFiniteDifferenceSpace,
}
const AllCenterFiniteDifferenceSpace = Union{
    Spaces.CenterFiniteDifferenceSpace,
    Spaces.CenterExtrudedFiniteDifferenceSpace,
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

# unlike getidx, we allow extracting the face local geometry from the center space, and vice-versa
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


"""
    AbstractBoundaryCondition

An abstract type for boundary conditions for [`FiniteDifferenceOperator`](@ref)s.

Subtypes should define:
- [`boundary_width`](@ref)
- [`stencil_left_boundary`](@ref)
- [`stencil_right_boundary`](@ref)
"""
abstract type AbstractBoundaryCondition end

strip_space(bc::AbstractBoundaryCondition, parent_space) =
    hasproperty(bc, :val) ?
    unionall_type(typeof(bc))(strip_space(bc.val, parent_space)) : bc

"""
    NullBoundaryCondition()

This is used as a placeholder when no other boundary condition can be applied.

Wherever an operator needs a boundary row for this condition (that is, wherever
[`boundary_width`](@ref) is nonzero for it), the value produced there is a placeholder
rather than a meaningful result, and which placeholder depends on how the operator is
evaluated:

 - an operator that is rewritten into an operator matrix multiply gets a zero boundary
   row, so its boundary output is zero;
 - any other operator goes through [`stencil_left_boundary`](@ref) /
   [`stencil_right_boundary`](@ref), which produce `NaN`.

Where `boundary_width` is zero the interior stencil applies instead and nothing special
happens, so the same operator can give a placeholder at one boundary and an ordinary
value at the other. Rather than relying on either placeholder, give the operator a
boundary condition, or overwrite the boundary afterwards with a
[`SetBoundaryOperator`](@ref).
"""
struct NullBoundaryCondition <: AbstractBoundaryCondition end

"""
    SetValue(val)

Set the value at the boundary to be `val`. In the case of gradient operators,
this will set the input value from which the gradient is computed.
"""
struct SetValue{S} <: AbstractBoundaryCondition
    val::S
end

"""
    SetGradient(val)

Set the gradient at the boundary to be `val`. In the case of gradient operators
this will set the output value of the gradient.
"""
struct SetGradient{S} <: AbstractBoundaryCondition
    val::S
end

"""
    SetDivergence(val)

Set the divergence at the boundary to be `val`.
"""
struct SetDivergence{S} <: AbstractBoundaryCondition
    val::S
end

"""
    SetCurl(val)

Set the curl at the boundary to be `val`.
"""
struct SetCurl{S} <: AbstractBoundaryCondition
    val::S
end

"""
    Extrapolate()

Set the value at the boundary to be the same as the closest interior point.
"""
struct Extrapolate <: AbstractBoundaryCondition end

"""
    FirstOrderOneSided()

Use a first-order up/down-wind scheme to compute the value at the boundary.
"""
struct FirstOrderOneSided <: AbstractBoundaryCondition end

"""
    ThirdOrderOneSided()

Use a third-order up/down-wind scheme to compute the value at the boundary.
"""
struct ThirdOrderOneSided <: AbstractBoundaryCondition end

abstract type Location end
abstract type Boundary <: Location end
abstract type BoundaryWindow <: Location end

struct Interior <: Location end
struct LeftBoundaryWindow{name} <: BoundaryWindow end
struct RightBoundaryWindow{name} <: BoundaryWindow end

"""
    FiniteDifferenceOperator

An abstract type for finite difference operators. Instances of this should define:

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

"""
    rebuild_op(op, bcs)

Rebuilds `op` with the boundary conditions `bcs`, preserving any non-bc fields
(like `LimitedFluxC2F`'s kernel) that `unionall_type(typeof(op))(bcs)` would
drop. Every generic path that transforms an operator's boundary conditions
(`strip_space`, `column`, `adapt_fd_operator`, `promote_bcs`) rebuilds through
this hook.
"""
@inline rebuild_op(op::FiniteDifferenceOperator, bcs) =
    unionall_type(typeof(op))(bcs)

strip_space(op::FiniteDifferenceOperator, parent_space) = rebuild_op(
    op,
    NamedTuple{keys(op.bcs)}(strip_space_args(values(op.bcs), parent_space)),
)

abstract type AbstractStencilStyle <: Fields.AbstractFieldStyle end

struct ColumnStencilStyle <: AbstractStencilStyle end

AbstractStencilStyle(bc, ::ClimaComms.AbstractCPUDevice) = ColumnStencilStyle

"""
    StencilBroadcasted{Style}(op, args[,axes[, work]])

This is similar to a `Base.Broadcast.Broadcasted` object.

This is returned by `Base.Broadcast.broadcasted(op::FiniteDifferenceOperator)`.
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
        strip_space_args(sbc.args, current_space),
        new_space,
    )
end

"""
    return_eltype(::Op, fields...)

Defines the element type of the result of operator `Op`
"""
function return_eltype end


"""
    stencil_interior_width(::Op, args...)

Defines the width of the interior stencil for the operator `Op` with the given
arguments. Returns a tuple of 2-tuples: each 2-tuple should be the lower and
upper bounds of the index offsets of the stencil for each argument in the
stencil.

## Example
```
stencil(::Op, arg1, arg2) = ((-half, 1+half), (0,0))
```
implies that at index `i`, the stencil accesses `arg1` at `i-half`, `i+half` and
`i+1+half`, and `arg2` at index `i`.
"""
function stencil_interior_width end

"""
    stencil_interior(::Op, space, idx, args...)

Defines the stencil of the operator `Op` in the interior of the domain at `idx`;
`args` are the input arguments.
"""
function stencil_interior end

"""
    boundary_width(::Op, ::BC, args...)

Defines the width of a boundary condition `BC` on an operator `Op`. This is the
number of locations that are used in a modified stencil. Either this function,
or [`left_interior_idx`](@ref) and [`right_interior_idx`](@ref) should be
defined for a specific `Op`/`BC` combination.
"""
function boundary_width end

"""
    stencil_left_boundary(op, bc, idx, hidx, args...)

The result of stencil operator `op` at horizontal index `hidx` and some vertical
index `idx` near the left boundary, with boundary condition `bc`. For operators
that cannot be evaluated without a boundary condition, a `NullBoundaryCondition`
generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this method:
their boundary rows come from `MatrixFields` instead, where a `NullBoundaryCondition` row
is zero rather than `NaN`.
"""
stencil_left_boundary(op, ::NullBoundaryCondition, space, _, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

"""
    stencil_right_boundary(op, bc, idx, hidx, args...)

The result of stencil operator `op` at horizontal index `hidx` and some vertical
index `idx` near the right boundary, with boundary condition `bc`. For operators
that cannot be evaluated without a boundary condition, a `NullBoundaryCondition`
generates `NaN` values here.

Operators that are rewritten into an operator matrix multiply do not reach this method:
their boundary rows come from `MatrixFields` instead, where a `NullBoundaryCondition` row
is zero rather than `NaN`.
"""
stencil_right_boundary(op, ::NullBoundaryCondition, space, _, _, args...) =
    new(return_eltype(op, args...)) * Spaces.undertype(space)(NaN)

abstract type InterpolationOperator <: FiniteDifferenceOperator end

function assert_no_bcs(op, kwargs)
    length(kwargs) == 0 && return nothing
    error("InterpolateF2C does not accept boundary conditions.")
end

import UnrolledUtilities as UU
function assert_valid_bcs(op, kwargs, valid_bcs)
    UU.unrolled_foreach(values(values(kwargs))) do bc
        @assert UU.unrolled_any(valid_bc -> bc isa valid_bc, valid_bcs) "$op only supports boundary conditions:\n\n\t $valid_bcs.\n\n BCs given:\n\n\t $(values(values(kwargs)))\n"
    end
    return nothing
end

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

boundary_width(::InterpolateF2C, ::AbstractBoundaryCondition) = 0


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
            (SetValue, Extrapolate),
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    InterpolateC2F(bcs) = InterpolateC2F(; bcs...)
end

return_space(::InterpolateC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::InterpolateC2F, arg) = ((-half, half),)

boundary_width(::InterpolateC2F, ::AbstractBoundaryCondition) = 1


"""
    L = LeftBiasedC2F(;boundaries)
    L.(x)

Interpolate a center-value field to a face-valued field from the left.
```math
L(x)[i] = x[i-\\tfrac{1}{2}]
```

Only the left boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
L(x)[\\tfrac{1}{2}] = x_0
```
"""
struct LeftBiasedC2F{BCS} <: InterpolationOperator
    bcs::BCS
    function LeftBiasedC2F(; kwargs...)
        assert_valid_bcs("LeftBiasedC2F", kwargs, (SetValue,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    LeftBiasedC2F(bcs) = LeftBiasedC2F(; bcs...)
end

return_space(::LeftBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::LeftBiasedC2F, arg) = ((-half, -half),)

left_interior_idx(
    space::AbstractSpace,
    ::LeftBiasedC2F,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space) + 1
right_interior_idx(
    space::AbstractSpace,
    ::LeftBiasedC2F,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space)

"""
    L = LeftBiasedF2C(;boundaries)
    L.(x)

Interpolate a face-value field to a center-valued field from the left.
```math
L(x)[i+\\tfrac{1}{2}] = x[i]
```

Only the left boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
L(x)[1] = x_0
```
"""
struct LeftBiasedF2C{BCS} <: InterpolationOperator
    bcs::BCS
    function LeftBiasedF2C(; kwargs...)
        assert_valid_bcs("LeftBiasedF2C", kwargs, (SetValue,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    LeftBiasedF2C(bcs) = LeftBiasedF2C(; bcs...)
end

return_space(::LeftBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::LeftBiasedF2C, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiasedF2C,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, idx - half, hidx)
left_interior_idx(
    space::AbstractSpace,
    ::LeftBiasedF2C,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::LeftBiasedF2C,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space)

left_interior_idx(space::AbstractSpace, ::LeftBiasedF2C, ::SetValue, arg) =
    left_idx(space) + 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiasedF2C,
    bc::SetValue,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    getidx(space, bc.val, nothing, hidx)
end

"""
    R = RightBiasedC2F(;boundaries)
    R.(x)

Interpolate a center-valued field to a face-valued field from the right.
```math
R(x)[i] = x[i+\\tfrac{1}{2}]
```

Only the right boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
R(x)[n+\\tfrac{1}{2}] = x_0
```
"""
struct RightBiasedC2F{BCS} <: InterpolationOperator
    bcs::BCS
    function RightBiasedC2F(; kwargs...)
        assert_valid_bcs("RightBiasedC2F", kwargs, (SetValue,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    RightBiasedC2F(bcs) = RightBiasedC2F(; bcs...)
end

return_space(::RightBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::RightBiasedC2F, arg) = ((half, half),)

left_interior_idx(
    space::AbstractSpace,
    ::RightBiasedC2F,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::RightBiasedC2F,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space) - 1

"""
    R = RightBiasedF2C(;boundaries)
    R.(x)

Interpolate a face-valued field to a center-valued field from the right.
```math
R(x)[i] = x[i+\\tfrac{1}{2}]
```

Only the right boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
R(x)[n+\\tfrac{1}{2}] = x_0
```
"""
struct RightBiasedF2C{BCS} <: InterpolationOperator
    bcs::BCS
    function RightBiasedF2C(; kwargs...)
        assert_valid_bcs("RightBiasedF2C", kwargs, (SetValue,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    RightBiasedF2C(bcs) = RightBiasedF2C(; bcs...)
end

return_space(::RightBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::RightBiasedF2C, arg) = ((half, half),)

left_interior_idx(
    space::AbstractSpace,
    ::RightBiasedF2C,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::RightBiasedF2C,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space)

right_interior_idx(space::AbstractSpace, ::RightBiasedF2C, ::SetValue, arg) =
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
        w[i+\\tfrac{1}{2}] x[i+\\tfrac{1}{2}] +  w[i-\\tfrac{1}{2}] x[i-\\tfrac{1}{2}])
    }{
        w[i+\\tfrac{1}{2}] + w[i-\\tfrac{1}{2}]
    }
```

No boundary conditions are required (or supported)
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

boundary_width(::WeightedInterpolateF2C, ::AbstractBoundaryCondition) = 0

"""
    WI = WeightedInterpolateC2F(; boundaries)
    WI.(w, x)

Interpolate a center-valued field `x` to faces, weighted by a center-valued field
`w`, using the stencil
```math
WI(w, x)[i] = \\frac{
    w[i+\\tfrac{1}{2}] x[i+\\tfrac{1}{2}] +  w[i-\\tfrac{1}{2}] x[i-\\tfrac{1}{2}])
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
            (SetValue, Extrapolate),
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

boundary_width(::WeightedInterpolateC2F, ::AbstractBoundaryCondition) = 1
# WeightedInterpolateC2F has no stencil_left_boundary/stencil_right_boundary methods:
# every broadcast over it is rewritten as an operator matrix multiply when it is
# instantiated, so its boundary values come from the operator matrix (for Extrapolate)
# or from a SetBoundaryOperator applied to the result (for SetValue); see
# MatrixFields/operator_matrices.jl.

abstract type AdvectionOperator <: FiniteDifferenceOperator end
return_eltype(::AdvectionOperator, velocity, arg) = eltype(arg)

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

Supported boundary conditions are:
- [`Extrapolate()`](@ref): set the value of `x` to be the same as the closest
  interior point. On the left boundary, the stencil is
  ```math
  U(\\boldsymbol{v},x)[\\tfrac{1}{2}] = U(\\boldsymbol{v},x)[1 + \\tfrac{1}{2}]
  ```
"""
struct UpwindBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
    function UpwindBiasedProductC2F(; kwargs...)
        assert_valid_bcs(
            "UpwindBiasedProductC2F",
            kwargs,
            (Extrapolate,),
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    UpwindBiasedProductC2F(bcs) = UpwindBiasedProductC2F(; bcs...)
end

return_eltype(::UpwindBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::UpwindBiasedProductC2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = velocity_space

"""
    upwind_select(s, x⁻, x⁺)

The upwind dispatch shared by all advection operators: `x⁻` (the form
reconstructed from the upwind side for positive flow) when `s >= 0`, and `x⁺`
otherwise. Every advection operator is `combine(s, upwind_select(s, x⁻, x⁺))`
for some pair of one-sided forms; `combine` is usually multiplication by `s`.
"""
@inline upwind_select(s, x⁻, x⁺) = s >= 0 ? x⁻ : x⁺

"""
    upwind_biased_product(v, a⁻, a⁺)
    upwind_biased_product(v, av, a⁻, a⁺)

`v * upwind_select(v, a⁻, a⁺)`, written in a form that is linear in `a⁻` and
`a⁺` so that it can also blend the matrix rows of the one-sided reconstructions
(see `MatrixFields.operator_matrices.jl`). The 4-argument form takes the
precomputed `av = abs(v)`, for callers whose `v` does not support `abs` (like a
`Contravariant3Vector` matrix entry).
"""
upwind_biased_product(v, a⁻, a⁺) = upwind_biased_product(v, abs(v), a⁻, a⁺)
@inline upwind_biased_product(v, av, a⁻, a⁺) =
    ((v + av) * a⁻ + (v - av) * a⁺) / 2

"""
    upwind_symmetric_difference(sgn, a⁻, a, a⁺)

The upwind difference `a - a⁻` when `sgn == 1`, `a⁺ - a` when `sgn == -1`, and
the centered difference `(a⁺ - a⁻) / 2` when `sgn == 0`.
"""
@inline upwind_symmetric_difference(sgn, a⁻, a, a⁺) =
    (1 - sgn) / 2 * a⁺ + sgn * a - (1 + sgn) / 2 * a⁻

"""
    left_biased_3rd(a₁, a₂, a₃)
    right_biased_3rd(a₁, a₂, a₃)

The 3rd-order biased reconstructions of the value at a face from three
consecutive center values: `left_biased_3rd` reconstructs at the face between
`a₂` and `a₃` (from mostly-left data), and `right_biased_3rd` at the face
between `a₁` and `a₂` (from mostly-right data). These are the interior stencils
of the removed `LeftBiased3rdOrderC2F`/`RightBiased3rdOrderC2F` operators;
their matrix-row forms are in `MatrixFields`
(`left/right_biased_3rd_interior_row`).
"""
@inline left_biased_3rd(a₁, a₂, a₃) = (-2a₁ + 10a₂ + 4a₃) / 12
@inline right_biased_3rd(a₁, a₂, a₃) = (4a₁ + 10a₂ - 2a₃) / 12

stencil_interior_width(::UpwindBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half, half))

boundary_width(::UpwindBiasedProductC2F, ::AbstractBoundaryCondition) = 1

"""
    LimitedFluxC2F(bcs::NamedTuple, kernel)

The single primitive behind the nonlinear (flux-limited) advection operators:
[`LinVanLeerC2F`](@ref), [`FCTBorisBook`](@ref), [`FCTZalesak`](@ref), and
[`TVDLimitedFluxC2F`](@ref) are all aliases for `LimitedFluxC2F` with a
particular `kernel`. At each cell face, the operator gathers a fixed window of
every argument -- the 4-point window of each center-valued argument and the
`Contravariant3` component of each face-valued argument, as specified by
`arg_gather_spec(kernel)` -- and wraps the pure scalar function
`kernel_flux(kernel, gathered...)` in a `Contravariant3Vector`. The boundary
conditions each kernel supports are given by `valid_bcs(kernel)`.

These operators apply nonlinear transformations to their arguments, so they
have no operator matrices, and the flux they produce at the outermost boundary
faces is only zero or one-sided: they are meant to be composed with an operator
that does not use those faces, such as a [`DivergenceF2C`](@ref) with
[`SetValue`](@ref) boundary conditions.
"""
struct LimitedFluxC2F{BCS, K} <: AdvectionOperator
    bcs::BCS
    kernel::K
    function LimitedFluxC2F(bcs::NamedTuple, kernel::K) where {K}
        assert_valid_bcs(operator_name(kernel), bcs, valid_bcs(kernel))
        new{typeof(bcs), K}(bcs, kernel)
    end
end

# `operator_name(kernel)` is the user-facing name of the kernel's operator
# alias, for error messages; it is defined beside each kernel.
function operator_name end

rebuild_op(op::LimitedFluxC2F, bcs) = LimitedFluxC2F(bcs, op.kernel)

# Backward compatibility for the replaced structs' public fields:
# `op.constraint` (LinVanLeerC2F) and `op.method` (TVDLimitedFluxC2F) forward
# to the kernel.
@inline Base.getproperty(op::LimitedFluxC2F, name::Symbol) =
    name === :constraint || name === :method ?
    getfield(getfield(op, :kernel), name) : getfield(op, name)

# What `stencil_interior` gathers for one argument of a LimitedFluxC2F operator.
struct FaceValue end        # Contravariant3 component of the argument at the face
struct FaceValueWindow end  # Contravariant3 components at the face and both neighbors
struct CenterWindow end     # the 4-point center window (a⁻⁻, a⁻, a⁺, a⁺⁺)
struct RawArg end           # the argument itself, e.g. a scalar like `dt`

gather_width(::FaceValue) = (0, 0)
gather_width(::FaceValueWindow) = (-1, 1)
gather_width(::CenterWindow) = (-half - 1, half + 1)
gather_width(::RawArg) = (0, 0)

Base.@propagate_inbounds face_ct3(space, arg, idx, hidx) =
    Geometry.contravariant3(
        getidx(space, arg, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )

Base.@propagate_inbounds gather_arg(::FaceValue, space, arg, idx, hidx) =
    face_ct3(space, arg, idx, hidx)
Base.@propagate_inbounds gather_arg(::FaceValueWindow, space, arg, idx, hidx) =
    (
        face_ct3(space, arg, idx - 1, hidx),
        face_ct3(space, arg, idx, hidx),
        face_ct3(space, arg, idx + 1, hidx),
    )
Base.@propagate_inbounds gather_arg(::CenterWindow, space, arg, idx, hidx) = (
    getidx(space, arg, idx - half - 1, hidx),
    getidx(space, arg, idx - half, hidx),
    getidx(space, arg, idx + half, hidx),
    getidx(space, arg, idx + half + 1, hidx),
)
gather_arg(::RawArg, space, arg, idx, hidx) = arg

Base.@propagate_inbounds gather_args(::Tuple{}, ::Tuple{}, space, idx, hidx) =
    ()
Base.@propagate_inbounds gather_args(specs, args, space, idx, hidx) = (
    gather_arg(first(specs), space, first(args), idx, hidx),
    gather_args(Base.tail(specs), Base.tail(args), space, idx, hidx)...,
)

stencil_interior_width(op::LimitedFluxC2F, args...) =
    map(gather_width, arg_gather_spec(op.kernel))

Base.@propagate_inbounds function stencil_interior(
    op::LimitedFluxC2F,
    space,
    idx,
    hidx,
    args...,
)
    gathered = gather_args(arg_gather_spec(op.kernel), args, space, idx, hidx)
    return Geometry.Contravariant3Vector(kernel_flux(op.kernel, gathered...))
end

# Exact-arity methods to avoid ambiguity with the generic
# `return_eltype(::AdvectionOperator, velocity, arg)`.
return_eltype(::LimitedFluxC2F, V, arg) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}
return_eltype(::LimitedFluxC2F, V, arg1, arg2) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

# The space staggering each gather spec requires of its argument, replacing the
# per-operator `return_space` signature constraints of the replaced operators.
valid_arg_space(::Union{FaceValue, FaceValueWindow}, arg_space) =
    arg_space isa AllFaceFiniteDifferenceSpace
valid_arg_space(::CenterWindow, arg_space) =
    arg_space isa AllCenterFiniteDifferenceSpace
valid_arg_space(::RawArg, arg_space) = true

function return_space(
    op::LimitedFluxC2F,
    flux_space::AllFaceFiniteDifferenceSpace,
    other_arg_spaces...,
)
    specs = arg_gather_spec(op.kernel)
    arg_spaces = (flux_space, other_arg_spaces...)
    length(specs) == length(arg_spaces) &&
        all(map(valid_arg_space, specs, arg_spaces)) || error(
        "$(operator_name(op.kernel)) was given arguments on the wrong spaces: \
         its face-valued (velocity or flux) arguments must be on face spaces \
         and its advected arguments on center spaces",
    )
    return flux_space
end

boundary_width(::LimitedFluxC2F, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    op::LimitedFluxC2F,
    bc::Union{FirstOrderOneSided, ThirdOrderOneSided},
    space,
    idx,
    hidx,
    args...,
)
    @assert idx <= left_face_boundary_idx(space) + 1
    return Geometry.Contravariant3Vector(
        kernel_left_boundary_flux(op.kernel, bc, space, idx, hidx, args...),
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    op::LimitedFluxC2F,
    bc::Union{FirstOrderOneSided, ThirdOrderOneSided},
    space,
    idx,
    hidx,
    args...,
)
    @assert idx >= right_face_boundary_idx(space) - 1
    return Geometry.Contravariant3Vector(
        kernel_right_boundary_flux(op.kernel, bc, space, idx, hidx, args...),
    )
end

# For every kernel except LinVanLeer's, FirstOrderOneSided zeroes the flux at
# the two faces nearest each boundary.
Base.@propagate_inbounds kernel_left_boundary_flux(
    kernel,
    ::FirstOrderOneSided,
    space,
    idx,
    hidx,
    args...,
) = zero(eltype(eltype(args[1])))
Base.@propagate_inbounds kernel_right_boundary_flux(
    kernel,
    ::FirstOrderOneSided,
    space,
    idx,
    hidx,
    args...,
) = zero(eltype(eltype(args[1])))

abstract type LimiterConstraint end
struct AlgebraicMean <: LimiterConstraint end
struct PositiveDefinite <: LimiterConstraint end
struct MonotoneHarmonic <: LimiterConstraint end
struct MonotoneLocalExtrema <: LimiterConstraint end

struct LinVanLeerKernel{C <: LimiterConstraint}
    constraint::C
end
operator_name(::LinVanLeerKernel) = "LinVanLeerC2F"
valid_bcs(::LinVanLeerKernel) = (FirstOrderOneSided, ThirdOrderOneSided)
arg_gather_spec(::LinVanLeerKernel) = (FaceValue(), CenterWindow(), RawArg())

"""
    LinVanLeerC2F

Following the van Leer class of limiters as noted in[Lin1994](@cite), four
limiter constraint options are provided for use with advection operators:

- `AlgebraicMean`: Algebraic mean, this guarantees neither positivity nor
  monotonicity (eq 2, `avg`)
- `PositiveDefinite`: Positive-definite with implicit diffusion based on local
  stencil extrema (eq 3b, 3c, 5a, 5b, `posd`)
- `MonotoneHarmonic`: Monotonicity preserving harmonic mean, this implies a strong
  monotonicity constraint (eq 4, `mono4`)
- `MonotoneLocalExtrema`: Monotonicity preserving, with extrema bounded by the
  edge cells in the stencil (eq 5, `mono5`)

The diffusion implied by these methods is proportional to the local upwind CFL
number. The `mismatch` Δ𝜙 = 0 returns the first-order upwind method. Special
cases (discussed in Lin et al (1994)) include setting the 𝜙_min = 0 or 𝜙_max =
saturation mixing ratio for water vapor are not considered here in favour of
the generalized local extrema in equation (5a, 5b).

Supported boundary conditions include:

 - [`FirstOrderOneSided`](@ref)
 - [`ThirdOrderOneSided`](@ref)
"""
const LinVanLeerC2F{BCS, C} = LimitedFluxC2F{BCS, LinVanLeerKernel{C}}
LinVanLeerC2F(; constraint, kwargs...) =
    LimitedFluxC2F(NamedTuple(kwargs), LinVanLeerKernel(constraint))
LinVanLeerC2F(bcs, constraint) =
    LimitedFluxC2F(NamedTuple(bcs), LinVanLeerKernel(constraint))

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
    # Eqn (2,5a,5b,5c): the mismatch Δ𝛼 is computed over the upwind-side window
    (b⁻, b⁰, b⁺) = upwind_select(v, (a⁻⁻, a⁻, a⁺), (a⁻, a⁺, a⁺⁺))
    Δ𝛼 = compute_Δ𝛼_linvanleer(b⁻, b⁰, b⁺, v, dt, constraint)
    return v * upwind_select(v, a⁻ + Δ𝛼 / 2, a⁺ - Δ𝛼 / 2)
end

@inline kernel_flux(k::LinVanLeerKernel, v³, (a⁻⁻, a⁻, a⁺, a⁺⁺), dt) =
    slope_limited_product(v³, a⁻, a⁻⁻, a⁺, a⁺⁺, dt, k.constraint)

# Pure boundary fluxes for LinVanLeerC2F, shared with the eager GPU
# implementation. `c1, c2, c3` are the values of `arg` at the three centers
# nearest the face: `(idx - half, idx + half, idx + half + 1)` on the left, and
# `(idx - half - 1, idx - half, idx + half)` on the right. FirstOrderOneSided
# blends the 1-point biased value with the opposite-side 3rd-order
# reconstruction; ThirdOrderOneSided uses the one-sided 3rd-order
# reconstruction regardless of the flow direction.
@inline lin_van_leer_left_flux(::FirstOrderOneSided, v³, c1, c2, c3) =
    upwind_biased_product(v³, c1, right_biased_3rd(c1, c2, c3))
@inline lin_van_leer_left_flux(::ThirdOrderOneSided, v³, c1, c2, c3) =
    v³ * right_biased_3rd(c1, c2, c3)
@inline lin_van_leer_right_flux(::FirstOrderOneSided, v³, c1, c2, c3) =
    upwind_biased_product(v³, left_biased_3rd(c1, c2, c3), c3)
@inline lin_van_leer_right_flux(::ThirdOrderOneSided, v³, c1, c2, c3) =
    v³ * left_biased_3rd(c1, c2, c3)

Base.@propagate_inbounds function lin_van_leer_left_boundary_flux(
    bc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = face_ct3(space, velocity, idx, hidx)
    # At the outermost face, `idx - half` is one center below the domain; clamp
    # it to the boundary center. (This read used to be out of bounds there.)
    c1 = getidx(space, arg, max(idx - half, left_center_boundary_idx(space)), hidx)
    c2 = getidx(space, arg, idx + half, hidx)
    c3 = getidx(space, arg, idx + half + 1, hidx)
    return lin_van_leer_left_flux(bc, v³, c1, c2, c3)
end
Base.@propagate_inbounds function lin_van_leer_right_boundary_flux(
    bc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = face_ct3(space, velocity, idx, hidx)
    c1 = getidx(space, arg, idx - half - 1, hidx)
    c2 = getidx(space, arg, idx - half, hidx)
    # At the outermost face, `idx + half` is one center above the domain; clamp
    # it to the boundary center. (This read used to be out of bounds there.)
    c3 = getidx(space, arg, min(idx + half, right_center_boundary_idx(space)), hidx)
    return lin_van_leer_right_flux(bc, v³, c1, c2, c3)
end

# Written as one method per boundary condition (rather than one method with a
# Union) so that each is strictly more specific than the zero-flux fallback.
Base.@propagate_inbounds kernel_left_boundary_flux(
    ::LinVanLeerKernel,
    bc::FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
) = lin_van_leer_left_boundary_flux(bc, space, idx, hidx, velocity, arg)
Base.@propagate_inbounds kernel_left_boundary_flux(
    ::LinVanLeerKernel,
    bc::ThirdOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
) = lin_van_leer_left_boundary_flux(bc, space, idx, hidx, velocity, arg)
Base.@propagate_inbounds kernel_right_boundary_flux(
    ::LinVanLeerKernel,
    bc::FirstOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
) = lin_van_leer_right_boundary_flux(bc, space, idx, hidx, velocity, arg)
Base.@propagate_inbounds kernel_right_boundary_flux(
    ::LinVanLeerKernel,
    bc::ThirdOrderOneSided,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
) = lin_van_leer_right_boundary_flux(bc, space, idx, hidx, velocity, arg)

# Window-based boundary fluxes, used by the eager GPU implementation of
# LimitedFluxC2F. They take the same gathered argument windows as
# `kernel_flux`, with window entries whose index lies outside the domain
# clamped to the boundary (so the values below match the clamped reads in the
# lazy boundary stencils above). `FT` is the flux scalar type.
@inline window_left_boundary_flux(kernel, ::FirstOrderOneSided, ::Type{FT}, gathered...) where {FT} =
    zero(FT)
@inline window_right_boundary_flux(kernel, ::FirstOrderOneSided, ::Type{FT}, gathered...) where {FT} =
    zero(FT)
# With no boundary condition, the flux at the boundary is NaN (as in the
# generic NullBoundaryCondition stencils).
@inline window_left_boundary_flux(kernel, ::NullBoundaryCondition, ::Type{FT}, gathered...) where {FT} =
    FT(NaN)
@inline window_right_boundary_flux(kernel, ::NullBoundaryCondition, ::Type{FT}, gathered...) where {FT} =
    FT(NaN)
# For LinVanLeer, the left boundary flux uses the three lowest centers of the
# clamped window, `(c0, c1, c2, c3)[2:4]`, and the right boundary flux uses the
# three highest, matching `lin_van_leer_left/right_boundary_flux`.
@inline window_left_boundary_flux(
    k::LinVanLeerKernel,
    bc::FirstOrderOneSided,
    ::Type{FT},
    v³,
    cwin,
    dt,
) where {FT} = lin_van_leer_left_flux(bc, v³, cwin[2], cwin[3], cwin[4])
@inline window_left_boundary_flux(
    k::LinVanLeerKernel,
    bc::ThirdOrderOneSided,
    ::Type{FT},
    v³,
    cwin,
    dt,
) where {FT} = lin_van_leer_left_flux(bc, v³, cwin[2], cwin[3], cwin[4])
@inline window_right_boundary_flux(
    k::LinVanLeerKernel,
    bc::FirstOrderOneSided,
    ::Type{FT},
    v³,
    cwin,
    dt,
) where {FT} = lin_van_leer_right_flux(bc, v³, cwin[1], cwin[2], cwin[3])
@inline window_right_boundary_flux(
    k::LinVanLeerKernel,
    bc::ThirdOrderOneSided,
    ::Type{FT},
    v³,
    cwin,
    dt,
) where {FT} = lin_van_leer_right_flux(bc, v³, cwin[1], cwin[2], cwin[3])

"""
    U = Upwind3rdOrderBiasedProductC2F(;boundaries)
    U.(v, x)

Compute the product of a face-valued vector field `v` and a center-valued field
`x` at cell faces by upwinding `x`, to third-order of accuracy, according to `v`
```math
U(v,x)[i] = \\begin{cases}
  v[i] \\left(-2 x[i-\\tfrac{3}{2}] + 10 x[i-\\tfrac{1}{2}] + 4 x[i+\\tfrac{1}{2}] \\right) / 12  \\textrm{, if } v[i] > 0 \\\\
  v[i] \\left(4 x[i-\\tfrac{1}{2}] + 10 x[i+\\tfrac{1}{2}] -2 x[i+\\tfrac{3}{2}]  \\right) / 12  \\textrm{, if } v[i] < 0
  \\end{cases}
```
This stencil is based on [WickerSkamarock2002](@cite), eq. 4(a).

Supported boundary conditions are:
- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind scheme to
  compute `x` on the left boundary, and the first-order upwind scheme to
  compute `x` on the right boundary.
- [`ThirdOrderOneSided(x₀)`](@ref): uses the third-order downwind reconstruction
  to compute `x` on the left boundary, and the third-order upwind
  reconstruction to compute `x` on the right boundary.

!!! note
    These boundary conditions do not define the value at the actual
    boundary faces, and so this operator should not be materialized directly: it
    needs to be composed with another operator that does not make use of this
    value, e.g. a [`DivergenceF2C`](@ref) operator, with a [`SetValue`]
    (@ref) boundary.
"""
struct Upwind3rdOrderBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
    function Upwind3rdOrderBiasedProductC2F(; kwargs...)
        assert_valid_bcs(
            "Upwind3rdOrderBiasedProductC2F",
            kwargs,
            (FirstOrderOneSided, ThirdOrderOneSided),
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    Upwind3rdOrderBiasedProductC2F(bcs) =
        Upwind3rdOrderBiasedProductC2F(; bcs...)
end

return_eltype(::Upwind3rdOrderBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::Upwind3rdOrderBiasedProductC2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = velocity_space


stencil_interior_width(::Upwind3rdOrderBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half - 1, half + 1))


boundary_width(::Upwind3rdOrderBiasedProductC2F, ::AbstractBoundaryCondition) =
    2

struct BorisBookKernel end
operator_name(::BorisBookKernel) = "FCTBorisBook"
valid_bcs(::BorisBookKernel) = (FirstOrderOneSided,)
arg_gather_spec(::BorisBookKernel) = (FaceValue(), CenterWindow())

"""
    U = FCTBorisBook(;boundaries)
    U.(v, x)

Correct the flux using the flux-corrected transport formulation by Boris and
Book [BorisBook1973](@cite).

Input arguments:
- a face-valued vector field `v`
- a center-valued field `x`

```math
Ac(v,x)[i] =
  s[i] \\max \\left\\{0, \\min \\left[ |v[i] |, s[i] \\left( x[i+\\tfrac{3}{2}] - x[i+\\tfrac{1}{2}]  \\right) ,  s[i] \\left( x[i-\\tfrac{1}{2}] - x[i-\\tfrac{3}{2}]  \\right) \\right] \\right\\},
```

where ``s[i] = +1`` if  `` v[i] \\geq 0`` and ``s[i] = -1`` if  `` v
[i] \\leq 0``, and ``Ac`` represents the resulting corrected antidiffusive
flux. This formulation is based on [BorisBook1973](@cite), as reported in
[durran2010](@cite) section 5.4.1.

Supported boundary conditions are:
- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind reconstruction
  to compute `x` on the left boundary, and the first-order upwind
  reconstruction to compute `x` on the right boundary.

!!! note
    Similar to the [`Upwind3rdOrderBiasedProductC2F`](@ref) operator, these
    boundary conditions do not define the value at the actual boundary faces,
    and so this operator cannot be materialized directly: it needs to be
    composed with another operator that does not make use of this value, e.g. a
    [`DivergenceF2C`](@ref) operator, with a [`SetValue`](@ref) boundary.
"""
const FCTBorisBook{BCS} = LimitedFluxC2F{BCS, BorisBookKernel}
FCTBorisBook(; kwargs...) = LimitedFluxC2F(NamedTuple(kwargs), BorisBookKernel())
FCTBorisBook(bcs) = FCTBorisBook(; bcs...)

# Equivalent to the sign-based form
# `sign(v) * max(0, min(|v|, sign(v)(a⁺⁺ - a⁺), sign(v)(a⁻ - a⁻⁻)))`, including
# at `v == 0`, where both give 0 (so no special case is needed).
fct_boris_book(v, a⁻⁻, a⁻, a⁺, a⁺⁺) = upwind_select(
    v,
    max(zero(v), min(v, a⁺⁺ - a⁺, a⁻ - a⁻⁻)),
    min(zero(v), max(v, a⁺⁺ - a⁺, a⁻ - a⁻⁻)),
)

@inline kernel_flux(::BorisBookKernel, v³, (a⁻⁻, a⁻, a⁺, a⁺⁺)) =
    fct_boris_book(v³, a⁻⁻, a⁻, a⁺, a⁺⁺)

struct ZalesakKernel end
operator_name(::ZalesakKernel) = "FCTZalesak"
valid_bcs(::ZalesakKernel) = (FirstOrderOneSided,)
arg_gather_spec(::ZalesakKernel) =
    (FaceValueWindow(), CenterWindow(), CenterWindow())

"""
    U = FCTZalesak(;boundaries)
    U.(A, Φ, Φᵗᵈ)

Correct the flux using the flux-corrected transport formulation by Zalesak
[zalesak1979fully](@cite).

Input arguments:
- a face-valued vector field `A`
- a center-valued field `Φ`
- a center-valued field `Φᵗᵈ`

```math
Φ_j^{n+1} = Φ_j^{td} - (C_{j+\\frac{1}{2}}A_{j+\\frac{1}{2}} - C_{j-\\frac{1}{2}}A_{j-\\frac{1}{2}})
```

This stencil is based on [zalesak1979fully](@cite), as reported in [durran2010]
(@cite) section 5.4.2, where ``C`` denotes the corrected antidiffusive flux.

Supported boundary conditions are:

- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind reconstruction
  to compute `x` on the left boundary, and the first-order upwind
  reconstruction to compute `x` on the right boundary.

!!! note
    Similar to the [`Upwind3rdOrderBiasedProductC2F`](@ref) operator, these
    boundary conditions do not define the value at the actual boundary faces,
    and so this operator cannot be materialized directly: it needs to be
    composed with another operator that does not make use of this value, e.g.
    a [`DivergenceF2C`](@ref) operator, with a [`SetValue`](@ref) boundary.
"""
const FCTZalesak{BCS} = LimitedFluxC2F{BCS, ZalesakKernel}
FCTZalesak(; kwargs...) = LimitedFluxC2F(NamedTuple(kwargs), ZalesakKernel())
FCTZalesak(bcs) = FCTZalesak(; bcs...)

@inline function kernel_flux(
    ::ZalesakKernel,
    (A₋₁, A, A₊₁),
    (ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂),
    (ϕ₋₃₂ᵗᵈ, ϕ₋₁₂ᵗᵈ, ϕ₊₁₂ᵗᵈ, ϕ₊₃₂ᵗᵈ),
)
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

    return upwind_select(A, min(R₊₁₂⁺, R₋₁₂⁻), min(R₋₁₂⁺, R₊₁₂⁻)) * A
end

"""
    AbstractTVDSlopeLimiter

An asbtract TVD-slope limiter type. Use `subtypes(AbstractTVDSlopeLimiter)`
to see the supported subtypes. See

`TVDLimitedFluxC2F` for the general formulation.
"""
abstract type AbstractTVDSlopeLimiter end


"""
    U = RZeroLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct RZeroLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RZeroLimiter) = zero(r)

"""
    U = RHalfLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct RHalfLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RHalfLimiter) = one(r) / 2

"""
    U = RMaxLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct RMaxLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::RMaxLimiter) = one(r)

"""
    U = MinModLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct MinModLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::MinModLimiter) = max(0, min(1, r))

"""
    U = KorenLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct KorenLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::KorenLimiter) = max(0, min(2r, (1 + 2r) / 3, 2))

"""
    U = SuperbeeLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct SuperbeeLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::SuperbeeLimiter) = max(0, min(1, r), min(2, r))

"""
    U = MonotonizedCentralLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
`TVDLimitedFluxC2F` for the general formulation.
"""
struct MonotonizedCentralLimiter <: AbstractTVDSlopeLimiter end
limiter_coeff(r, ::MonotonizedCentralLimiter) = max(0, min(2r, (1 + r) / 2, 2))

struct TVDKernel{M <: AbstractTVDSlopeLimiter}
    method::M
end
operator_name(::TVDKernel) = "TVDLimitedFluxC2F"
valid_bcs(::TVDKernel) = (FirstOrderOneSided,)
arg_gather_spec(::TVDKernel) = (FaceValue(), CenterWindow(), FaceValue())

"""
    TVDLimitedFluxC2F{BCS, M} <: AdvectionOperator

    U = TVDLimitedFluxC2F(;boundaries)
    U.(𝒜, Φ, 𝓊)

`𝒜`, following the notation of Durran (Numerical Methods for Fluid Dynamics, 2ⁿᵈ
ed.) is the antidiffusive flux given by

``` 𝒜 = ℱʰ - ℱˡ ``` where h and l superscripts represent the high and lower
order (monotone) fluxes respectively. The effect of the TVD limiters is then to
adjust the flux

``` F_{j+1/2} = F^{l}_{j+1/2} + C_{j+1/2}(F^{h}_{j+1/2} - F^{l}_{j+1/2}) where
C_{j+1/2} is the multiplicative limiter which is a function of ```

the ratio of the slope of the solution across a cell interface.

 - `C=1` recovers the high order flux.
 - `C=0` recovers the low order flux.

Supported limiter types are

- RZeroLimiter (returns low order flux)
- RHalfLimiter (flux multiplier == 1/2)
- RMaxLimiter (returns high order flux)
- MinModLimiter
- KorenLimiter
- SuperbeeLimiter
- MonotonizedCentralLimiter

Supported boundary conditions are:

 - [`FirstOrderOneSided`](@ref)
"""
const TVDLimitedFluxC2F{BCS, M} = LimitedFluxC2F{BCS, TVDKernel{M}}
TVDLimitedFluxC2F(; method, kwargs...) =
    LimitedFluxC2F(NamedTuple(kwargs), TVDKernel(method))
TVDLimitedFluxC2F(bcs, method) =
    LimitedFluxC2F(NamedTuple(bcs), TVDKernel(method))

@inline function kernel_flux(k::TVDKernel, A, (ϕ₋₃₂, ϕ₋₁₂, ϕ₊₁₂, ϕ₊₃₂), 𝓊)
    Δϕ = ϕ₊₁₂ - ϕ₋₁₂ + eps(typeof(ϕ₋₁₂))
    # Δϕ_clipped = sign(Δϕ) * max(abs(Δϕ), eps(typeof(Δϕ)))
    r = upwind_select(𝓊, ϕ₋₁₂ - ϕ₋₃₂, ϕ₊₃₂ - ϕ₊₁₂) / Δϕ # Δϕ_clipped
    return limiter_coeff(r, k.method) * A
end

abstract type BoundaryOperator <: FiniteDifferenceOperator end

"""
    SetBoundaryOperator(;boundaries...)

This operator is the identity in the interior, and replaces the value at each boundary
for which a condition is given. It preserves the space of its argument, so it modifies
the boundary faces of a face field or the boundary center cells of a center field. A side
with no condition is left untouched.

The following boundary conditions are supported:

 - [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
 - [`SetGradient(val)`](@ref): set the value to be `val` on the boundary, projected onto
   the `Covariant3` axis.
 - [`SetCurl(val)`](@ref): set the value to be `val` on the boundary, projected onto the
   `Contravariant123` axis.
 - [`SetDivergence(val)`](@ref): set the value to be `val` on the boundary.

The projecting conditions exist so that this operator can reapply the boundary conditions
of the operator it was derived from when a broadcast is rewritten as an operator matrix
multiply; see `MatrixFields.modifies_output`.
"""
struct SetBoundaryOperator{BCS} <: BoundaryOperator
    bcs::BCS
    function SetBoundaryOperator(; kwargs...)
        assert_valid_bcs(
            "SetBoundaryOperator",
            kwargs,
            (SetValue, SetGradient, SetCurl, SetDivergence),
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
boundary_width(::SetBoundaryOperator, ::AbstractBoundaryCondition) = 0
boundary_width(
    ::SetBoundaryOperator,
    ::Union{SetValue, SetGradient, SetCurl, SetDivergence},
) = 1
# The value a `SetBoundaryOperator` imposes at a boundary. `SetGradient` and `SetCurl`
# hold values in the axis the operator they were taken from writes its output in, so they
# are projected onto that axis; `SetValue` and `SetDivergence` are imposed as given.
Base.@propagate_inbounds imposed_boundary_value(
    bc::Union{SetValue, SetDivergence},
    space,
    idx,
    hidx,
) = getidx(space, bc.val, nothing, hidx)
Base.@propagate_inbounds imposed_boundary_value(
    bc::SetGradient,
    space,
    idx,
    hidx,
) = Geometry.project(
    Geometry.Covariant3Axis(),
    getidx(space, bc.val, nothing, hidx),
    Geometry.LocalGeometry(space, idx, hidx),
)
Base.@propagate_inbounds imposed_boundary_value(bc::SetCurl, space, idx, hidx) =
    Geometry.project(
        Geometry.Contravariant123Axis(),
        getidx(space, bc.val, nothing, hidx),
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
# TODO: we should probably make the axis the operator is working over as part of the operator type
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

We note that the usual division factor ``1 / \\Delta z`` that appears in a first-order
finite difference operator is accounted for in the `LocalVector` basis. Hence, users
need to cast the output of the `GradientF2C` to a `UVector`, `VVector` or `WVector`,
according to the type of domain on which the operator is defined.

The following boundary conditions are supported:
 - by default, the value of `x` at the boundary face will be used.
 - [`SetValue(x₀)`](@ref): calculate the gradient assuming the value at the
   boundary is `x₀`. For the left boundary, this becomes:
```math
G(x)[1]³ = x[1+\\tfrac{1}{2}] - x₀
```
- [`Extrapolate()`](@ref): set the value at the center closest to the boundary
to be the same as the neighbouring interior value. For the left boundary, this becomes:
```math
G(x)[1]³ = G(x)[2]³
```
"""
struct GradientF2C{BCS} <: GradientOperator
    bcs::BCS
    function GradientF2C(; kwargs...)
        assert_valid_bcs("GradientF2C", kwargs, (SetValue, Extrapolate))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    GradientF2C(bcs) = GradientF2C(; bcs...)
end

return_space(::GradientF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::GradientF2C, arg) = ((-half, half),)

boundary_width(::GradientF2C, ::AbstractBoundaryCondition) = 0

boundary_width(::GradientF2C, ::SetValue) = 1
boundary_width(::GradientF2C, ::Extrapolate) = 1
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
"""
struct GradientC2F{BC} <: GradientOperator
    bcs::BC
    function GradientC2F(; kwargs...)
        assert_valid_bcs("GradientC2F", kwargs, (SetGradient,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    GradientC2F(bcs) = GradientC2F(; bcs...)
end

return_space(::GradientC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::GradientC2F, arg) = ((-half, half),)
boundary_width(::GradientC2F, ::AbstractBoundaryCondition) = 1

"""
    UG = UpwindBiasedGradient()
    UG.(v, θ)

Compute the gradient of a field `θ` by upwinding it according to the direction
of a vector field `v` on the same space. The gradient stencil is determined by
the sign of the 3rd contravariant component of `v`:
```math
UG(\\boldsymbol{v}, θ)[i] = \\begin{cases}
    G(L(θ))[i] \\textrm{, if } v^3[i] > 0 \\\\
    G(R(θ))[i] \\textrm{, if } v^3[i] < 0
\\end{cases}
```
where `G` is a gradient operator and `L`/`R` are left/right-bias operators. When
`θ` and `v` are located on centers, `G = GradientF2C()`, `L = LeftBiasedC2F()`,
and `R = RightBiasedC2F()`. When they are located on faces, `G = GradientC2F()`,
`L = LeftBiasedF2C()`, and `R = RightBiasedF2C()`.

No boundary conditions are currently supported. The default behavior on the left
boundary (with index `i_min`) is
```math
UG(\\boldsymbol{v}, θ)[i_min] = G(R(θ))[i_min]
```
and the default behavior on the right boundary (with index `i_max`) is
```math
UG(\\boldsymbol{v}, θ)[i_max] = G(L(θ))[i_max]
```
"""
struct UpwindBiasedGradient{BCS} <: FiniteDifferenceOperator
    bcs::BCS
end
function UpwindBiasedGradient(; kwargs...)
    assert_no_bcs("UpwindBiasedGradient", kwargs)
    return UpwindBiasedGradient(NamedTuple())
end

return_eltype(::UpwindBiasedGradient, velocity, arg) =
    Geometry.gradient_result_type(Val((3,)), eltype(arg))

return_space(
    ::UpwindBiasedGradient,
    velocity_space::AllCenterFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = arg_space
return_space(
    ::UpwindBiasedGradient,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllFaceFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::UpwindBiasedGradient, velocity, arg) =
    ((0, 0), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::UpwindBiasedGradient,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    a⁺ = getidx(space, arg, idx + 1, hidx)
    a = getidx(space, arg, idx, hidx)
    a⁻ = getidx(space, arg, idx - 1, hidx)
    v = Geometry.contravariant3(
        getidx(space, velocity, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Covariant3Vector(1) ⊗
           upwind_symmetric_difference(sign(v), a⁻, a, a⁺)
end

boundary_width(::UpwindBiasedGradient, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::UpwindBiasedGradient,
    ::NullBoundaryCondition,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(space, arg, idx + 1, hidx)
    a = getidx(space, arg, idx, hidx)
    return Geometry.Covariant3Vector(1) ⊗ (a⁺ - a)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::UpwindBiasedGradient,
    ::NullBoundaryCondition,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a = getidx(space, arg, idx, hidx)
    a⁻ = getidx(space, arg, idx - 1, hidx)
    return Geometry.Covariant3Vector(1) ⊗ (a - a⁻)
end

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
 - by default, the value of `v` at the boundary face will be used.
 - [`SetValue(v₀)`](@ref): calculate the divergence assuming the value at the
   boundary is `v₀`. For the left boundary, this becomes:
```math
D(v)[1] = (Jv³[1+\\tfrac{1}{2}] - Jv³₀) / J[i]
```
- [`Extrapolate()`](@ref): set the value at the center closest to the boundary
  to be the same as the neighbouring interior value. For the left boundary, this
  becomes:
```math
D(v)[1]³ = D(v)[2]³

- [`SetDivergence(v₀)`](@ref): set the divergence at the cell center  closest to
  the boundary

"""
struct DivergenceF2C{BCS} <: DivergenceOperator
    bcs::BCS
    function DivergenceF2C(; kwargs...)
        assert_valid_bcs(
            "DivergenceF2C",
            kwargs,
            (SetValue, Extrapolate, SetDivergence),
        )
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    DivergenceF2C(bcs) = DivergenceF2C(; bcs...)
end

return_space(::DivergenceF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::DivergenceF2C, arg) = ((-half, half),)
boundary_width(::DivergenceF2C, ::AbstractBoundaryCondition) = 0
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

# Kernels and other non-bc fields are isbits, so `rebuild_op` can reuse them
# as-is.
@inline adapt_fd_operator(to, op, bcs) = rebuild_op(op, adapt_bcs(to, bcs))

@inline adapt_bcs(to, bcs) = NamedTuple{keys(bcs)}(
    unrolled_map(bc -> Adapt.adapt_structure(to, bc), values(bcs)),
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
"""
struct DivergenceC2F{BC} <: DivergenceOperator
    bcs::BC
    function DivergenceC2F(; kwargs...)
        assert_valid_bcs("DivergenceC2F", kwargs, (SetDivergence,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    DivergenceC2F(bcs) = DivergenceC2F(; bcs...)
end

return_space(::DivergenceC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::DivergenceC2F, arg) = ((-half, half),)

boundary_width(::DivergenceC2F, ::AbstractBoundaryCondition) = 1


abstract type CurlFiniteDifferenceOperator <: FiniteDifferenceOperator end
return_eltype(::CurlFiniteDifferenceOperator, arg) =
    Geometry.curl_result_type(Val((3,)), eltype(arg))

"""
    C = CurlC2F(;boundaryname=boundarycondition...)
    C.(v)

Compute the vertical-derivative contribution to the curl of a center-valued
covariant vector field `v`. It acts on the horizontal covariant components of
`v` (that is it only depends on ``v₁`` and ``v₂``), and will return a face-valued horizontal
contravariant vector field (that is ``C(v)³ = 0``).

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
"""
struct CurlC2F{BC} <: CurlFiniteDifferenceOperator
    bcs::BC
    function CurlC2F(; kwargs...)
        assert_valid_bcs("CurlC2F", kwargs, (SetCurl,))
        new{typeof(NamedTuple(kwargs))}(NamedTuple(kwargs))
    end
    CurlC2F(bcs) = CurlC2F(; bcs...)
end

return_space(::CurlC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())


stencil_interior_width(::CurlC2F, arg) = ((-half, half),)

boundary_width(::CurlC2F, ::AbstractBoundaryCondition) = 1



# code for figuring out boundary widths
# TODO: should move this to `instantiate` and store this in the StencilBroadcasted object?

_stencil_interior_width(bc::StencilBroadcasted) =
    stencil_interior_width(bc.op, bc.args...)

"""
    left_interior_idx(space::AbstractSpace, op::FiniteDifferenceOperator, bc::AbstractBoundaryCondition, args..)

The index of the left-most interior point of the operator `op` with boundary
`bc` when used with arguments `args...`. By default, this is
```julia
left_idx(space) + boundary_width(op, bc)
```
but can be overwritten for specific stencil types (e.g. if the stencil is
assymetric).
"""
@inline function left_interior_idx(
    space::AbstractSpace,
    op::FiniteDifferenceOperator,
    bc::AbstractBoundaryCondition,
    args...,
)
    left_idx(space) + boundary_width(op, bc)
end

"""
    right_interior_idx(space::AbstractSpace, op::FiniteDifferenceOperator, bc::AbstractBoundaryCondition, args..)

The index of the right-most interior point of the operator `op` with boundary
`bc` when used with arguments `args...`. By default, this is
```julia
right_idx(space) - boundary_width(op, bc)
```
but can be overwritten for specific stencil types (e.g. if the stencil is
assymetric).
"""
@inline function right_interior_idx(
    space::AbstractSpace,
    op::FiniteDifferenceOperator,
    bc::AbstractBoundaryCondition,
    args...,
)
    right_idx(space) - boundary_width(op, bc)
end


@inline _left_interior_window_idx_args(args::Tuple, space, loc) =
    unrolled_map(args) do arg
        left_interior_window_idx(arg, space, loc)
    end

"""
    left_interior_window_idx(arg, space, loc)

Compute the index of the leftmost point which uses only the interior stencil of the space.
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
    unrolled_map(args) do arg
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
    boundary_condition = Operators.get_boundary(op, loc)
    return idx < Operators.left_interior_idx(
        space,
        op,
        boundary_condition,
        args...,
    )
end

@inline function should_call_right_boundary(idx, space, op, args...)
    Topologies.isperiodic(space) && return false
    loc = right_boundary_window(space)
    boundary_condition = Operators.get_boundary(op, loc)
    return idx > Operators.right_interior_idx(
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

function vidx(space::AllFaceFiniteDifferenceSpace, idx)
    @assert idx isa PlusHalf
    v = idx + half
    if Topologies.isperiodic(space)
        v = mod1(v, Spaces.nlevels(space))
    end
    return v
end
function vidx(space::AllCenterFiniteDifferenceSpace, idx)
    @assert idx isa Integer
    v = idx
    if Topologies.isperiodic(space)
        v = mod1(v, Spaces.nlevels(space))
    end
    return v
end
function vidx(space::AbstractSpace, idx)
    return 1
end

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

# recursively unwrap getidx broadcast arguments in a way that is statically reducible by the optimizer
@generated function call_bc_f(f::F, space, idx, hidx, args...) where {F}
    N = length(args)
    return quote
        Base.@_propagate_inbounds_meta
        Base.Cartesian.@ncall $N f i -> getidx(space, args[i], idx, hidx)
    end
end

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(call_bc_f)
        m.recursion_relation = dont_limit
    end
    for m in methods(getidx)
        m.recursion_relation = dont_limit
    end
    for m in methods(gather_args)
        m.recursion_relation = dont_limit
    end
end

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
        unrolled_map(args) do arg
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
        # but this could lead to a confusing error message (same space type but different instances)
        inferred_stencil_spaces_error(typeof(dest_space), typeof(result_space))
    end
    # the default Base behavior is to instantiate a Broadcasted object with the same axes as the dest
    return copyto!(
        dest,
        Base.Broadcast.instantiate(
            Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, dest_space),
        ),
    )
end

Base.@propagate_inbounds column(op::FiniteDifferenceOperator, inds...) =
    rebuild_op(op, column_args(op.bcs, inds...))
Base.@propagate_inbounds column(sbc::StencilBroadcasted{S}, inds...) where {S} =
    StencilBroadcasted{S}(
        column(sbc.op, inds...),
        column_args(sbc.args, inds...),
        column(sbc.axes, inds...),
    )

#TODO: the optimizer dies with column broadcast expressions over a certain complexity
if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(column)
        m.recursion_relation = dont_limit
    end
    for m in methods(column_args)
        m.recursion_relation = dont_limit
    end
end

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

@inline function reconstruct_placeholder_broadcasted(
    parent_space::Spaces.AbstractSpace,
    sbc::StencilBroadcasted{Style},
) where {Style}
    space = reconstruct_placeholder_space(axes(sbc), parent_space)
    args = _reconstruct_placeholder_broadcasted(space, sbc.args)
    return StencilBroadcasted{Style}(sbc.op, args, space, sbc.work)
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
    end
    @assert li <= lw <= rw <= ri
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

Returns a Bool indicating whether or not the broadcasted object supports
shared memory, allowing us to dispatch into an optimized kernel.

This function and dispatch should be removed once all operators support
shared memory.
"""
function fd_shmem_is_supported end

"""
    any_fd_shmem_supported(::Base.Broadcast.AbstractBroadcasted)

Returns a Bool indicating if any operators in the broadcasted object support 
finite difference shared memory shmem.
"""
function any_fd_shmem_supported end

"""
    promote_bcs

Used to promote integer-specified boundary conditions to the
given type (the space's undertype) so that `getidx` is
type-stable throughout the entire broadcast expression.

This is an internal method.
"""
@inline function promote_bcs(
    op::FiniteDifferenceOperator,
    ::Type{FT},
) where {FT}
    if hasfield(typeof(op), :bcs)
        rebuild_op(op, promote_bcs(op.bcs, FT))
    else
        op
    end
end

@inline promote_bcs(x::Fields.Field, ::Type{FT}) where {FT} = x
@inline promote_bcs(bcs::@NamedTuple{}, ::Type{FT}) where {FT} = NamedTuple()
@inline promote_bcs(bcs::NamedTuple{N, V}, ::Type{FT}) where {FT} where {N, V} =
    NamedTuple{N}(map(x -> promote_bc(x, FT), values(bcs)))

"""
    promote_bc

Used to promote integer-specified boundary conditions to the
given type (the space's undertype) so that `getidx` is
type-stable throughout the entire broadcast expression.

This is an internal method.
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


if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(reconstruct_placeholder_broadcasted)
        m.recursion_relation = dont_limit
    end
    for m in methods(_reconstruct_placeholder_broadcasted)
        m.recursion_relation = dont_limit
    end
end

"""
    use_fd_shmem()

Allows users to, from global scope, enable finite
difference shmem for operators that support it.
TODO: ~30% slowdown was noticed with CC 0.14.31 
in Aquaplanet benchmarks. This may need attention in 
future releases

## Usage
```julia
Operators.use_fd_shmem() = false
```
"""
use_fd_shmem() = false
