import ..Utilities: PlusHalf, half, UnrolledFunctions

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



left_idx(space::AllCenterFiniteDifferenceSpace) =
    left_center_boundary_idx(space)
right_idx(space::AllCenterFiniteDifferenceSpace) =
    right_center_boundary_idx(space)
left_idx(space::AllFaceFiniteDifferenceSpace) = left_face_boundary_idx(space)
right_idx(space::AllFaceFiniteDifferenceSpace) = right_face_boundary_idx(space)

left_center_boundary_idx(space::AllFiniteDifferenceSpace) = 1
right_center_boundary_idx(space::AllFiniteDifferenceSpace) = size(
    Spaces.local_geometry_data(Spaces.space(space, Spaces.CellCenter())),
    4,
)
left_face_boundary_idx(space::AllFiniteDifferenceSpace) = half
right_face_boundary_idx(space::AllFiniteDifferenceSpace) =
    size(
        Spaces.local_geometry_data(Spaces.space(space, Spaces.CellFace())),
        4,
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
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        v = mod1(v, Spaces.nlevels(space))
    end
    i, j, h = hidx
    local_geom =
        Grids.local_geometry_data(Spaces.grid(space), Grids.CellCenter())
    return @inbounds local_geom[CartesianIndex(i, j, 1, v, h)]
end
Base.@propagate_inbounds function Geometry.LocalGeometry(
    space::AllFiniteDifferenceSpace,
    idx::PlusHalf,
    hidx,
)
    v = idx + half
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        v = mod1(v, Spaces.nlevels(space))
    end
    i, j, h = hidx
    local_geom = Grids.local_geometry_data(Spaces.grid(space), Grids.CellFace())
    return @inbounds local_geom[CartesianIndex(i, j, 1, v, h)]
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

"""
    NullBoundaryCondition()

This is used as a placeholder when no other boundary condition can be applied.
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

get_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} =
    hasproperty(op.bcs, name) ? getproperty(op.bcs, name) :
    NullBoundaryCondition()

get_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} =
    hasproperty(op.bcs, name) ? getproperty(op.bcs, name) :
    NullBoundaryCondition()

has_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} = hasproperty(op.bcs, name)

has_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} = hasproperty(op.bcs, name)


abstract type AbstractStencilStyle <: Fields.AbstractFieldStyle end

# the .f field is an operator
struct StencilStyle <: AbstractStencilStyle end

struct ColumnStencilStyle <: AbstractStencilStyle end

AbstractStencilStyle(::ClimaComms.AbstractCPUDevice) = ColumnStencilStyle

"""
    StencilBroadcasted{Style}(op, args[,axes[, work]])

This is similar to a `Base.Broadcast.Broadcasted` object.

This is returned by `Base.Broadcast.broadcasted(op::FiniteDifferenceOperator)`.
"""
struct StencilBroadcasted{Style, Op, Args, Axes} <: OperatorBroadcasted{Style}
    op::Op
    args::Args
    axes::Axes
end
StencilBroadcasted{Style}(
    op::Op,
    args::Args,
    axes::Axes = nothing,
) where {Style, Op, Args, Axes} =
    StencilBroadcasted{Style, Op, Args, Axes}(op, args, axes)

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
        Base.Broadcast.check_broadcast_axes(axes, args...)
    end
    Style = AbstractStencilStyle(ClimaComms.device(axes))
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
    Style = AbstractStencilStyle(ClimaComms.device(axes))
    return Base.Broadcast.Broadcasted{Style}(bc.f, args, axes)
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
    stencil_interior(::Op, loc, space, idx, args...)

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
    stencil_left_boundary(::Op, ::BC, loc, idx, args...)

Defines the stencil of operator `Op` at `idx` near the left boundary, with boundary condition `BC`.
"""
function stencil_left_boundary end

"""
    stencil_right_boundary(::Op, ::BC, loc, idx, args...)

Defines the stencil of operator `Op` at `idx` near the right boundary, with boundary condition `BC`.
"""
function stencil_right_boundary end


abstract type InterpolationOperator <: FiniteDifferenceOperator end

"""
    InterpolateF2C()

Interpolate from face to center mesh. No boundary conditions are required (or supported).
"""
struct InterpolateF2C{BCS} <: InterpolationOperator
    bcs::BCS
end
InterpolateF2C(; kwargs...) = InterpolateF2C(NamedTuple(kwargs))

return_space(::InterpolateF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::InterpolateF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::InterpolateF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
end

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
- [`SetGradient(v)`](@ref): set the value at the boundary such that the gradient
  is `v`. At the left boundary the stencil is
```math
I(x)[\\tfrac{1}{2}] = x[1] - \\frac{1}{2} v³
```
- [`Extrapolate`](@ref): use the closest interior point as the boundary value.
  At the left boundary the stencil is
```math
I(x)[\\tfrac{1}{2}] = x[1]
```
"""
struct InterpolateC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
InterpolateC2F(; kwargs...) = InterpolateC2F(NamedTuple(kwargs))

return_space(::InterpolateC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::InterpolateC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::InterpolateC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
end
boundary_width(::InterpolateC2F, ::AbstractBoundaryCondition) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    v₃ = Geometry.covariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ ⊟ RecursiveApply.rdiv(v₃, 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    v₃ = Geometry.covariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ ⊞ RecursiveApply.rdiv(v₃, 2)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁺
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    a⁻
end

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
end
LeftBiasedC2F(; kwargs...) = LeftBiasedC2F(NamedTuple(kwargs))

return_space(::LeftBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::LeftBiasedC2F, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiasedC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx - half, hidx)

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

Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiasedC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

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
end
LeftBiasedF2C(; kwargs...) = LeftBiasedF2C(NamedTuple(kwargs))

return_space(::LeftBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::LeftBiasedF2C, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiasedF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx - half, hidx)
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
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

"""
    L = LeftBiased3rdOrderC2F(;boundaries)
    L.(x)

Interpolate a center-value field to a face-valued field from the left, using a 3rd-order reconstruction.
```math
L(x)[i] =  \\left(-2 x[i-\\tfrac{3}{2}] + 10 x[i-\\tfrac{1}{2}] + 4 x[i+\\tfrac{1}{2}] \\right) / 12
```

Only the left boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
L(x)[\\tfrac{1}{2}] = x_0
```
"""
struct LeftBiased3rdOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
LeftBiased3rdOrderC2F(; kwargs...) = LeftBiased3rdOrderC2F(NamedTuple(kwargs))

return_space(::LeftBiased3rdOrderC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::LeftBiased3rdOrderC2F, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiased3rdOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) =
    (
        -2 * getidx(space, arg, loc, idx - 1 - half, hidx) +
        10 * getidx(space, arg, loc, idx - half, hidx) +
        4 * getidx(space, arg, loc, idx + half, hidx)
    ) / 12

left_interior_idx(
    space::AbstractSpace,
    ::LeftBiased3rdOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space) + 2
right_interior_idx(
    space::AbstractSpace,
    ::LeftBiased3rdOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space) - 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased3rdOrderC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

"""
    L = LeftBiased3rdOrderF2C(;boundaries)
    L.(x)

Interpolate a face-value field to a center-valued field from the left, using a 3rd-order reconstruction.
```math
L(x)[i+\\tfrac{1}{2}] =  \\left(-2 x[i-1] + 10 x[i] + 4 x[i+1] \\right) / 12
```

Only the left boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
L(x)[1] = x_0
```
"""
struct LeftBiased3rdOrderF2C{BCS} <: InterpolationOperator
    bcs::BCS
end
LeftBiased3rdOrderF2C(; kwargs...) = LeftBiased3rdOrderF2C(NamedTuple(kwargs))

return_space(::LeftBiased3rdOrderF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::LeftBiased3rdOrderF2C, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiased3rdOrderF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
) =
    (
        -2 * getidx(space, arg, loc, idx - 1 - half, hidx) +
        10 * getidx(space, arg, loc, idx - half, hidx) +
        4 * getidx(space, arg, loc, idx + half, hidx)
    ) / 12

left_interior_idx(
    space::AbstractSpace,
    ::LeftBiased3rdOrderF2C,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space) + 1
right_interior_idx(
    space::AbstractSpace,
    ::LeftBiased3rdOrderF2C,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space)

Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased3rdOrderF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
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
end
RightBiasedC2F(; kwargs...) = RightBiasedC2F(NamedTuple(kwargs))

return_space(::RightBiasedC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::RightBiasedC2F, arg) = ((half, half),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiasedC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx + half, hidx)

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

Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiasedC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

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
end
RightBiasedF2C(; kwargs...) = RightBiasedF2C(NamedTuple(kwargs))

return_space(::RightBiasedF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::RightBiasedF2C, arg) = ((half, half),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiasedF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx + half, hidx)

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
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiasedF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end


"""
    R = RightBiased3rdOrderC2F(;boundaries)
    R.(x)

Interpolate a center-valued field to a face-valued field from the right, using a 3rd-order reconstruction.
```math
R(x)[i] = \\left(4 x[i-\\tfrac{1}{2}] + 10 x[i+\\tfrac{1}{2}] -2 x[i+\\tfrac{3}{2}]  \\right) / 12
```

Only the right boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
R(x)[n+\\tfrac{1}{2}] = x_0
```
"""
struct RightBiased3rdOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
RightBiased3rdOrderC2F(; kwargs...) = RightBiased3rdOrderC2F(NamedTuple(kwargs))

return_space(::RightBiased3rdOrderC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::RightBiased3rdOrderC2F, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiased3rdOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) =
    (
        4 * getidx(space, arg, loc, idx - half, hidx) +
        10 * getidx(space, arg, loc, idx + half, hidx) -
        2 * getidx(space, arg, loc, idx + half + 1, hidx)
    ) / 12

boundary_width(::RightBiased3rdOrderC2F, ::SetValue) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

"""
    R = RightBiased3rdOrderF2C(;boundaries)
    R.(x)

Interpolate a face-valued field to a center-valued field from the right, using a 3rd-order reconstruction.
```math
R(x)[i] = \\left(4 x[i] + 10 x[i+1] -2 x[i+2]  \\right) / 12
```

Only the right boundary condition should be set. Currently supported is:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
R(x)[n+\\tfrac{1}{2}] = x_0
```
"""
struct RightBiased3rdOrderF2C{BCS} <: InterpolationOperator
    bcs::BCS
end
RightBiased3rdOrderF2C(; kwargs...) = RightBiased3rdOrderF2C(NamedTuple(kwargs))

return_space(::RightBiased3rdOrderF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::RightBiased3rdOrderF2C, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiased3rdOrderF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
) =
    (
        4 * getidx(space, arg, loc, idx - half, hidx) +
        10 * getidx(space, arg, loc, idx + half, hidx) -
        2 * getidx(space, arg, loc, idx + half + 1, hidx)
    ) / 12

boundary_width(::RightBiased3rdOrderF2C, ::SetValue) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

abstract type WeightedInterpolationOperator <: InterpolationOperator end
# TODO: this is not in general correct and the return type
# should be based on the component operator types (rdiv, rmul) but we don't have a good way
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
struct WeightedInterpolateF2C{BCS} <: WeightedInterpolationOperator
    bcs::BCS
end
WeightedInterpolateF2C(; kwargs...) = WeightedInterpolateF2C(NamedTuple(kwargs))

return_space(
    ::WeightedInterpolateF2C,
    weight_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllFaceFiniteDifferenceSpace,
) = Spaces.space(arg_space, Spaces.CellCenter())

stencil_interior_width(::WeightedInterpolateF2C, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateF2C,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(space, weight, loc, idx + half, hidx)
    w⁻ = getidx(space, weight, loc, idx - half, hidx)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (w⁺ ⊞ w⁻))
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
- [`SetGradient`](@ref): set the value at the boundary such that the gradient is `val`.
- [`Extrapolate`](@ref): use the closest interior point as the boundary value.

These have the same stencil as in [`InterpolateC2F`](@ref).
"""
struct WeightedInterpolateC2F{BCS} <: WeightedInterpolationOperator
    bcs::BCS
end
WeightedInterpolateC2F(; kwargs...) = WeightedInterpolateC2F(NamedTuple(kwargs))

return_space(
    ::WeightedInterpolateC2F,
    weight_space::AllCenterFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = Spaces.space(arg_space, Spaces.CellFace())

stencil_interior_width(::WeightedInterpolateC2F, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateC2F,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(space, weight, loc, idx + half, hidx)
    w⁻ = getidx(space, weight, loc, idx - half, hidx)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (w⁺ ⊞ w⁻))
end

boundary_width(::WeightedInterpolateC2F, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    v₃ = Geometry.covariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ ⊟ RecursiveApply.rdiv(v₃, 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    v₃ = Geometry.covariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ ⊞ RecursiveApply.rdiv(v₃, 2)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁺
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    weight,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    a⁻
end


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
- [`SetValue(x₀)`](@ref): set the value of `x` to be `x₀` in a hypothetical
  ghost cell on the other side of the boundary. On the left boundary the stencil
  is
  ```math
  U(\\boldsymbol{v},x)[\\tfrac{1}{2}] = \\begin{cases}
    v^3[\\tfrac{1}{2}] x_0  \\boldsymbol{e}_3 \\textrm{, if }  v^3[\\tfrac{1}{2}] > 0 \\\\
    v^3[\\tfrac{1}{2}] x[1] \\boldsymbol{e}_3 \\textrm{, if }  v^3[\\tfrac{1}{2}] < 0
    \\end{cases}
  ```
- [`Extrapolate()`](@ref): set the value of `x` to be the same as the closest
  interior point. On the left boundary, the stencil is
  ```math
  U(\\boldsymbol{v},x)[\\tfrac{1}{2}] = U(\\boldsymbol{v},x)[1 + \\tfrac{1}{2}]
  ```
"""
struct UpwindBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
end
UpwindBiasedProductC2F(; kwargs...) = UpwindBiasedProductC2F(NamedTuple(kwargs))

return_eltype(::UpwindBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::UpwindBiasedProductC2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = velocity_space

function upwind_biased_product(v, a⁻, a⁺)
    RecursiveApply.rdiv(
        ((v ⊞ RecursiveApply.rmap(abs, v)) ⊠ a⁻) ⊞
        ((v ⊟ RecursiveApply.rmap(abs, v)) ⊠ a⁺),
        2,
    )
end

stencil_interior_width(::UpwindBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half, half))

Base.@propagate_inbounds function stencil_interior(
    ::UpwindBiasedProductC2F,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, space, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, space, idx, hidx, arg)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, a⁺))
end

boundary_width(::UpwindBiasedProductC2F, ::AbstractBoundaryCondition) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    aᴸᴮ = getidx(space, bc.val, loc, nothing, hidx)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, space, idx, hidx, arg)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, aᴸᴮ, a⁺))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, space, idx, hidx, arg)
    aᴿᴮ = getidx(space, bc.val, loc, nothing, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, aᴿᴮ))
end

Base.@propagate_inbounds function stencil_left_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    stencil_interior(op, loc, space, idx + 1, hidx, velocity, arg)
end

Base.@propagate_inbounds function stencil_right_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    stencil_interior(op, loc, space, idx - 1, hidx, velocity, arg)
end

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
"""
struct LinVanLeerC2F{BCS, C} <: AdvectionOperator
    bcs::BCS
    constraint::C
end
abstract type LimiterConstraint end
struct AlgebraicMean <: LimiterConstraint end
struct PositiveDefinite <: LimiterConstraint end
struct MonotoneHarmonic <: LimiterConstraint end
struct MonotoneLocalExtrema <: LimiterConstraint end

LinVanLeerC2F(; constraint, kwargs...) =
    LinVanLeerC2F(NamedTuple(kwargs), constraint)

return_eltype(::LinVanLeerC2F, V, A, dt) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::LinVanLeerC2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
    dt,
) = velocity_space

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
        return eltype(v)(0)
    end
end

posdiff(x, y) = ifelse(x - y ≥ 0, x - y, eltype(x)(0))

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
        return v ⊠ (a⁻ ⊞ RecursiveApply.rdiv(Δ𝛼, 2))
    else
        # Eqn (2,5a,5b,5c)
        Δ𝛼 = compute_Δ𝛼_linvanleer(a⁻, a⁺, a⁺⁺, v, dt, constraint)
        return v ⊠ (a⁺ ⊟ RecursiveApply.rdiv(Δ𝛼, 2))
    end
end

stencil_interior_width(::LinVanLeerC2F, velocity, arg, dt) =
    ((0, 0), (-half - 1, half + 1), (0, 0))

Base.@propagate_inbounds function stencil_interior(
    op::LinVanLeerC2F,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    a⁻⁻ = getidx(space, arg, loc, idx - half - 1, hidx)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁺⁺ = getidx(space, arg, loc, idx + half + 1, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(
        slope_limited_product(vᶠ, a⁻, a⁻⁻, a⁺, a⁺⁺, dt, op.constraint),
    )
end

boundary_width(::LinVanLeerC2F, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::LinVanLeerC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
)
    @assert idx <= left_face_boundary_idx(space) + 1
    v = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, space, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::LinVanLeerC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
)
    @assert idx >= right_face_boundary_idx(space) - 1
    v = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, space, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))

end

Base.@propagate_inbounds function stencil_left_boundary(
    op::LinVanLeerC2F,
    bc::ThirdOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
)
    @assert idx <= left_face_boundary_idx(space) + 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)

    return Geometry.Contravariant3Vector(vᶠ * a)
end

Base.@propagate_inbounds function stencil_right_boundary(
    op::LinVanLeerC2F,
    bc::ThirdOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
    dt,
)
    @assert idx <= right_face_boundary_idx(space) - 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)

    return Geometry.Contravariant3Vector(vᶠ * a)
end

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
- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind scheme to compute `x` on the left boundary,
  and the first-order upwind scheme to compute `x` on the right boundary.
- [`ThirdOrderOneSided(x₀)`](@ref): uses the third-order downwind reconstruction to compute `x` on the left boundary,
and the third-order upwind reconstruction to compute `x` on the right boundary.

!!! note
    These boundary conditions do not define the value at the actual boundary faces, and so this operator cannot be materialized directly: it needs to be composed with another operator that does not make use of this value, e.g. a [`DivergenceF2C`](@ref) operator, with a [`SetValue`](@ref) boundary.
"""
struct Upwind3rdOrderBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
end
Upwind3rdOrderBiasedProductC2F(; kwargs...) =
    Upwind3rdOrderBiasedProductC2F(NamedTuple(kwargs))

return_eltype(::Upwind3rdOrderBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::Upwind3rdOrderBiasedProductC2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = velocity_space

function upwind_3rdorder_biased_product(v, a⁻, a⁻⁻, a⁺, a⁺⁺)
    RecursiveApply.rdiv(
        (v ⊠ (7 ⊠ (a⁺ + a⁻) ⊟ (a⁺⁺ + a⁻⁻))) ⊟
        (RecursiveApply.rmap(abs, v) ⊠ (3 ⊠ (a⁺ - a⁻) ⊟ (a⁺⁺ - a⁻⁻))),
        12,
    )
end

stencil_interior_width(::Upwind3rdOrderBiasedProductC2F, velocity, arg) =
    ((0, 0), (-half - 1, half + 1))

Base.@propagate_inbounds function stencil_interior(
    ::Upwind3rdOrderBiasedProductC2F,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    a⁻⁻ = getidx(space, arg, loc, idx - half - 1, hidx)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁺⁺ = getidx(space, arg, loc, idx + half + 1, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(
        upwind_3rdorder_biased_product(vᶠ, a⁻, a⁻⁻, a⁺, a⁺⁺),
    )
end

boundary_width(::Upwind3rdOrderBiasedProductC2F, ::AbstractBoundaryCondition) =
    2

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx <= left_face_boundary_idx(space) + 1
    v = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, space, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx >= right_face_boundary_idx(space) - 1
    v = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, space, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))

end

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::ThirdOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx <= left_face_boundary_idx(space) + 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)

    return Geometry.Contravariant3Vector(vᶠ * a)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::ThirdOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx <= right_face_boundary_idx(space) - 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)

    return Geometry.Contravariant3Vector(vᶠ * a)
end

"""
    U = FCTBorisBook(;boundaries)
    U.(v, x)

Correct the flux using the flux-corrected transport formulation by Boris and Book [BorisBook1973](@cite).

Input arguments:
- a face-valued vector field `v`
- a center-valued field `x`
```math
Ac(v,x)[i] =
  s[i] \\max \\left\\{0, \\min \\left[ |v[i] |, s[i] \\left( x[i+\\tfrac{3}{2}] - x[i+\\tfrac{1}{2}]  \\right) ,  s[i] \\left( x[i-\\tfrac{1}{2}] - x[i-\\tfrac{3}{2}]  \\right) \\right] \\right\\},
```
where ``s[i] = +1`` if  `` v[i] \\geq 0`` and ``s[i] = -1`` if  `` v[i] \\leq 0``, and ``Ac`` represents the resulting corrected antidiffusive flux.
This formulation is based on [BorisBook1973](@cite), as reported in [durran2010](@cite) section 5.4.1.

Supported boundary conditions are:
- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind reconstruction to compute `x` on the left boundary, and the first-order upwind reconstruction to compute `x` on the right boundary.

!!! note
    Similar to the [`Upwind3rdOrderBiasedProductC2F`](@ref) operator, these boundary conditions do not define the value at the actual boundary faces,
    and so this operator cannot be materialized directly: it needs to be composed with another operator that does not make use of this value, e.g. a
    [`DivergenceF2C`](@ref) operator, with a [`SetValue`](@ref) boundary.
"""
struct FCTBorisBook{BCS} <: AdvectionOperator
    bcs::BCS
end
FCTBorisBook(; kwargs...) = FCTBorisBook(NamedTuple(kwargs))

return_eltype(::FCTBorisBook, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

return_space(
    ::FCTBorisBook,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = velocity_space

function fct_boris_book(v, a⁻⁻, a⁻, a⁺, a⁺⁺)
    if v != zero(eltype(v))
        sign(v) ⊠ (RecursiveApply.rmap(
            max,
            zero(eltype(v)),
            RecursiveApply.rmap(
                min,
                RecursiveApply.rmap(abs, v),
                RecursiveApply.rmap(
                    min,
                    sign(v) ⊠ (a⁺⁺ - a⁺),
                    sign(v) ⊠ (a⁻ - a⁻⁻),
                ),
            ),
        ))
    else
        RecursiveApply.rmap(
            max,
            zero(eltype(v)),
            RecursiveApply.rmap(
                min,
                v,
                RecursiveApply.rmap(min, (a⁺⁺ - a⁺), (a⁻ - a⁻⁻)),
            ),
        )
    end
end

stencil_interior_width(::FCTBorisBook, velocity, arg) =
    ((0, 0), (-half - 1, half + 1))

Base.@propagate_inbounds function stencil_interior(
    ::FCTBorisBook,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    a⁻⁻ = getidx(space, arg, loc, idx - half - 1, hidx)
    a⁻ = getidx(space, arg, loc, idx - half, hidx)
    a⁺ = getidx(space, arg, loc, idx + half, hidx)
    a⁺⁺ = getidx(space, arg, loc, idx + half + 1, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(fct_boris_book(vᶠ, a⁻⁻, a⁻, a⁺, a⁺⁺))
end

boundary_width(::FCTBorisBook, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::FCTBorisBook,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx <= left_face_boundary_idx(space) + 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(zero(eltype(vᶠ)))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::FCTBorisBook,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx <= right_face_boundary_idx(space) - 1

    vᶠ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(zero(eltype(vᶠ)))
end

"""
    U = FCTZalesak(;boundaries)
    U.(A, Φ, Φᵗᵈ)

Correct the flux using the flux-corrected transport formulation by Zalesak [zalesak1979fully](@cite).

Input arguments:
- a face-valued vector field `A`
- a center-valued field `Φ`
- a center-valued field `Φᵗᵈ`
```math
Φ_j^{n+1} = Φ_j^{td} - (C_{j+\\frac{1}{2}}A_{j+\\frac{1}{2}} - C_{j-\\frac{1}{2}}A_{j-\\frac{1}{2}})
```
This stencil is based on [zalesak1979fully](@cite), as reported in [durran2010](@cite) section 5.4.2, where ``C`` denotes
the corrected antidiffusive flux.

Supported boundary conditions are:
- [`FirstOrderOneSided(x₀)`](@ref): uses the first-order downwind reconstruction to compute `x` on the left boundary, and the first-order upwind reconstruction to compute `x` on the right boundary.

!!! note
    Similar to the [`Upwind3rdOrderBiasedProductC2F`](@ref) operator, these boundary conditions do not define
    the value at the actual boundary faces, and so this operator cannot be materialized directly: it needs to
    be composed with another operator that does not make use of this value, e.g. a [`DivergenceF2C`](@ref) operator,
    with a [`SetValue`](@ref) boundary.
"""
struct FCTZalesak{BCS} <: AdvectionOperator
    bcs::BCS
end
FCTZalesak(; kwargs...) = FCTZalesak(NamedTuple(kwargs))

return_eltype(::FCTZalesak, A, Φ, Φᵗᵈ) =
    Geometry.Contravariant3Vector{eltype(eltype(A))}

return_space(
    ::FCTZalesak,
    A_space::AllFaceFiniteDifferenceSpace,
    Φ_space::AllCenterFiniteDifferenceSpace,
    Φᵗᵈ_space::AllCenterFiniteDifferenceSpace,
) = A_space

function fct_zalesak(
    Aⱼ₋₁₂,
    Aⱼ₊₁₂,
    Aⱼ₊₃₂,
    ϕⱼ₋₁,
    ϕⱼ,
    ϕⱼ₊₁,
    ϕⱼ₊₂,
    ϕⱼ₋₁ᵗᵈ,
    ϕⱼᵗᵈ,
    ϕⱼ₊₁ᵗᵈ,
    ϕⱼ₊₂ᵗᵈ,
)
    # 1/dt is in ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₊₂, ϕⱼ₋₁ᵗᵈ, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ, ϕⱼ₊₂ᵗᵈ

    stable_zero = zero(eltype(Aⱼ₊₁₂))
    stable_one = one(eltype(Aⱼ₊₁₂))

    # 𝒮5.4.2 (1)  Durran (5.32)  Zalesak's cosmetic correction
    # which is usually omitted but used in Durran's textbook
    # implementation of the flux corrected transport method.
    # (Textbook suggests mixed results in 3 reported scenarios)
    if (
        Aⱼ₊₁₂ * (ϕⱼ₊₁ᵗᵈ - ϕⱼᵗᵈ) < stable_zero && (
            Aⱼ₊₁₂ * (ϕⱼ₊₂ᵗᵈ - ϕⱼ₊₁ᵗᵈ) < stable_zero ||
            Aⱼ₊₁₂ * (ϕⱼᵗᵈ - ϕⱼ₋₁ᵗᵈ) < stable_zero
        )
    )
        Aⱼ₊₁₂ = stable_zero
    end

    # 𝒮5.4.2 (2)
    # If flow is nondivergent, ϕᵗᵈ are not needed in the formulae below
    ϕⱼᵐᵃˣ = max(ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₋₁ᵗᵈ, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ)
    ϕⱼᵐⁱⁿ = min(ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₋₁ᵗᵈ, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ)
    Pⱼ⁺ = max(stable_zero, Aⱼ₋₁₂) - min(stable_zero, Aⱼ₊₁₂)
    # Zalesak also requires, in equation (5.33) Δx/Δt, which for the
    # reference element we may assume Δζ = 1 between interfaces
    Qⱼ⁺ = (ϕⱼᵐᵃˣ - ϕⱼᵗᵈ)
    Rⱼ⁺ = (Pⱼ⁺ > stable_zero ? min(stable_one, Qⱼ⁺ / Pⱼ⁺) : stable_zero)
    Pⱼ⁻ = max(stable_zero, Aⱼ₊₁₂) - min(stable_zero, Aⱼ₋₁₂)
    Qⱼ⁻ = (ϕⱼᵗᵈ - ϕⱼᵐⁱⁿ)
    Rⱼ⁻ = (Pⱼ⁻ > stable_zero ? min(stable_one, Qⱼ⁻ / Pⱼ⁻) : stable_zero)
    ϕⱼ₊₁ᵐᵃˣ = max(ϕⱼ, ϕⱼ₊₁, ϕⱼ₊₂, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ, ϕⱼ₊₂ᵗᵈ)
    ϕⱼ₊₁ᵐⁱⁿ = min(ϕⱼ, ϕⱼ₊₁, ϕⱼ₊₂, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ, ϕⱼ₊₂ᵗᵈ)
    Pⱼ₊₁⁺ = max(stable_zero, Aⱼ₊₁₂) - min(stable_zero, Aⱼ₊₃₂)
    Qⱼ₊₁⁺ = (ϕⱼ₊₁ᵐᵃˣ - ϕⱼ₊₁ᵗᵈ)
    Rⱼ₊₁⁺ = (Pⱼ₊₁⁺ > stable_zero ? min(stable_one, Qⱼ₊₁⁺ / Pⱼ₊₁⁺) : stable_zero)
    Pⱼ₊₁⁻ = max(stable_zero, Aⱼ₊₃₂) - min(stable_zero, Aⱼ₊₁₂)
    Qⱼ₊₁⁻ = (ϕⱼ₊₁ᵗᵈ - ϕⱼ₊₁ᵐⁱⁿ)
    Rⱼ₊₁⁻ = (Pⱼ₊₁⁻ > stable_zero ? min(stable_one, Qⱼ₊₁⁻ / Pⱼ₊₁⁻) : stable_zero)

    Cⱼ₊₁₂ = (Aⱼ₊₁₂ ≥ stable_zero ? min(Rⱼ₊₁⁺, Rⱼ⁻) : min(Rⱼ⁺, Rⱼ₊₁⁻))

    return Cⱼ₊₁₂ * Aⱼ₊₁₂
end

stencil_interior_width(::FCTZalesak, A_space, Φ_space, Φᵗᵈ_space) =
    ((-1, 1), (-half - 1, half + 1), (-half - 1, half + 1))

Base.@propagate_inbounds function stencil_interior(
    ::FCTZalesak,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    Φᵗᵈ_field,
)
    # cell center variables
    ϕⱼ₋₁ = getidx(space, Φ_field, loc, idx - half - 1, hidx)
    ϕⱼ = getidx(space, Φ_field, loc, idx - half, hidx)
    ϕⱼ₊₁ = getidx(space, Φ_field, loc, idx + half, hidx)
    ϕⱼ₊₂ = getidx(space, Φ_field, loc, idx + half + 1, hidx)
    # cell center variables
    ϕⱼ₋₁ᵗᵈ = getidx(space, Φᵗᵈ_field, loc, idx - half - 1, hidx)
    ϕⱼᵗᵈ = getidx(space, Φᵗᵈ_field, loc, idx - half, hidx)
    ϕⱼ₊₁ᵗᵈ = getidx(space, Φᵗᵈ_field, loc, idx + half, hidx)
    ϕⱼ₊₂ᵗᵈ = getidx(space, Φᵗᵈ_field, loc, idx + half + 1, hidx)
    # cell face variables
    Aⱼ₊₁₂ = Geometry.contravariant3(
        getidx(space, A_field, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    Aⱼ₋₁₂ = Geometry.contravariant3(
        getidx(space, A_field, loc, idx - 1, hidx),
        Geometry.LocalGeometry(space, idx - 1, hidx),
    )
    Aⱼ₊₃₂ = Geometry.contravariant3(
        getidx(space, A_field, loc, idx + 1, hidx),
        Geometry.LocalGeometry(space, idx + 1, hidx),
    )

    return Geometry.Contravariant3Vector(
        fct_zalesak(
            Aⱼ₋₁₂,
            Aⱼ₊₁₂,
            Aⱼ₊₃₂,
            ϕⱼ₋₁,
            ϕⱼ,
            ϕⱼ₊₁,
            ϕⱼ₊₂,
            ϕⱼ₋₁ᵗᵈ,
            ϕⱼᵗᵈ,
            ϕⱼ₊₁ᵗᵈ,
            ϕⱼ₊₂ᵗᵈ,
        ),
    )
end

boundary_width(::FCTZalesak, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::FCTZalesak,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    Φᵗᵈ_field,
)
    @assert idx <= left_face_boundary_idx(space) + 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_field))))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::FCTZalesak,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    Φᵗᵈ_field,
)
    @assert idx <= right_face_boundary_idx(space) - 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_field))))
end

"""
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

"""
abstract type AbstractTVDSlopeLimiter end


"""
    U = RZeroLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct RZeroLimiter <: AbstractTVDSlopeLimiter end

"""
    U = RHalfLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct RHalfLimiter <: AbstractTVDSlopeLimiter end

"""
    U = RMaxLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct RMaxLimiter <: AbstractTVDSlopeLimiter end

"""
    U = MinModLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct MinModLimiter <: AbstractTVDSlopeLimiter end

"""
    U = KorenLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct KorenLimiter <: AbstractTVDSlopeLimiter end

"""
    U = SuperbeeLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct SuperbeeLimiter <: AbstractTVDSlopeLimiter end

"""
    U = MonotonizedCentralLimiter(;boundaries)
    U.(𝒜, Φ, 𝓊)

A subtype of [`AbstractTVDSlopeLimiter`](@ref) limiter. See
[`AbstractTVDSlopeLimiter`](@ref) for the general formulation.
"""
struct MonotonizedCentralLimiter <: AbstractTVDSlopeLimiter end

@inline function compute_limiter_coeff(r, ::RZeroLimiter)
    return zero(eltype(r))
end

@inline function compute_limiter_coeff(r, ::RHalfLimiter)
    return one(eltype(r)) * 1 / 2
end

@inline function compute_limiter_coeff(r, ::RMaxLimiter)
    return one(eltype(r))
end

@inline function compute_limiter_coeff(r, ::MinModLimiter)
    return max(zero(eltype(r)), min(one(eltype(r)), r))
end

@inline function compute_limiter_coeff(r, ::KorenLimiter)
    return max(zero(eltype(r)), min(2r, min(1 / 3 + 2r / 3, 2)))
end

@inline function compute_limiter_coeff(r, ::SuperbeeLimiter)
    return max(zero(eltype(r)), min(one(eltype(r)), r), min(2, r))
end

@inline function compute_limiter_coeff(r, ::MonotonizedCentralLimiter)
    return max(zero(eltype(r)), min(2r, (1 + r) / 2, 2))
end

struct TVDLimitedFluxC2F{BCS, M} <: AdvectionOperator
    bcs::BCS
    method::M
end

TVDLimitedFluxC2F(; method, kwargs...) =
    TVDLimitedFluxC2F((; kwargs...), method)

return_eltype(::TVDLimitedFluxC2F, A, Φ, 𝓊) =
    Geometry.Contravariant3Vector{eltype(eltype(A))}

return_space(
    ::TVDLimitedFluxC2F,
    A_space::AllFaceFiniteDifferenceSpace,
    Φ_space::AllCenterFiniteDifferenceSpace,
    u_space::AllFaceFiniteDifferenceSpace,
) = A_space

function tvd_limited_flux(Aⱼ₋₁₂, Aⱼ₊₁₂, ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₊₂, rⱼ₊₁₂, constraint)
    stable_zero = zero(eltype(Aⱼ₊₁₂))
    stable_one = one(eltype(Aⱼ₊₁₂))
    Cⱼ₊₁₂ = compute_limiter_coeff(rⱼ₊₁₂, constraint)
    @assert Cⱼ₊₁₂ <= 2
    @assert Cⱼ₊₁₂ >= 0
    return Cⱼ₊₁₂ * Aⱼ₊₁₂
end

stencil_interior_width(::TVDLimitedFluxC2F, A_space, Φ_space, u_space) =
    ((-1, 1), (-half - 1, half + 1), (-1, +1))

Base.@propagate_inbounds function stencil_interior(
    op::TVDLimitedFluxC2F,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    𝓊_field,
)
    # cell center variables
    ϕⱼ₋₁ = getidx(space, Φ_field, loc, idx - half - 1, hidx)
    ϕⱼ = getidx(space, Φ_field, loc, idx - half, hidx)
    ϕⱼ₊₁ = getidx(space, Φ_field, loc, idx + half, hidx)
    ϕⱼ₊₂ = getidx(space, Φ_field, loc, idx + half + 1, hidx)
    𝓊 = Geometry.contravariant3(
        getidx(space, 𝓊_field, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    # cell face variables
    Aⱼ₊₁₂ = Geometry.contravariant3(
        getidx(space, A_field, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    Aⱼ₋₁₂ = Geometry.contravariant3(
        getidx(space, A_field, loc, idx - 1, hidx),
        Geometry.LocalGeometry(space, idx - 1, hidx),
    )
    # See filter options below
    rⱼ₊₁₂ = compute_slope_ratio(ϕⱼ, ϕⱼ₋₁, ϕⱼ₊₁, ϕⱼ₊₂, 𝓊)

    return Geometry.Contravariant3Vector(
        tvd_limited_flux(Aⱼ₋₁₂, Aⱼ₊₁₂, ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₊₂, rⱼ₊₁₂, op.method),
    )
end

@inline function compute_slope_ratio(ϕⱼ, ϕⱼ₋₁, ϕⱼ₊₁, ϕⱼ₊₂, 𝓊)
    if 𝓊 >= 0
        return (ϕⱼ - ϕⱼ₋₁) / (ϕⱼ₊₁ - ϕⱼ + eps(eltype(ϕⱼ)))
    else
        return (ϕⱼ₊₂ - ϕⱼ₊₁) / (ϕⱼ₊₁ - ϕⱼ + eps(eltype(ϕⱼ)))
    end
end

boundary_width(::TVDLimitedFluxC2F, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::TVDLimitedFluxC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    𝓊_field,
)
    @assert idx <= left_face_boundary_idx(space) + 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_field))))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::TVDLimitedFluxC2F,
    bc::FirstOrderOneSided,
    loc,
    space,
    idx,
    hidx,
    A_field,
    Φ_field,
    𝓊_field,
)
    @assert idx <= right_face_boundary_idx(space) - 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_field))))
end

"""
    A = AdvectionF2F(;boundaries)
    A.(v, θ)

Vertical advection operator at cell faces, for a face-valued velocity field `v` and face-valued
variables `θ`, approximating ``v^3 \\partial_3 \\theta``.

It uses the following stencil
```math
A(v,θ)[i] = \\frac{1}{2} (θ[i+1] - θ[i-1]) v³[i]
```

No boundary conditions are currently supported.
"""
struct AdvectionF2F{BCS} <: AdvectionOperator
    bcs::BCS
end
AdvectionF2F(; kwargs...) = AdvectionF2F(NamedTuple(kwargs))

return_space(
    ::AdvectionF2F,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllFaceFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::AdvectionF2F, velocity, arg) = ((0, 0), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::AdvectionF2F,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    ∂θ₃ = RecursiveApply.rdiv(θ⁺ ⊟ θ⁻, 2)
    return w³ ⊠ ∂θ₃
end
boundary_width(::AdvectionF2F, ::AbstractBoundaryCondition) = 1

"""
    A = AdvectionC2C(;boundaries)
    A.(v, θ)

Vertical advection operator at cell centers, for cell face velocity field `v` cell center
variables `θ`, approximating ``v^3 \\partial_3 \\theta``.

It uses the following stencil
```math
A(v,θ)[i] = \\frac{1}{2} \\{ (θ[i+1] - θ[i]) v³[i+\\tfrac{1}{2}] + (θ[i] - θ[i-1])v³[i-\\tfrac{1}{2}]\\}
```

Supported boundary conditions:

- [`SetValue(θ₀)`](@ref): set the value of `θ` at the boundary face to be `θ₀`.
  At the lower boundary, this is:
```math
A(v,θ)[1] = \\frac{1}{2} \\{ (θ[2] - θ[1]) v³[1 + \\tfrac{1}{2}] + (θ[1] - θ₀)v³[\\tfrac{1}{2}]\\}
```
- [`Extrapolate`](@ref): use the closest interior point as the boundary value.
  At the lower boundary, this is:
```math
A(v,θ)[1] = (θ[2] - θ[1]) v³[1 + \\tfrac{1}{2}] \\}
```
"""
struct AdvectionC2C{BCS} <: AdvectionOperator
    bcs::BCS
end
AdvectionC2C(; kwargs...) = AdvectionC2C(NamedTuple(kwargs))

return_space(
    ::AdvectionC2C,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::AdvectionC2C, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::AdvectionC2C,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

boundary_width(::AdvectionC2C, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::AdvectionC2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, bc.val, loc, nothing, hidx) # defined at face, not the center
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = 2 ⊠ (θ ⊟ θ⁻)
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::AdvectionC2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    θ⁺ = getidx(space, bc.val, loc, nothing, hidx) # value at the face
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = 2 ⊠ (θ⁺ ⊟ θ)
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (w³⁺ ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (w³⁻ ⊠ ∂θ₃⁻)
end

struct FluxCorrectionC2C{BCS} <: AdvectionOperator
    bcs::BCS
end
FluxCorrectionC2C(; kwargs...) = FluxCorrectionC2C(NamedTuple(kwargs))

return_space(
    ::FluxCorrectionC2C,
    velocity_space::AllFaceFiniteDifferenceSpace,
    arg_space::AllCenterFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::FluxCorrectionC2C, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::FluxCorrectionC2C,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (abs(w³⁺) ⊠ ∂θ₃⁺) ⊟ (abs(w³⁻) ⊠ ∂θ₃⁻)
end

boundary_width(::FluxCorrectionC2C, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::FluxCorrectionC2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (abs(w³⁺) ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::FluxCorrectionC2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁻ = θ ⊟ θ⁻
    return ⊟(abs(w³⁻) ⊠ ∂θ₃⁻)
end

struct FluxCorrectionF2F{BCS} <: AdvectionOperator
    bcs::BCS
end
FluxCorrectionF2F(; kwargs...) = FluxCorrectionF2F(NamedTuple(kwargs))

return_space(
    ::FluxCorrectionF2F,
    velocity_space::AllCenterFiniteDifferenceSpace,
    arg_space::AllFaceFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::FluxCorrectionF2F, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::FluxCorrectionF2F,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (abs(w³⁺) ⊠ ∂θ₃⁺) ⊟ (abs(w³⁻) ⊠ ∂θ₃⁻)
end

boundary_width(::FluxCorrectionF2F, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::FluxCorrectionF2F,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    θ⁺ = getidx(space, arg, loc, idx + 1, hidx)
    θ = getidx(space, arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (abs(w³⁺) ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::FluxCorrectionF2F,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    θ = getidx(space, arg, loc, idx, hidx)
    θ⁻ = getidx(space, arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁻ = θ ⊟ θ⁻
    return ⊟(abs(w³⁻) ⊠ ∂θ₃⁻)
end


abstract type BoundaryOperator <: FiniteDifferenceOperator end

"""
    SetBoundaryOperator(;boundaries...)

This operator only modifies the values at the boundary:
 - [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
"""
struct SetBoundaryOperator{BCS} <: BoundaryOperator
    bcs::BCS
end
SetBoundaryOperator(; kwargs...) = SetBoundaryOperator(NamedTuple(kwargs))

return_space(::SetBoundaryOperator, space::AllFaceFiniteDifferenceSpace) = space

stencil_interior_width(::SetBoundaryOperator, arg) = ((0, 0),)
Base.@propagate_inbounds stencil_interior(
    ::SetBoundaryOperator,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx, hidx)

boundary_width(::SetBoundaryOperator, ::AbstractBoundaryCondition) = 0
boundary_width(::SetBoundaryOperator, ::SetValue) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
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
end
GradientF2C(; kwargs...) = GradientF2C(NamedTuple(kwargs))

return_space(::GradientF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::GradientF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::GradientF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Geometry.Covariant3Vector(1) ⊗ (
        getidx(space, arg, loc, idx + half, hidx) ⊟
        getidx(space, arg, loc, idx - half, hidx)
    )
end

boundary_width(::GradientF2C, ::AbstractBoundaryCondition) = 0

boundary_width(::GradientF2C, ::SetValue) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    Geometry.Covariant3Vector(1) ⊗ (
        getidx(space, arg, loc, idx + half, hidx) ⊟
        getidx(space, bc.val, loc, nothing, hidx)
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    Geometry.Covariant3Vector(1) ⊗ (
        getidx(space, bc.val, loc, nothing, hidx) ⊟
        getidx(space, arg, loc, idx - half, hidx)
    )
end

boundary_width(::GradientF2C, ::Extrapolate) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    op::GradientF2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    Geometry.project(
        Geometry.Covariant3Axis(),
        stencil_interior(op, loc, space, idx + 1, hidx, arg),
        Geometry.LocalGeometry(space, idx, hidx),
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    op::GradientF2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    Geometry.project(
        Geometry.Covariant3Axis(),
        stencil_interior(op, loc, space, idx - 1, hidx, arg),
        Geometry.LocalGeometry(space, idx, hidx),
    )
end

"""
    G = GradientC2F(;boundaryname=boundarycondition...)
    G.(x)

Compute the gradient of a center-valued field `x`, returning a face-valued
`Covariant3` vector field, using the stencil:
```math
G(x)[i]^3 = x[i+\\tfrac{1}{2}] - x[i-\\tfrac{1}{2}]
```

The following boundary conditions are supported:
- [`SetValue(x₀)`](@ref): calculate the gradient assuming the value at the
  boundary is `x₀`. For the left boundary, this becomes:
  ```math
  G(x)[\\tfrac{1}{2}]³ = 2 (x[1] - x₀)
  ```
- [`SetGradient(v₀)`](@ref): set the value of the gradient at the boundary to be
  `v₀`. For the left boundary, this becomes:
  ```math
  G(x)[\\tfrac{1}{2}] = v₀
  ```
"""
struct GradientC2F{BC} <: GradientOperator
    bcs::BC
end
GradientC2F(; kwargs...) = GradientC2F(NamedTuple(kwargs))

return_space(::GradientC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::GradientC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::GradientC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    Geometry.Covariant3Vector(1) ⊗ (
        getidx(space, arg, loc, idx + half, hidx) ⊟
        getidx(space, arg, loc, idx - half, hidx)
    )
end

boundary_width(::GradientC2F, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    # ∂x[i] = 2(∂x[i + half] - val)
    Geometry.Covariant3Vector(2) ⊗ (
        getidx(space, arg, loc, idx + half, hidx) ⊟
        getidx(space, bc.val, loc, nothing, hidx)
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    Geometry.Covariant3Vector(2) ⊗ (
        getidx(space, bc.val, loc, nothing, idx) ⊟
        getidx(space, arg, loc, idx - half, hidx)
    )
end


# left / right SetGradient boundary conditions
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    # imposed flux boundary condition at left most face
    Geometry.project(
        Geometry.Covariant3Axis(),
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientC2F,
    bc::SetGradient,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    # imposed flux boundary condition at right most face
    Geometry.project(
        Geometry.Covariant3Axis(),
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
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
```
"""
struct DivergenceF2C{BCS} <: DivergenceOperator
    bcs::BCS
end
DivergenceF2C(; kwargs...) = DivergenceF2C(NamedTuple(kwargs))

return_space(::DivergenceF2C, space::AllFaceFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellCenter())

stencil_interior_width(::DivergenceF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::DivergenceF2C,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceF2C, ::AbstractBoundaryCondition) = 0
boundary_width(::DivergenceF2C, ::SetValue) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceF2C,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceF2C, ::SetDivergence) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceF2C,
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceF2C,
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    getidx(space, bc.val, loc, nothing, hidx)
end

boundary_width(::DivergenceF2C, ::Extrapolate) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    op::DivergenceF2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(space)
    stencil_interior(op, loc, space, idx + 1, hidx, arg)
end
Base.@propagate_inbounds function stencil_right_boundary(
    op::DivergenceF2C,
    ::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(space)
    stencil_interior(op, loc, space, idx - 1, hidx, arg)
end

"""
    unionall_type(::Type{T})

Extract the type of the input, and strip it of any type parameters.
"""
unionall_type(::Type{T}) where {T} = T.name.wrapper

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
    unionall_type(typeof(op))(adapt_bcs(to, bcs))

@inline adapt_bcs(to, bcs) = NamedTuple{keys(bcs)}(
    UnrolledFunctions.unrolled_map(
        bc -> Adapt.adapt_structure(to, bc),
        values(bcs),
    ),
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
- [`SetValue(v₀)`](@ref): calculate the divergence assuming the value at the
   boundary is `v₀`. For the left boundary, this becomes:
  ```math
  D(v)[\\tfrac{1}{2}] = \\frac{1}{2} (Jv³[1] - Jv³₀) / J[i]
  ```
- [`SetDivergence(x)`](@ref): set the value of the divergence at the boundary to be `x`.
  ```math
  D(v)[\\tfrac{1}{2}] = x
  ```
"""
struct DivergenceC2F{BC} <: DivergenceOperator
    bcs::BC
end
DivergenceC2F(; kwargs...) = DivergenceC2F(NamedTuple(kwargs))

return_space(::DivergenceC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

stencil_interior_width(::DivergenceC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::DivergenceC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceC2F, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    # ∂x[i] = 2(∂x[i + half] - val)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³ = Geometry.Jcontravariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        local_geometry,
    )
    (Ju³₊ ⊟ Ju³) ⊠ (2 * local_geometry.invJ)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³ = Geometry.Jcontravariant3(
        getidx(space, bc.val, loc, nothing, hidx),
        local_geometry,
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(space, arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³ ⊟ Ju³₋) ⊠ (2 * local_geometry.invJ)
end

# left / right SetDivergence boundary conditions
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceC2F,
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    # imposed flux boundary condition at left most face
    getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceC2F,
    bc::SetDivergence,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    # imposed flux boundary condition at right most face
    getidx(space, bc.val, loc, nothing, hidx)
end


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

- [`SetValue(v₀)`](@ref): calculate the curl assuming the value of ``v`` at the
   boundary is `v₀`. For the left boundary, this becomes:
  ```math
  C(v)[\\tfrac{1}{2}]^1 = -\\frac{2}{J[i]} (v_2[1] - (v₀)_2)
  C(v)[\\tfrac{1}{2}]^2 = \\frac{2}{J[i]} (v_1[1] - (v₀)_1)
  ```
- [`SetCurl(v⁰)`](@ref): enforce the curl operator output at the boundary to be
  the contravariant vector `v⁰`.
"""
struct CurlC2F{BC} <: CurlFiniteDifferenceOperator
    bcs::BC
end
CurlC2F(; kwargs...) = CurlC2F(NamedTuple(kwargs))

return_space(::CurlC2F, space::AllCenterFiniteDifferenceSpace) =
    Spaces.space(space, Spaces.CellFace())

fd3_curl(u₊::Geometry.Covariant1Vector, u₋::Geometry.Covariant1Vector, invJ) =
    Geometry.Contravariant2Vector((u₊.u₁ - u₋.u₁) * invJ)
fd3_curl(u₊::Geometry.Covariant2Vector, u₋::Geometry.Covariant2Vector, invJ) =
    Geometry.Contravariant1Vector(-(u₊.u₂ - u₋.u₂) * invJ)
fd3_curl(::Geometry.Covariant3Vector, ::Geometry.Covariant3Vector, invJ) =
    Geometry.Contravariant3Vector(zero(eltype(invJ)))
fd3_curl(u₊::Geometry.Covariant12Vector, u₋::Geometry.Covariant12Vector, invJ) =
    Geometry.Contravariant12Vector(
        -(u₊.u₂ - u₋.u₂) * invJ,
        (u₊.u₁ - u₋.u₁) * invJ,
    )

stencil_interior_width(::CurlC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::CurlC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u₊ = getidx(space, arg, loc, idx + half, hidx)
    u₋ = getidx(space, arg, loc, idx - half, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u₊, u₋, local_geometry.invJ)
end

boundary_width(::CurlC2F, ::AbstractBoundaryCondition) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::CurlC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u₊ = getidx(space, arg, loc, idx + half, hidx)
    u = getidx(space, bc.val, loc, nothing, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u₊, u, local_geometry.invJ * 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::CurlC2F,
    bc::SetValue,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    u = getidx(space, bc.val, loc, nothing, hidx)
    u₋ = getidx(space, arg, loc, idx - half, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u, u₋, local_geometry.invJ * 2)
end

Base.@propagate_inbounds function stencil_left_boundary(
    ::CurlC2F,
    bc::SetCurl,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    return getidx(space, bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::CurlC2F,
    bc::SetCurl,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    return getidx(space, bc.val, loc, nothing, hidx)
end


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
right_idx(space) + boundary_width(op, bc)
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


@inline _left_interior_window_idx_args(args::Tuple, space, loc) = (
    left_interior_window_idx(args[1], space, loc),
    _left_interior_window_idx_args(Base.tail(args), space, loc)...,
)
@inline _left_interior_window_idx_args(args::Tuple{Any}, space, loc) =
    (left_interior_window_idx(args[1], space, loc),)
@inline _left_interior_window_idx_args(args::Tuple{}, space, loc) = ()

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

@inline _right_interior_window_idx_args(args::Tuple, space, loc) = (
    right_interior_window_idx(args[1], space, loc),
    _right_interior_window_idx_args(Base.tail(args), space, loc)...,
)
@inline _right_interior_window_idx_args(args::Tuple{Any}, space, loc) =
    (right_interior_window_idx(args[1], space, loc),)
@inline _right_interior_window_idx_args(args::Tuple{}, space, loc) = ()

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


Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted,
    loc::Interior,
    idx,
    hidx,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    stencil_interior(bc.op, loc, space, idx, hidx, bc.args...)
end

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted,
    loc::LeftBoundaryWindow,
    idx,
    hidx,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    op = bc.op
    if has_boundary(op, loc) &&
       idx <
       left_interior_idx(space, bc.op, get_boundary(bc.op, loc), bc.args...)
        stencil_left_boundary(
            op,
            get_boundary(op, loc),
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        stencil_interior(op, loc, space, idx, hidx, bc.args...)
    end
end

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::StencilBroadcasted,
    loc::RightBoundaryWindow,
    idx,
    hidx,
)
    op = bc.op
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    if has_boundary(op, loc) &&
       idx >
       right_interior_idx(space, bc.op, get_boundary(bc.op, loc), bc.args...)
        stencil_right_boundary(
            op,
            get_boundary(op, loc),
            loc,
            space,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        stencil_interior(op, loc, space, idx, hidx, bc.args...)
    end
end

# broadcasting a StencilStyle gives a CompositeStencilStyle
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
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        v = mod1(v, Spaces.nlevels(space))
    end
    return v
end
function vidx(space::AllCenterFiniteDifferenceSpace, idx)
    @assert idx isa Integer
    v = idx
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        v = mod1(v, Spaces.nlevels(space))
    end
    return v
end
function vidx(space::AbstractSpace, idx)
    return 1
end

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::Fields.Field,
    ::Location,
    idx,
)
    field_data = Fields.field_values(bc)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    v = vidx(space, idx)
    return @inbounds field_data[vindex(v)]
end
Base.@propagate_inbounds function getidx(
    parent_space,
    bc::Fields.Field,
    ::Location,
    idx,
    hidx,
)
    field_data = Fields.field_values(bc)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    v = vidx(space, idx)
    i, j, h = hidx
    return @inbounds field_data[CartesianIndex(i, j, 1, v, h)]
end


# unwap boxed scalars
@inline getidx(
    parent_space,
    scalar::Tuple{T},
    loc::Location,
    idx,
    hidx,
) where {T} = scalar[1]
@inline getidx(parent_space, scalar::Ref, loc::Location, idx, hidx) = scalar[]
@inline getidx(
    parent_space,
    field::Fields.PointField,
    loc::Location,
    idx,
    hidx,
) = field[]
@inline getidx(parent_space, field::Fields.PointField, loc::Location, idx) =
    field[]

# recursive fallback for scalar, just return
@inline getidx(parent_space, scalar, ::Location, idx, hidx) = scalar

# getidx error fallbacks
@noinline inferred_getidx_error(idx_type::Type, space_type::Type) =
    error("Invalid index type `$idx_type` for field on space `$space_type`")


# recursively unwrap getidx broadcast arguments in a way that is statically reducible by the optimizer
Base.@propagate_inbounds getidx_args(
    space,
    args::Tuple,
    loc::Location,
    idx,
    hidx,
) = (
    getidx(space, args[1], loc, idx, hidx),
    getidx_args(space, Base.tail(args), loc, idx, hidx)...,
)
Base.@propagate_inbounds getidx_args(
    space,
    arg::Tuple{Any},
    loc::Location,
    idx,
    hidx,
) = (getidx(space, arg[1], loc, idx, hidx),)
Base.@propagate_inbounds getidx_args(
    space,
    ::Tuple{},
    loc::Location,
    idx,
    hidx,
) = ()

Base.@propagate_inbounds function getidx(
    parent_space,
    bc::Base.Broadcast.Broadcasted,
    loc::Location,
    idx,
    hidx,
)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    _args = getidx_args(space, bc.args, loc, idx, hidx)
    bc.f(_args...)
end

if hasfield(Method, :recursion_relation)
    dont_limit = (args...) -> true
    for m in methods(getidx_args)
        m.recursion_relation = dont_limit
    end
    for m in methods(getidx)
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
    @inbounds field_data[CartesianIndex(i, j, 1, v, h)] = val
    val
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    args′ = map(Base.Broadcast.broadcastable, args)
    style = Base.Broadcast.result_style(
        StencilStyle(),
        Base.Broadcast.combine_styles(args′...),
    )
    Base.Broadcast.broadcasted(style, op, args′...)
end

function Base.Broadcast.broadcasted(
    ::Style,
    op::FiniteDifferenceOperator,
    args...,
) where {Style <: AbstractStencilStyle}
    StencilBroadcasted{Style}(op, args)
end

allow_mismatched_fd_spaces() = false

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
    if result_space !== dest_space && !allow_mismatched_fd_spaces()
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
    unionall_type(typeof(op))(column_args(op.bcs, inds...))
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

function Base.similar(
    bc::Base.Broadcast.Broadcasted{S},
    ::Type{Eltype},
) where {Eltype, S <: AbstractStencilStyle}
    sp = axes(bc)
    return Field(Eltype, sp)
end

function _serial_copyto!(field_out::Field, bc, Ni::Int, Nj::Int, Nh::Int)
    space = axes(field_out)
    bounds = window_bounds(space, bc)
    bcs = bc # strip_space(bc, space)
    @inbounds for h in 1:Nh, j in 1:Nj, i in 1:Ni
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
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for j in 1:Nj, i in 1:Ni
                apply_stencil!(space, field_out, bcs, (i, j, h), bounds)
            end
        end
    end
    call_post_op_callback() &&
        post_op_callback(field_out, field_out, bc, Ni, Nj, Nh)
    return field_out
end

function strip_space(bc::StencilBroadcasted{Style}, parent_space) where {Style}
    current_space = axes(bc)
    new_space = placeholder_space(current_space, parent_space)
    return StencilBroadcasted{Style}(
        bc.op,
        strip_space_args(bc.args, current_space),
        new_space,
    )
end

function Base.copyto!(
    field_out::Field,
    bc::Union{
        StencilBroadcasted{ColumnStencilStyle},
        Broadcasted{ColumnStencilStyle},
    },
)
    space = axes(bc)
    local_geometry = Spaces.local_geometry_data(space)
    (Ni, Nj, _, _, Nh) = size(local_geometry)
    context = ClimaComms.context(axes(field_out))
    device = ClimaComms.device(context)
    if (device isa ClimaComms.CPUMultiThreaded) && Nh > 1
        return _threaded_copyto!(field_out, bc, Ni, Nj, Nh)
    end
    return _serial_copyto!(field_out, bc, Ni, Nj, Nh)
end

function window_bounds(space, bc)
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        li = lw = left_idx(space)
        ri = rw = right_idx(space)
    else
        lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
        rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
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
    if !Topologies.isperiodic(Spaces.vertical_topology(space))
        # left window
        lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
        @inbounds for idx in li:(lw - 1)
            setidx!(
                space,
                field_out,
                idx,
                hidx,
                getidx(space, bc, lbw, idx, hidx),
            )
        end
    end
    # interior
    @inbounds for idx in lw:rw
        setidx!(
            space,
            field_out,
            idx,
            hidx,
            getidx(space, bc, Interior(), idx, hidx),
        )
    end
    if !Topologies.isperiodic(Spaces.vertical_topology(space))
        # right window
        rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
        @inbounds for idx in (rw + 1):ri
            setidx!(
                space,
                field_out,
                idx,
                hidx,
                getidx(space, bc, rbw, idx, hidx),
            )
        end
    end
    return field_out
end
# Compute slope ratio 𝜃 and limiter coefficient 𝜙
#𝜃 = compute_slope_ratio(a⁻, a⁻⁻, a⁺, a⁺⁺, v)
#𝜙 = compute_limiter_coeff(𝜃, method)


#@assert 0 <= 𝜙 <= 2
#if v >= 0
#    return v ⊠ (a⁻ ⊞ RecursiveApply.rdiv((a⁺ - a⁻) ⊠ 𝜙 ,2))
#else
#    return v ⊠ (a⁺ ⊟ RecursiveApply.rdiv((a⁺ - a⁻) ⊠ 𝜙 ,2)) # Current working solution
#end
