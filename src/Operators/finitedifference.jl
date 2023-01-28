# The geometry of the staggerd grid used by FiniteDifference in the vertical (sometimes called the C-grid)
# is (in one dimension) shown below

#         face   cell   face   cell   face
# left                                        right
#                 i-1            i
#                  ↓             ↓
#           |      ×      |      ×      |
#           ↑             ↑             ↑
#          i-1            i            i+1

# Gradient:
#  scalar => d₃ = D₃ x = Covariant3Vector(x[i+half] - x[i-half])
# ∂ξ³∂x * d₃

# Divergence: 1/J * D₃ (Jv³):
#  Jv³ = J * contravariant3(v)
#  div = 1/J * (Jv³[i+half] - Jv³[i-half])



#====

interpolate, multiply, gradient

  --->
ca  fa  fb  cb
    1 x F
  x       \
1           1    Boundary indices
  \       /
    2 - 2        --------------
  /       \
2           2    Interior indices
  \       /
    3 - 3
  /       \l
3           3

gradient with dirichlet

ca  fa
D - 1   modified stencil
  /
1
  \
    2   regular stencil
  /
2
  \
    3
  /
3


c2c laplacian, dirichlet

D - 1
  /   \
1       1
  \   /
    2   ---------------------
  /   \
2       2
  \   /
    3
  /   \
3

c2c laplacian, neumann
c   f   c

    N
      \
1       1
  \   /
    2  ----------------------
  /   \
2       2
  \   /
    n
  /   \
n       n
      /
   n+1
    =N


f2f laplacian, dirichlet
LaplacianF2F(left=Dirichlet(D))(x) = GradientC2F(left=Neumann(0))(GradientF2C(left=Dirichlet(D))(x))


dθ/dt = - ∇^2 θ
  --->
fa  ca  fb
D       _   effectively ignored: -dD/dt since is set by equations: use 2-point, 1-sided stencil?
  \
    1       boundary window
  /   \
2       2
  \   /
    2     ---------------
  /   \
3       3   interior
  \   /
    3

interior_indices1 = 2:...
interior_indices2 = 3:...


f2f laplacian, neumann
LaplacianF2F(left=Neumann(N))(x) = GradientC2F(left=Dirichlet(N))(GradientF2C(left=nothing)(x))

  --->
fa  ca  fb
1   N - 1   set by 1-sided stencil
  \   /
    1       boundary window
  /   \     ---------------
2       2
  \   /
    2
  /   \
3       3   interior
  \   /
    3

interior_indices1 = 1:n
interior_indices2 = 2:n-1
===#

import ..Utilities: PlusHalf, half

left_idx(
    space::Union{
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterExtrudedFiniteDifferenceSpace,
    },
) = left_center_boundary_idx(space)
right_idx(
    space::Union{
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterExtrudedFiniteDifferenceSpace,
    },
) = right_center_boundary_idx(space)
left_idx(
    space::Union{
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceExtrudedFiniteDifferenceSpace,
    },
) = left_face_boundary_idx(space)
right_idx(
    space::Union{
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceExtrudedFiniteDifferenceSpace,
    },
) = right_face_boundary_idx(space)

left_center_boundary_idx(space::Spaces.AbstractSpace) = 1
right_center_boundary_idx(space::Spaces.AbstractSpace) =
    size(space.center_local_geometry, 4)
left_face_boundary_idx(space::Spaces.AbstractSpace) = half
right_face_boundary_idx(space::Spaces.AbstractSpace) =
    size(space.face_local_geometry, 4) - half


left_face_boundary_idx(arg) = left_face_boundary_idx(axes(arg))
right_face_boundary_idx(arg) = right_face_boundary_idx(axes(arg))
left_center_boundary_idx(arg) = left_center_boundary_idx(axes(arg))
right_center_boundary_idx(arg) = right_center_boundary_idx(axes(arg))

Base.@propagate_inbounds function Geometry.LocalGeometry(
    space::Union{
        Spaces.FiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    idx::Integer,
    hidx,
)
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        idx = mod1(idx, length(space))
    end
    @inbounds column(space.center_local_geometry, hidx...)[idx]
end
Base.@propagate_inbounds function Geometry.LocalGeometry(
    space::Union{
        Spaces.FiniteDifferenceSpace,
        Spaces.ExtrudedFiniteDifferenceSpace,
    },
    idx::PlusHalf,
    hidx,
)
    i = idx.i + 1
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        i = mod1(i, length(space))
    end
    @inbounds column(space.face_local_geometry, hidx...)[i]
end


"""
    BoundaryCondition

An abstract type for boundary conditions for [`FiniteDifferenceOperator`](@ref)s.

Subtypes should define:
- [`boundary_width`](@ref)
- [`stencil_left_boundary`](@ref)
- [`stencil_right_boundary`](@ref)
"""
abstract type BoundaryCondition end

"""
    SetValue(val)

Set the value at the boundary to be `val`. In the case of gradient operators,
this will set the input value from which the gradient is computed.
"""
struct SetValue{S} <: BoundaryCondition
    val::S
end

"""
    SetGradient(val)

Set the gradient at the boundary to be `val`. In the case of gradient operators
this will set the output value of the gradient.
"""
struct SetGradient{S} <: BoundaryCondition
    val::S
end

"""
    SetDivergence(val)

Set the divergence at the boundary to be `val`.
"""
struct SetDivergence{S} <: BoundaryCondition
    val::S
end

"""
    SetCurl(val)

Set the curl at the boundary to be `val`.
"""
struct SetCurl{S} <: BoundaryCondition
    val::S
end

"""
    Extrapolate()

Set the value at the boundary to be the same as the closest interior point.
"""
struct Extrapolate <: BoundaryCondition end

"""
    FirstOrderOneSided()

Use a first-order up/down-wind scheme to compute the value at the boundary.
"""
struct FirstOrderOneSided <: BoundaryCondition end

"""
    ThirdOrderOneSided()

Use a third-order up/down-wind scheme to compute the value at the boundary.
"""
struct ThirdOrderOneSided <: BoundaryCondition end

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

See also [`BoundaryCondition`](@ref) for how to define the boundaries.
"""
abstract type FiniteDifferenceOperator end

return_eltype(::FiniteDifferenceOperator, arg) = eltype(arg)

# boundary width error fallback
@noinline invalid_boundary_condition_error(op_type::Type, bc_type::Type) =
    error("Boundary `$bc_type` is not supported for operator `$op_type`")

boundary_width(op::FiniteDifferenceOperator, bc::BoundaryCondition, args...) =
    invalid_boundary_condition_error(typeof(op), typeof(bc))

get_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} = getproperty(op.bcs, name)

get_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} = getproperty(op.bcs, name)

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

# the .f field is a function, but one of the args (or their args) is a StencilStyle
struct CompositeStencilStyle <: AbstractStencilStyle end


"""
    return_eltype(::Op, fields...)

Defines the element type of the result of operator `Op`
"""
function return_eltype end

"""
    return_space(::Op, spaces...)

Defines the space the operator `Op` returns values on.
"""
function return_space end

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
    stencil_interior(::Op, loc, idx, args...)

Defines the stencil of the operator `Op` in the interior of the domain at `idx`;
`args` are the input arguments.
"""
function stencil_interior end


"""
    boundary_width(::Op, ::BC, args...)

Defines the width of a boundary condition `BC` on an operator `Op`. This is the
number of locations that are used in a modified stencil.
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

return_space(::InterpolateF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(
    ::InterpolateF2C,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::InterpolateF2C,
    loc,
    idx,
    hidx,
    arg,
)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
end

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

return_space(::InterpolateC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::InterpolateC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::InterpolateC2F,
    loc,
    idx,
    hidx,
    arg,
)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
end

boundary_width(::InterpolateC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
end

boundary_width(::InterpolateC2F, ::SetGradient, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    v₃ = Geometry.covariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ ⊟ RecursiveApply.rdiv(v₃, 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    v₃ = Geometry.covariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ ⊞ RecursiveApply.rdiv(v₃, 2)
end

boundary_width(::InterpolateC2F, ::Extrapolate, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁺
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    a⁻ = getidx(arg, loc, idx - half, hidx)
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

return_space(::LeftBiasedC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::LeftBiasedC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::LeftBiasedC2F, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiasedC2F,
    loc,
    idx,
    hidx,
    arg,
) = getidx(arg, loc, idx - half, hidx)

boundary_width(::LeftBiasedC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiasedC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::LeftBiasedF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(::LeftBiasedF2C, space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::LeftBiasedF2C, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiasedF2C,
    loc,
    idx,
    hidx,
    arg,
) = getidx(arg, loc, idx - half, hidx)

boundary_width(::LeftBiasedF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiasedF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(
    ::LeftBiased3rdOrderC2F,
    space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::LeftBiased3rdOrderC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::LeftBiased3rdOrderC2F, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiased3rdOrderC2F,
    loc,
    idx,
    hidx,
    arg,
) =
    (
        -2 * getidx(arg, loc, idx - 1 - half, hidx) +
        10 * getidx(arg, loc, idx - half, hidx) +
        4 * getidx(arg, loc, idx + half, hidx)
    ) / 12

boundary_width(::LeftBiased3rdOrderC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased3rdOrderC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::LeftBiased3rdOrderF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(
    ::LeftBiased3rdOrderF2C,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::LeftBiased3rdOrderF2C, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiased3rdOrderF2C,
    loc,
    idx,
    hidx,
    arg,
) =
    (
        -2 * getidx(arg, loc, idx - 1 - half, hidx) +
        10 * getidx(arg, loc, idx - half, hidx) +
        4 * getidx(arg, loc, idx + half, hidx)
    ) / 12

boundary_width(::LeftBiased3rdOrderF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased3rdOrderF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::RightBiasedC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::RightBiasedC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::RightBiasedC2F, arg) = ((half, half),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiasedC2F,
    loc,
    idx,
    hidx,
    arg,
) = getidx(arg, loc, idx + half, hidx)

boundary_width(::RightBiasedC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiasedC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::RightBiasedF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(
    ::RightBiasedF2C,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::RightBiasedF2C, arg) = ((half, half),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiasedF2C,
    loc,
    idx,
    hidx,
    arg,
) = getidx(arg, loc, idx + half, hidx)

boundary_width(::RightBiasedF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiasedF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(
    ::RightBiased3rdOrderC2F,
    space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::RightBiased3rdOrderC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::RightBiased3rdOrderC2F, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiased3rdOrderC2F,
    loc,
    idx,
    hidx,
    arg,
) =
    (
        4 * getidx(arg, loc, idx - half, hidx) +
        10 * getidx(arg, loc, idx + half, hidx) -
        2 * getidx(arg, loc, idx + half + 1, hidx)
    ) / 12

boundary_width(::RightBiased3rdOrderC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(
    ::RightBiased3rdOrderF2C,
    space::Spaces.FaceFiniteDifferenceSpace,
) = Spaces.CenterFiniteDifferenceSpace(space)
return_space(
    ::RightBiased3rdOrderF2C,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::RightBiased3rdOrderF2C, arg) = ((-half - 1, half + 1),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiased3rdOrderF2C,
    loc,
    idx,
    hidx,
    arg,
) =
    (
        4 * getidx(arg, loc, idx - half, hidx) +
        10 * getidx(arg, loc, idx + half, hidx) -
        2 * getidx(arg, loc, idx + half + 1, hidx)
    ) / 12

boundary_width(::RightBiased3rdOrderF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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
    weight_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.FaceFiniteDifferenceSpace,
) = Spaces.CenterFiniteDifferenceSpace(arg_space)
return_space(
    ::WeightedInterpolateF2C,
    weight_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(arg_space)

stencil_interior_width(::WeightedInterpolateF2C, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateF2C,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(weight, loc, idx + half, hidx)
    w⁻ = getidx(weight, loc, idx - half, hidx)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (w⁺ ⊞ w⁻))
end

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
    weight_space::Spaces.CenterFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(arg_space)
return_space(
    ::WeightedInterpolateC2F,
    weight_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(arg_space)

stencil_interior_width(::WeightedInterpolateC2F, weight, arg) =
    ((-half, half), (-half, half))
Base.@propagate_inbounds function stencil_interior(
    ::WeightedInterpolateC2F,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    w⁺ = getidx(weight, loc, idx + half, hidx)
    w⁻ = getidx(weight, loc, idx - half, hidx)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (w⁺ ⊞ w⁻))
end

boundary_width(::WeightedInterpolateC2F, ::SetValue, weight, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    getidx(bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    getidx(bc.val, loc, nothing, hidx)
end

boundary_width(::WeightedInterpolateC2F, ::SetGradient, weight, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    v₃ = Geometry.covariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ ⊟ RecursiveApply.rdiv(v₃, 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    v₃ = Geometry.covariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ ⊞ RecursiveApply.rdiv(v₃, 2)
end

boundary_width(::WeightedInterpolateC2F, ::Extrapolate, weight, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁺
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    hidx,
    weight,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(arg, loc, idx - half, hidx)
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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = velocity_space
return_space(
    ::UpwindBiasedProductC2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
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
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, idx, hidx, arg)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, a⁺))
end

boundary_width(::UpwindBiasedProductC2F, ::SetValue, velocity, arg) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    aᴸᴮ = getidx(bc.val, loc, nothing, hidx)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, idx, hidx, arg)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, aᴸᴮ, a⁺))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, idx, hidx, arg)
    aᴿᴮ = getidx(bc.val, loc, nothing, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, aᴿᴮ))
end

boundary_width(::UpwindBiasedProductC2F, ::Extrapolate, velocity, arg) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    stencil_interior(op, loc, idx + 1, hidx, velocity, arg)
end

Base.@propagate_inbounds function stencil_right_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    stencil_interior(op, loc, idx - 1, hidx, velocity, arg)
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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = velocity_space
return_space(
    ::Upwind3rdOrderBiasedProductC2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
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
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    a⁻⁻ = getidx(arg, loc, idx - half - 1, hidx)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁺⁺ = getidx(arg, loc, idx + half + 1, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(
        upwind_3rdorder_biased_product(vᶠ, a⁻, a⁻⁻, a⁺, a⁺⁺),
    )
end

boundary_width(
    ::Upwind3rdOrderBiasedProductC2F,
    ::FirstOrderOneSided,
    velocity,
    arg,
) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx <= left_face_boundary_idx(space) + 1
    v = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiased3rdOrderC2F(), loc, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx >= right_face_boundary_idx(space) - 1
    v = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁻ = stencil_interior(LeftBiased3rdOrderC2F(), loc, idx, hidx, arg)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, idx, hidx, arg)
    return Geometry.Contravariant3Vector(upwind_biased_product(v, a⁻, a⁺))

end

boundary_width(
    ::Upwind3rdOrderBiasedProductC2F,
    ::ThirdOrderOneSided,
    velocity,
    arg,
) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::ThirdOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx <= left_face_boundary_idx(space) + 1

    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(RightBiased3rdOrderC2F(), loc, idx, hidx, arg)

    return Geometry.Contravariant3Vector(vᶠ * a)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::ThirdOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx <= right_face_boundary_idx(space) - 1

    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a = stencil_interior(LeftBiased3rdOrderC2F(), loc, idx, hidx, arg)

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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = velocity_space
return_space(
    ::FCTBorisBook,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
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
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    a⁻⁻ = getidx(arg, loc, idx - half - 1, hidx)
    a⁻ = getidx(arg, loc, idx - half, hidx)
    a⁺ = getidx(arg, loc, idx + half, hidx)
    a⁺⁺ = getidx(arg, loc, idx + half + 1, hidx)
    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(fct_boris_book(vᶠ, a⁻⁻, a⁻, a⁺, a⁺⁺))
end

boundary_width(::FCTBorisBook, ::FirstOrderOneSided, velocity, arg) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::FCTBorisBook,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx <= left_face_boundary_idx(space) + 1

    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(zero(eltype(vᶠ)))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::FCTBorisBook,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx <= right_face_boundary_idx(space) - 1

    vᶠ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    return Geometry.Contravariant3Vector(zero(eltype(vᶠ)))
end



#########################
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
    A_space::Spaces.FaceFiniteDifferenceSpace,
    Φ_space::Spaces.CenterFiniteDifferenceSpace,
    Φᵗᵈ_space::Spaces.CenterFiniteDifferenceSpace,
) = A_space
return_space(
    ::FCTZalesak,
    A_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    Φ_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    Φᵗᵈ_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
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

    if (
        Aⱼ₊₁₂ * (ϕⱼ₊₁ᵗᵈ - ϕⱼᵗᵈ) < stable_zero ||
        Aⱼ₊₁₂ * (ϕⱼ₊₂ᵗᵈ - ϕⱼ₊₁ᵗᵈ) < stable_zero ||
        Aⱼ₊₁₂ * (ϕⱼᵗᵈ - ϕⱼ₋₁ᵗᵈ) < stable_zero
    )
        Aⱼ₊₁₂ = stable_zero
    end
    ϕⱼᵐᵃˣ = max(ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₋₁ᵗᵈ, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ)
    ϕⱼᵐⁱⁿ = min(ϕⱼ₋₁, ϕⱼ, ϕⱼ₊₁, ϕⱼ₋₁ᵗᵈ, ϕⱼᵗᵈ, ϕⱼ₊₁ᵗᵈ)
    Pⱼ⁺ = max(stable_zero, Aⱼ₋₁₂) - min(stable_zero, Aⱼ₊₁₂)
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
    idx,
    hidx,
    A_space,
    Φ_space,
    Φᵗᵈ_space,
)

    center_space = axes(Φ_space)
    face_space = axes(A_space)
    # cell center variables
    ϕⱼ₋₁ = getidx(Φ_space, loc, idx - half - 1, hidx)
    ϕⱼ = getidx(Φ_space, loc, idx - half, hidx)
    ϕⱼ₊₁ = getidx(Φ_space, loc, idx + half, hidx)
    ϕⱼ₊₂ = getidx(Φ_space, loc, idx + half + 1, hidx)
    # cell center variables
    ϕⱼ₋₁ᵗᵈ = getidx(Φᵗᵈ_space, loc, idx - half - 1, hidx)
    ϕⱼᵗᵈ = getidx(Φᵗᵈ_space, loc, idx - half, hidx)
    ϕⱼ₊₁ᵗᵈ = getidx(Φᵗᵈ_space, loc, idx + half, hidx)
    ϕⱼ₊₂ᵗᵈ = getidx(Φᵗᵈ_space, loc, idx + half + 1, hidx)
    # cell face variables
    Aⱼ₊₁₂ = Geometry.contravariant3(
        getidx(A_space, loc, idx, hidx),
        Geometry.LocalGeometry(face_space, idx, hidx),
    )
    Aⱼ₋₁₂ = Geometry.contravariant3(
        getidx(A_space, loc, idx - 1, hidx),
        Geometry.LocalGeometry(face_space, idx - 1, hidx),
    )
    Aⱼ₊₃₂ = Geometry.contravariant3(
        getidx(A_space, loc, idx + 1, hidx),
        Geometry.LocalGeometry(face_space, idx + 1, hidx),
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

boundary_width(
    ::FCTZalesak,
    ::FirstOrderOneSided,
    A_space,
    Φ_space,
    Φᵗᵈ_space,
) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::FCTZalesak,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    A_space,
    Φ_space,
    Φᵗᵈ_space,
)
    face_space = axes(A_space)
    @assert idx <= left_face_boundary_idx(face_space) + 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_space))))
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::FCTZalesak,
    bc::FirstOrderOneSided,
    loc,
    idx,
    hidx,
    A_space,
    Φ_space,
    Φᵗᵈ_space,
)
    face_space = axes(A_space)
    @assert idx <= right_face_boundary_idx(face_space) - 1

    return Geometry.Contravariant3Vector(zero(eltype(eltype(A_space))))
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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.FaceFiniteDifferenceSpace,
) = arg_space
return_space(
    ::AdvectionF2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::AdvectionF2F, velocity, arg) = ((0, 0), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::AdvectionF2F,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³ = Geometry.contravariant3(
        getidx(velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    ∂θ₃ = RecursiveApply.rdiv(θ⁺ ⊟ θ⁻, 2)
    return w³ ⊠ ∂θ₃
end

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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = arg_space
return_space(
    ::AdvectionC2C,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::AdvectionC2C, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::AdvectionC2C,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

boundary_width(::AdvectionC2C, ::SetValue, velocity, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::AdvectionC2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(bc.val, loc, nothing, hidx) # defined at face, not the center
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
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
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(space)
    θ⁺ = getidx(bc.val, loc, nothing, hidx) # value at the face
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = 2 ⊠ (θ⁺ ⊟ θ)
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

boundary_width(::AdvectionC2C, ::Extrapolate, velocity, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (w³⁺ ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(space)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
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
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = arg_space
return_space(
    ::FluxCorrectionC2C,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::FluxCorrectionC2C, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::FluxCorrectionC2C,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (abs(w³⁺) ⊠ ∂θ₃⁺) ⊟ (abs(w³⁻) ⊠ ∂θ₃⁻)
end

boundary_width(::FluxCorrectionC2C, ::Extrapolate, velocity, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::FluxCorrectionC2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (abs(w³⁺) ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::FluxCorrectionC2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(space)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
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
    velocity_space::Spaces.CenterFiniteDifferenceSpace,
    arg_space::Spaces.FaceFiniteDifferenceSpace,
) = arg_space
return_space(
    ::FluxCorrectionF2F,
    velocity_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = arg_space

stencil_interior_width(::FluxCorrectionF2F, velocity, arg) =
    ((-half, +half), (-1, 1))
Base.@propagate_inbounds function stencil_interior(
    ::FluxCorrectionF2F,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (abs(w³⁺) ⊠ ∂θ₃⁺) ⊟ (abs(w³⁻) ⊠ ∂θ₃⁻)
end

boundary_width(::FluxCorrectionF2F, ::Extrapolate, velocity, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::FluxCorrectionF2F,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    θ⁺ = getidx(arg, loc, idx + 1, hidx)
    θ = getidx(arg, loc, idx, hidx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (abs(w³⁺) ⊠ ∂θ₃⁺)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::FluxCorrectionF2F,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    velocity,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    θ = getidx(arg, loc, idx, hidx)
    θ⁻ = getidx(arg, loc, idx - 1, hidx)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity, loc, idx - half, hidx),
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

return_space(::SetBoundaryOperator, space::Spaces.FaceFiniteDifferenceSpace) =
    space
return_space(
    ::SetBoundaryOperator,
    space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = space

stencil_interior_width(::SetBoundaryOperator, arg) = ((0, 0),)
Base.@propagate_inbounds stencil_interior(
    ::SetBoundaryOperator,
    loc,
    idx,
    hidx,
    arg,
) = getidx(arg, loc, idx, hidx)

boundary_width(::SetBoundaryOperator, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::GradientF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(::GradientF2C, space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::GradientF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::GradientF2C,
    loc,
    idx,
    hidx,
    arg,
)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(arg, loc, idx + half, hidx) ⊟ getidx(arg, loc, idx - half, hidx))
end

boundary_width(::GradientF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(arg)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(arg, loc, idx + half, hidx) ⊟ getidx(bc.val, loc, nothing, hidx))
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(arg)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(bc.val, loc, nothing, hidx) ⊟ getidx(arg, loc, idx - half, hidx))
end

boundary_width(::GradientF2C, ::Extrapolate, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    op::GradientF2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(arg)
    Geometry.project(
        Geometery.Covariant3Axis(),
        stencil_interior(op, loc, idx + 1, hidx, arg),
        Geometry.LocalGeometry(space, idx, hidx),
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    op::GradientF2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(arg)
    Geometry.project(
        Geometry.Covariant3Axis(),
        stencil_interior(op, loc, idx - 1, hidx, arg),
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

return_space(::GradientC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(::GradientC2F, space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::GradientC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::GradientC2F,
    loc,
    idx,
    hidx,
    arg,
)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(arg, loc, idx + half, hidx) ⊟ getidx(arg, loc, idx - half, hidx))
end

boundary_width(::GradientC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    # ∂x[i] = 2(∂x[i + half] - val)
    Geometry.Covariant3Vector(2) ⊗
    (getidx(arg, loc, idx + half, hidx) ⊟ getidx(bc.val, loc, nothing, hidx))
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    Geometry.Covariant3Vector(2) ⊗
    (getidx(bc.val, loc, nothing, idx) ⊟ getidx(arg, loc, idx - half, hidx))
end

# left / right SetGradient boundary conditions
boundary_width(::GradientC2F, ::SetGradient, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::GradientC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    space = axes(arg)
    # imposed flux boundary condition at left most face
    Geometry.project(
        Geometry.Covariant3Axis(),
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::GradientC2F,
    bc::SetGradient,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    space = axes(arg)
    # imposed flux boundary condition at right most face
    Geometry.project(
        Geometry.Covariant3Axis(),
        getidx(bc.val, loc, nothing, hidx),
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

return_space(::DivergenceF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)
return_space(::DivergenceF2C, space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::DivergenceF2C, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::DivergenceF2C,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceF2C, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceF2C,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(bc.val, loc, nothing, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceF2C, ::Extrapolate, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    op::DivergenceF2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_center_boundary_idx(arg)
    stencil_interior(op, loc, idx + 1, hidx, arg)
end
Base.@propagate_inbounds function stencil_right_boundary(
    op::DivergenceF2C,
    ::Extrapolate,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_center_boundary_idx(arg)
    stencil_interior(op, loc, idx - 1, hidx, arg)
end

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

return_space(::DivergenceC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::DivergenceC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::DivergenceC2F, arg) = ((-half, half),)
Base.@propagate_inbounds function stencil_interior(
    ::DivergenceC2F,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³₊ ⊟ Ju³₋) ⊠ local_geometry.invJ
end

boundary_width(::DivergenceC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    # ∂x[i] = 2(∂x[i + half] - val)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half, hidx),
        Geometry.LocalGeometry(space, idx + half, hidx),
    )
    Ju³ = Geometry.Jcontravariant3(
        getidx(bc.val, loc, nothing, hidx),
        local_geometry,
    )
    (Ju³₊ ⊟ Ju³) ⊠ (2 * local_geometry.invJ)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    Ju³ = Geometry.Jcontravariant3(
        getidx(bc.val, loc, nothing, hidx),
        local_geometry,
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half, hidx),
        Geometry.LocalGeometry(space, idx - half, hidx),
    )
    (Ju³ ⊟ Ju³₋) ⊠ (2 * local_geometry.invJ)
end

# left / right SetDivergence boundary conditions
boundary_width(::DivergenceC2F, ::SetDivergence, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::DivergenceC2F,
    bc::SetDivergence,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    # imposed flux boundary condition at left most face
    getidx(bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::DivergenceC2F,
    bc::SetDivergence,
    loc,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    # imposed flux boundary condition at right most face
    getidx(bc.val, loc, nothing, hidx)
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

return_space(::CurlC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)
return_space(::CurlC2F, space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(space)

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
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    u₊ = getidx(arg, loc, idx + half, hidx)
    u₋ = getidx(arg, loc, idx - half, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u₊, u₋, local_geometry.invJ)
end

boundary_width(::CurlC2F, ::SetValue, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::CurlC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    u₊ = getidx(arg, loc, idx + half, hidx)
    u = getidx(bc.val, loc, nothing, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u₊, u, local_geometry.invJ * 2)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::CurlC2F,
    bc::SetValue,
    loc,
    idx,
    hidx,
    arg,
)
    space = axes(arg)
    u = getidx(bc.val, loc, nothing, hidx)
    u₋ = getidx(arg, loc, idx - half, hidx)
    local_geometry = Geometry.LocalGeometry(space, idx, hidx)
    return fd3_curl(u, u₋, local_geometry.invJ * 2)
end

boundary_width(::CurlC2F, ::SetCurl, arg) = 1
Base.@propagate_inbounds function stencil_left_boundary(
    ::CurlC2F,
    bc::SetCurl,
    loc,
    idx,
    hidx,
    arg,
)
    return getidx(bc.val, loc, nothing, hidx)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::CurlC2F,
    bc::SetCurl,
    loc,
    idx,
    hidx,
    arg,
)
    return getidx(bc.val, loc, nothing, hidx)
end



stencil_interior_width(bc::Base.Broadcast.Broadcasted{StencilStyle}) =
    stencil_interior_width(bc.f, bc.args...)

boundary_width(bc::Base.Broadcast.Broadcasted{StencilStyle}, loc) =
    has_boundary(bc.f, loc) ?
    boundary_width(bc.f, get_boundary(bc.f, loc), bc.args...) : 0

@inline function left_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    widths = stencil_interior_width(bc)
    args_idx = _left_interior_window_idx_args(bc.args, space, loc)
    args_idx_widths = tuplemap((arg, width) -> arg - width[1], args_idx, widths)
    return max(
        max(args_idx_widths...),
        left_idx(space) + boundary_width(bc, loc),
    )
end

@inline function right_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    widths = stencil_interior_width(bc)
    args_idx = _right_interior_window_idx_args(bc.args, space, loc)
    args_widths = tuplemap((arg, width) -> arg - width[2], args_idx, widths)
    return min(min(args_widths...), right_idx(space) - boundary_width(bc, loc))
end

@inline _left_interior_window_idx_args(args::Tuple, space, loc) = (
    left_interior_window_idx(args[1], space, loc),
    _left_interior_window_idx_args(Base.tail(args), space, loc)...,
)
@inline _left_interior_window_idx_args(args::Tuple{Any}, space, loc) =
    (left_interior_window_idx(args[1], space, loc),)
@inline _left_interior_window_idx_args(args::Tuple{}, space, loc) = ()

@inline function left_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = _left_interior_window_idx_args(bc.args, space, loc)
    maximum(arg_idxs)
end

@inline _right_interior_window_idx_args(args::Tuple, space, loc) = (
    right_interior_window_idx(args[1], space, loc),
    _right_interior_window_idx_args(Base.tail(args), space, loc)...,
)
@inline _right_interior_window_idx_args(args::Tuple{Any}, space, loc) =
    (right_interior_window_idx(args[1], space, loc),)
@inline _right_interior_window_idx_args(args::Tuple{}, space, loc) = ()

@inline function right_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = _right_interior_window_idx_args(bc.args, space, loc)
    minimum(arg_idxs)
end

@inline function left_interior_window_idx(
    field::Field,
    _,
    loc::LeftBoundaryWindow,
)
    left_idx(axes(field))
end

@inline function right_interior_window_idx(
    field::Field,
    _,
    loc::RightBoundaryWindow,
)
    right_idx(axes(field))
end

@inline function left_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::LeftBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    left_idx(axes(bc))
end

@inline function right_interior_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::RightBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    right_idx(axes(bc))
end

@inline function left_interior_window_idx(_, space, loc::LeftBoundaryWindow)
    left_idx(space)
end

@inline function right_interior_window_idx(_, space, loc::RightBoundaryWindow)
    right_idx(space)
end

Base.@propagate_inbounds function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::Interior,
    idx,
    hidx,
)
    stencil_interior(bc.f, loc, idx, hidx, bc.args...)
end

Base.@propagate_inbounds function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::LeftBoundaryWindow,
    idx,
    hidx,
)
    op = bc.f
    space = axes(bc)
    if has_boundary(bc.f, loc) &&
       idx < left_idx(space) + boundary_width(bc, loc)
        stencil_left_boundary(
            op,
            get_boundary(op, loc),
            loc,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, hidx, bc.args...)
    end
end

Base.@propagate_inbounds function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::RightBoundaryWindow,
    idx,
    hidx,
)
    op = bc.f
    space = axes(bc)
    if has_boundary(bc.f, loc) &&
       idx > (right_idx(space) - boundary_width(bc, loc))
        stencil_right_boundary(
            op,
            get_boundary(op, loc),
            loc,
            idx,
            hidx,
            bc.args...,
        )
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, hidx, bc.args...)
    end
end

# broadcasting a StencilStyle gives a CompositeStencilStyle
Base.Broadcast.BroadcastStyle(
    ::Type{<:Base.Broadcast.Broadcasted{StencilStyle}},
) = CompositeStencilStyle()

Base.Broadcast.BroadcastStyle(
    ::AbstractStencilStyle,
    ::Fields.AbstractFieldStyle,
) = CompositeStencilStyle()

Base.eltype(bc::Base.Broadcast.Broadcasted{StencilStyle}) =
    return_eltype(bc.f, bc.args...)

Base.@propagate_inbounds function getidx(
    bc::Fields.CenterFiniteDifferenceField,
    ::Location,
    idx::Integer,
)
    field_data = Fields.field_values(bc)
    space = axes(bc)
    if Topologies.isperiodic(space.topology)
        idx = mod1(idx, length(space))
    end
    return @inbounds field_data[idx]
end

Base.@propagate_inbounds function getidx(
    bc::Fields.FaceFiniteDifferenceField,
    ::Location,
    idx::PlusHalf,
)
    field_data = Fields.field_values(bc)
    space = axes(bc)
    i = idx.i + 1
    if Topologies.isperiodic(space.topology)
        i = mod1(i, length(space))
    end
    return @inbounds field_data[i]
end

# unwap boxed scalars
@inline getidx(scalar::Ref, loc::Location, idx, hidx) = scalar[]
@inline getidx(field::Fields.PointField, loc::Location, idx, hidx) = field[]
@inline getidx(field::Fields.PointField, loc::Location, idx) = field[]

# recursive fallback for scalar, just return
@inline getidx(scalar, ::Location, idx, hidx) = scalar

# getidx error fallbacks
@noinline inferred_getidx_error(idx_type::Type, space_type::Type) =
    error("Invalid index type `$idx_type` for field on space `$space_type`")

Base.@propagate_inbounds function getidx(
    field::Fields.Field,
    loc::Location,
    idx,
    hidx,
)
    getidx(column(field, hidx...), loc, idx)
    # inferred_getidx_error(typeof(idx), typeof(axes(field)))
end

function getidx(
    field::Base.Broadcast.Broadcasted{StencilStyle},
    ::Location,
    idx,
    hidx,
)
    space = axes(field)
    inferred_getidx_error(typeof(idx), typeof(space))
end

# recursively unwrap getidx broadcast arguments in a way that is statically reducible by the optimizer
Base.@propagate_inbounds getidx_args(args::Tuple, loc::Location, idx, hidx) = (
    getidx(args[1], loc, idx, hidx),
    getidx_args(Base.tail(args), loc, idx, hidx)...,
)
Base.@propagate_inbounds getidx_args(
    arg::Tuple{Any},
    loc::Location,
    idx,
    hidx,
) = (getidx(arg[1], loc, idx, hidx),)
Base.@propagate_inbounds getidx_args(::Tuple{}, loc::Location, idx, hidx) = ()

Base.@propagate_inbounds function getidx(
    bc::Base.Broadcast.Broadcasted,
    loc::Location,
    idx,
    hidx,
)
    #_args = tuplemap(arg -> getidx(arg, loc, idx), bc.args)
    _args = getidx_args(bc.args, loc, idx, hidx)
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
    field::Union{
        Fields.CenterFiniteDifferenceField,
        Fields.CenterExtrudedFiniteDifferenceField,
    },
    idx::Integer,
    hidx,
    val,
)
    field_data = Fields.field_values(field)
    column(field_data, hidx...)[idx] = val
    val
end

Base.@propagate_inbounds function setidx!(
    field::Union{
        Fields.FaceFiniteDifferenceField,
        Fields.FaceExtrudedFiniteDifferenceField,
    },
    idx::PlusHalf,
    hidx,
    val,
)
    field_data = Fields.field_values(field)
    @inbounds column(field_data, hidx...)[idx.i + 1] = val
    val
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    Base.Broadcast.broadcasted(StencilStyle(), op, args...)
end

# recursively unwrap axes broadcast arguments in a way that is statically reducible by the optimizer
@inline axes_args(args::Tuple) = (axes(args[1]), axes_args(Base.tail(args))...)
@inline axes_args(arg::Tuple{Any}) = (axes(arg[1]),)
@inline axes_args(::Tuple{}) = ()

function Base.Broadcast.broadcasted(
    ::StencilStyle,
    op::FiniteDifferenceOperator,
    args...,
)
    _axes = return_space(op, axes_args(args)...)
    Base.Broadcast.Broadcasted{StencilStyle}(op, args, _axes)
end

Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{StencilStyle}) = bc
Base.Broadcast._broadcast_getindex_eltype(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
) = eltype(bc)

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

@inline function _serial_copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
    Ni::Int,
    Nj::Int,
    Nh::Int,
) where {S <: AbstractStencilStyle}
    @inbounds for h in 1:Nh, j in 1:Nj, i in 1:Ni
        apply_stencil!(field_out, bc, (i, j, h))
    end
    return field_out
end

@inline function _threaded_copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
    Ni::Int,
    Nj::Int,
    Nh::Int,
) where {S <: AbstractStencilStyle}
    @inbounds begin
        Threads.@threads for h in 1:Nh
            for j in 1:Nj, i in 1:Ni
                apply_stencil!(field_out, bc, (i, j, h))
            end
        end
    end
    return field_out
end

@inline function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
) where {S <: AbstractStencilStyle}
    space = axes(bc)
    local_geometry = Spaces.local_geometry_data(space)
    (Ni, Nj, _, _, Nh) = size(local_geometry)
    if enable_threading() && Nh > 1
        return _threaded_copyto!(field_out, bc, Ni, Nj, Nh)
    end
    return _serial_copyto!(field_out, bc, Ni, Nj, Nh)
end

Base.@propagate_inbounds function apply_stencil!(field_out, bc, hidx)
    space = axes(bc)
    if Topologies.isperiodic(Spaces.vertical_topology(space))
        @inbounds for idx in
                      left_idx(space):(left_idx(space) + length(space) - 1)
            setidx!(field_out, idx, hidx, getidx(bc, Interior(), idx, hidx))
        end
    else
        lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
        rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
        li = left_idx(space)
        lw = left_interior_window_idx(bc, space, lbw)::typeof(li)
        ri = right_idx(space)
        rw = right_interior_window_idx(bc, space, rbw)::typeof(ri)
        @assert li <= lw <= rw <= ri
        @inbounds for idx in li:(lw - 1)
            setidx!(field_out, idx, hidx, getidx(bc, lbw, idx, hidx))
        end
        @inbounds for idx in lw:rw
            setidx!(field_out, idx, hidx, getidx(bc, Interior(), idx, hidx))
        end
        @inbounds for idx in (rw + 1):ri
            setidx!(field_out, idx, hidx, getidx(bc, rbw, idx, hidx))
        end
    end
    return field_out
end
