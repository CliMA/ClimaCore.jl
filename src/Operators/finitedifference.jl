# The geometry of the staggerd grid used by FiniteDifference in the vertical (sometimes called the C-grid)
# is (in one dimension) shown below

#         face   cell   face   cell   face
# left                                        right
#                 i-1            i
#                  ↓             ↓
#           |      ×      |      ×      |
#           ↑             ↑             ↑
#          i-1            i            i+1


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
  /       \
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
    Extrapolate()

Set the value at the boundary to be the same as the closest interior point.
"""
struct Extrapolate <: BoundaryCondition end

abstract type Location end
abstract type Boundary <: Location end
abstract type BoundaryWindow <: Location end

struct Interior <: Location end
struct LeftBoundaryWindow{name} <: BoundaryWindow end
struct RightBoundaryWindow{name} <: BoundaryWindow end

abstract type FiniteDifferenceOperator end

return_eltype(::FiniteDifferenceOperator, arg) = eltype(arg)
boundary_width(op::FiniteDifferenceOperator, bc) =
    error("Boundary $(typeof(bc)) is not supported for operator $(typeof(op))")

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

stencil_interior_width(::InterpolateF2C) = ((0, 1),)

function stencil_interior(::InterpolateF2C, loc, idx, arg)
    RecursiveApply.rdiv((getidx(arg, loc, idx + 1) ⊞ getidx(arg, loc, idx)), 2)
end

"""
    InterpolateC2F(;boundaries..)

Interpolate from center to face. Supported boundary conditions are:

- [`SetValue(val)`](@ref): set the value at the boundary face to be `val`.
- [`SetGradient`](@ref): set the value at the boundary such that the gradient is `val`.
- [`Extrapolate`](@ref): use the closest interior point as the boundary value
"""
struct InterpolateC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
InterpolateC2F(; kwargs...) = InterpolateC2F(NamedTuple(kwargs))

return_space(::InterpolateC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateC2F) = ((-1, 0),)

function stencil_interior(::InterpolateC2F, loc, idx, arg)
    space = Fields.space(arg)
    #RecursiveApply.rdiv(((getidx(arg, loc, idx) ⊠ space.Δh_f2f[idx]) ⊞ (getidx(arg, loc, idx - 1) ⊠ space.Δh_f2f[idx-1])), 2 ⊠ space.Δh_c2c[idx])
    RecursiveApply.rdiv(getidx(arg, loc, idx) ⊞ getidx(arg, loc, idx - 1), 2)
end

boundary_width(op::InterpolateC2F, ::SetValue) = 1
function stencil_left_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    @assert idx == 1
    bc.val
end
function stencil_right_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == length(space.Δh_c2c)
    bc.val
end

boundary_width(op::InterpolateC2F, ::SetGradient) = 1
function stencil_left_boundary(::InterpolateC2F, bc::SetGradient, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == 1
    # Δh_c2c[1] is f1 to c1 distance
    getidx(arg, loc, idx) ⊟ (space.Δh_c2c[idx] ⊠ bc.val)
end
function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    space = Fields.space(arg)
    @assert idx == length(space.Δh_c2c) # n+1
    getidx(arg, loc, idx - 1) ⊞ (space.Δh_c2c[idx] ⊠ bc.val)
end

boundary_width(op::InterpolateC2F, ::Extrapolate) = 1
function stencil_left_boundary(::InterpolateC2F, bc::Extrapolate, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == 1
    # Δh_c2c[1] is f1 to c1 distance
    getidx(arg, loc, idx)
end
function stencil_right_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    arg,
)
    space = Fields.space(arg)
    @assert idx == length(space.Δh_c2c) # n+1
    getidx(arg, loc, idx - 1)
end

"""
    UpwindLeftC2F(;boundaries)

Interpolate from the left. Only the left boundary condition should be set:
- [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
"""
struct UpwindLeftC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
UpwindLeftC2F(; kwargs...) = UpwindLeftC2F(NamedTuple(kwargs))

return_space(::UpwindLeftC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::UpwindLeftC2F) = ((-1, -1),)

stencil_interior(::UpwindLeftC2F, loc, idx, arg) = getidx(arg, loc, idx - 1)

boundary_width(op::UpwindLeftC2F, ::SetValue) = 1

function stencil_left_boundary(::UpwindLeftC2F, bc::SetValue, loc, idx, arg)
    @assert idx == 1
    bc.val
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

stencil_interior_width(::SetBoundaryOperator) = ((0, 0),)

function stencil_interior(::SetBoundaryOperator, loc, idx, arg)
    getidx(arg, loc, idx)
end

boundary_width(op::SetBoundaryOperator, ::SetValue) = 1
function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    arg,
)
    @assert idx == 1
    bc.val
end
function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    arg,
)
    bc.val
end


abstract type GradientOperator <: FiniteDifferenceOperator end

"""
    GradientF2C(;boundaryname=boundarycondition...)

Centered-difference gradient operator from a `FaceFiniteDifferenceSpace` to a
`CenterFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - by default, the current value at the boundary face will be used.
 - [`SetValue(val)`](@ref): calculate the gradient assuming the value at the boundary is `val`.
 - [`Extrapolate()`](@ref): set the value at the center closest to the boundary
   to be the same as the neighbouring interior value.
"""
struct GradientF2C{BCS} <: GradientOperator
    bcs::BCS
end
GradientF2C(; kwargs...) = GradientF2C(NamedTuple(kwargs))

return_space(::GradientF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)

stencil_interior_width(::GradientF2C) = ((0, 1),)

function stencil_interior(::GradientF2C, loc, idx, arg)
    space = Fields.space(arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + 1) ⊟ getidx(arg, loc, idx)),
        space.Δh_f2f[idx],
    )
end

boundary_width(op::GradientF2C, ::SetValue) = 1
function stencil_left_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == 1
    RecursiveApply.rdiv((getidx(arg, loc, idx + 1) ⊟ bc.val), space.Δh_f2f[idx])
end

function stencil_right_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    space = Fields.space(arg)
    # Δh_f2f = [f[2] - f[1], f[3] - f[2], ..., f[n] - f[n-1], f[n+1] - f[n]]
    @assert idx == length(space.Δh_f2f) # n
    RecursiveApply.rdiv((bc.val ⊟ getidx(arg, loc, idx)), space.Δh_f2f[idx])
end


boundary_width(op::GradientF2C, ::Extrapolate) = 1
function stencil_left_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == 1
    stencil_interior(op, loc, idx + 1, arg)
end
function stencil_right_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = Fields.space(arg)
    # Δh_f2f = [f[2] - f[1], f[3] - f[2], ..., f[n] - f[n-1], f[n+1] - f[n]]
    @assert idx == length(space.Δh_f2f) # n
    stencil_interior(op, loc, idx - 1, arg)
end

"""
    GradientC2F(;boundaries)

Centered-difference gradient operator from a `CenterFiniteDifferenceSpace` to a
`FaceFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - [`SetValue(val)`](@ref): calculate the gradient assuming the value at the boundary is `val`.
 - [`SetGradient(val)`](@ref): set the value of the gradient at the boundary to be `val`.
"""
struct GradientC2F{BC} <: GradientOperator
    bcs::BC
end
GradientC2F(; kwargs...) = GradientC2F(NamedTuple(kwargs))

return_space(::GradientC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::GradientC2F) = ((-1, 0),)

function stencil_interior(::GradientC2F, loc, idx, arg)
    space = Fields.space(arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx) ⊟ getidx(arg, loc, idx - 1)),
        space.Δh_c2c[idx],
    )
end

boundary_width(op::GradientC2F, ::SetValue) = 1
boundary_width(op::GradientC2F, ::SetGradient) = 1

function stencil_left_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == 1
    # Δh_c2c[1] is f1 to c1 distance
    RecursiveApply.rdiv((getidx(arg, loc, idx) ⊟ bc.val), space.Δh_c2c[idx])
end

function stencil_right_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    space = Fields.space(arg)
    # Δh_c2c = [c[1] - f[1], c[2] - c[1], ..., c[n] - c[n-1], f[n+1] - c[n]]
    @assert idx == length(space.Δh_c2c) # n+1
    # Δh_c2c[end] is c[n] to f[n+1] distance
    RecursiveApply.rdiv((bc.val ⊟ getidx(arg, loc, idx - 1)), space.Δh_c2c[idx])
end

# left / right SetGradient boundary conditions
function stencil_left_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == 1
    # imposed flux boundary condition at left most face
    bc.val
end

function stencil_right_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    space = Fields.space(arg)
    @assert idx == length(space.Δh_c2c)  # n+1
    # imposed flux boundary condition at right most face
    bc.val
end

left_boundary_width(obj, loc) = 0
right_boundary_width(obj, loc) = 0

left_boundary_width(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::LeftBoundaryWindow,
) = has_boundary(bc.f, loc) ? boundary_width(bc.f, get_boundary(bc.f, loc)) : 0

right_boundary_width(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::RightBoundaryWindow,
) = has_boundary(bc.f, loc) ? boundary_width(bc.f, get_boundary(bc.f, loc)) : 0

left_boundary_window_width(obj, loc) = 0
right_boundary_window_width(obj, loc) = 0

function left_boundary_window_width(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    loc,
)
    maximum(arg -> left_boundary_window_width(arg, loc), bc.args)
end

function right_boundary_window_width(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    loc,
)
    maximum(arg -> right_boundary_window_width(arg, loc), bc.args)
end

function left_boundary_window_width(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc,
)
    op = bc.f
    args = bc.args
    l = maximum(
        map(
            (a, w) -> left_boundary_window_width(a, loc) - w[1],
            args,
            stencil_interior_width(op),
        ),
    )
    max(l, left_boundary_width(bc, loc))
end

function right_boundary_window_width(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc,
)
    op = bc.f
    args = bc.args
    l = maximum(
        map(
            (a, w) ->
                right_boundary_window_width(a, loc) +
                w[2] +
                stagger_correct(Fields.space(bc), Fields.space(a)),
            args,
            stencil_interior_width(op),
        ),
    )
    max(l, right_boundary_width(bc, loc))
end

# when we go from a center to face, we get an extra point
stagger_correct(
    to::Spaces.FaceFiniteDifferenceSpace,
    from::Spaces.CenterFiniteDifferenceSpace,
) = 1

# when we go from a face to center, we lose an extra point
stagger_correct(
    to::Spaces.CenterFiniteDifferenceSpace,
    from::Spaces.FaceFiniteDifferenceSpace,
) = -1

stagger_correct(to::S, from::S) where {S <: Spaces.FiniteDifferenceSpace} = 0

# left / right dirchlet boundry conditions

# Gradient Face-> Center
# interior stencil

# left / right SetGradient boundary conditions

function getidx(
    bd::Base.Broadcast.Broadcasted{StencilStyle},
    loc::Interior,
    idx,
)
    stencil_interior(bd.f, loc, idx, bd.args...)
end

function getidx(
    bd::Base.Broadcast.Broadcasted{StencilStyle},
    loc::LeftBoundaryWindow,
    idx,
)
    op = bd.f
    if idx <= left_boundary_width(bd, loc)
        stencil_left_boundary(op, get_boundary(op, loc), loc, idx, bd.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bd.args...)
    end
end

# faces: 1|2 3 4 ... 9 10|11
# center: 1|2 3 ....  9|10

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::RightBoundaryWindow,
    idx,
)
    op = bc.f
    n = length(Fields.space(bc))
    if idx > (n - right_boundary_width(bc, loc))
        stencil_right_boundary(op, get_boundary(op, loc), loc, idx, bc.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bc.args...)
    end
end

# broadcasting a StencilStyle gives a CompositeStencilStyle
# TODO: define precedence when multiple args are used
Base.Broadcast.BroadcastStyle(
    ::Type{<:Base.Broadcast.Broadcasted{StencilStyle}},
) = CompositeStencilStyle()

Base.Broadcast.BroadcastStyle(
    ::AbstractStencilStyle,
    ::Fields.AbstractFieldStyle,
) = CompositeStencilStyle()

Base.eltype(bc::Base.Broadcast.Broadcasted{StencilStyle}) =
    return_eltype(bc.f, bc.args...)

function getidx(bc::Field, ::Location, idx)
    Fields.field_values(bc)[idx]
end
getidx(scalar, ::Location, idx) = scalar

function getidx(
    bc::Base.Broadcast.Broadcasted{FS},
    loc::Location,
    idx,
) where {FS <: Fields.AbstractFieldStyle}
    bc.f(map(arg -> getidx(arg, loc, idx), bc.args)...)
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    Base.Broadcast.broadcasted(StencilStyle(), op, args...)
end

function Base.Broadcast.broadcasted(
    ::StencilStyle,
    op::FiniteDifferenceOperator,
    args...,
)
    axes = (return_space(op, map(Fields.space, args)...),)
    Base.Broadcast.Broadcasted{StencilStyle}(op, args, axes)
end

Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{StencilStyle}) = bc
Base.Broadcast._broadcast_getindex_eltype(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
) = eltype(bc)

Fields.space(bc::Base.Broadcast.Broadcasted{StencilStyle}) = axes(bc)[1]

function Base.similar(
    bc::Base.Broadcast.Broadcasted{S},
    ::Type{Eltype},
) where {Eltype, S <: AbstractStencilStyle}
    sp = Fields.space(bc)
    return Field(similar(Spaces.coordinates(sp), Eltype), sp)
end

function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
) where {S <: AbstractStencilStyle}
    data_out = Fields.field_values(field_out)
    apply_stencil!(data_out, bc)
    return field_out
end

Spaces.interior_indices(field::Field) = Spaces.real_indices(Fields.space(field))

Spaces.interior_indices(
    field::Base.Broadcast.Broadcasted{FS},
) where {FS <: Fields.AbstractFieldStyle} =
    Spaces.real_indices(Fields.space(field))

function Spaces.interior_indices(bc::Base.Broadcast.Broadcasted{StencilStyle})
    width_op = stencil_interior_width(bc.f)   # tuple of 2-tuples
    indices_args = map(Spaces.interior_indices, bc.args) # tuple of indices
    @assert length(width_op) == length(indices_args)
    lo = minimum(map((o, a) -> first(a) - o[1], width_op, indices_args))
    hi = maximum(map((o, a) -> last(a) - o[2], width_op, indices_args))
    return lo:hi
end

function apply_stencil!(data_out, bc)
    space = Fields.space(bc)
    n = length(space)
    lb = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    rb = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    l = left_boundary_window_width(bc, lb) + 1
    r = n - right_boundary_window_width(bc, rb)
    for idx in 1:(l - 1)
        data_out[idx] = getidx(bc, lb, idx)
    end
    for idx in l:r
        data_out[idx] = getidx(bc, Interior(), idx)
    end
    for idx in (r + 1):n
        data_out[idx] = getidx(bc, rb, idx)
    end
    return data_out
end
