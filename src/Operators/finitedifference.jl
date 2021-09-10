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


"""
    PlusHalf(i)

Represents `i + 1/2`, but stored as internally as an integer value. Used for
indexing into staggered finite difference meshes: the convention "half" values
are indexed at cell faces, whereas centers are indexed at cell centers.

Supports `+`, `-` and inequalities.

See also [`half`](@ref).
"""
struct PlusHalf{I <: Integer} <: Real
    i::I
end
PlusHalf{I}(h::PlusHalf{I}) where {I <: Integer} = h

"""
    const half = PlusHalf(0)
"""
const half = PlusHalf(0)

Base.:+(h::PlusHalf) = h
Base.:-(h::PlusHalf) = PlusHalf(-h.i - one(h.i))
Base.:+(i::Integer, h::PlusHalf) = PlusHalf(i + h.i)
Base.:+(h::PlusHalf, i::Integer) = PlusHalf(h.i + i)
Base.:+(h1::PlusHalf, h2::PlusHalf) = h1.i + h2.i + one(h1.i)
Base.:-(i::Integer, h::PlusHalf) = PlusHalf(i - h.i - one(h.i))
Base.:-(h::PlusHalf, i::Integer) = PlusHalf(h.i - i)
Base.:-(h1::PlusHalf, h2::PlusHalf) = h1.i - h2.i

Base.:<=(h1::PlusHalf, h2::PlusHalf) = h1.i <= h2.i
Base.:<(h1::PlusHalf, h2::PlusHalf) = h1.i < h2.i
Base.max(h1::PlusHalf, h2::PlusHalf) = PlusHalf(max(h1.i, h2.i))
Base.min(h1::PlusHalf, h2::PlusHalf) = PlusHalf(min(h1.i, h2.i))

Base.convert(::Type{P}, i::Integer) where {P <: PlusHalf} =
    throw(InexactError(:convert, P, i))
Base.convert(::Type{I}, h::PlusHalf) where {I <: Integer} =
    throw(InexactError(:convert, I, h))

Base.step(::AbstractUnitRange{PlusHalf{I}}) where {I} = one(I)


left_idx(::Spaces.CenterFiniteDifferenceSpace) = 1
right_idx(space::Spaces.CenterFiniteDifferenceSpace) =
    length(space.center_local_geometry)
left_idx(::Spaces.FaceFiniteDifferenceSpace) = half
right_idx(space::Spaces.FaceFiniteDifferenceSpace) =
    PlusHalf(length(space.center_local_geometry))

left_face_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    left_idx(Spaces.FaceFiniteDifferenceSpace(space))
right_face_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    right_idx(Spaces.FaceFiniteDifferenceSpace(space))
left_center_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    left_idx(Spaces.CenterFiniteDifferenceSpace(space))
right_center_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    right_idx(Spaces.CenterFiniteDifferenceSpace(space))

left_face_boundary_idx(arg) = left_face_boundary_idx(axes(arg))
right_face_boundary_idx(arg) = right_face_boundary_idx(axes(arg))
left_center_boundary_idx(arg) = left_center_boundary_idx(axes(arg))
right_center_boundary_idx(arg) = right_center_boundary_idx(axes(arg))

function Geometry.LocalGeometry(
    space::Spaces.FiniteDifferenceSpace,
    idx::Integer,
)
    space.center_local_geometry[idx]
end
function Geometry.LocalGeometry(
    space::Spaces.FiniteDifferenceSpace,
    idx::PlusHalf,
)
    space.face_local_geometry[idx.i + 1]
end



Δh_f2f(space::Spaces.FiniteDifferenceSpace, idx::Integer) =
    space.center_local_geometry.WJ[idx]
Δh_c2c(space::Spaces.FiniteDifferenceSpace, idx::PlusHalf) =
    space.face_local_geometry.WJ[idx.i + 1]

# boundary face to left first cell center distance
Δh_left_bf2c(space::Spaces.FiniteDifferenceSpace) =
    space.face_local_geometry.J[1] / 2

# last right cell center to last boundary face distance
Δh_right_c2bf(space::Spaces.FiniteDifferenceSpace) =
    space.face_local_geometry.J[end] / 2

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


"""
    stencil_interior_width(op::Op)

The width of the interior stencil for operator of type `Op`. This should return a tuple
of 2-tuples: each 2-tuple should be the lower and upper bounds of the index offsets of
the stencil for each argument in the stencil.
"""
function stencil_interior_width end


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

stencil_interior_width(::InterpolateF2C) = ((-half, half),)

function stencil_interior(::InterpolateF2C, loc, idx, arg)
    a⁺ = getidx(arg, loc, idx + half)
    a⁻ = getidx(arg, loc, idx - half)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
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

return_space(
    ::InterpolateC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateC2F) = ((-half, half),)

function stencil_interior(::InterpolateC2F, loc, idx, arg)
    #RecursiveApply.rdiv(((getidx(arg, loc, idx) ⊠ space.Δh_f2f[idx]) ⊞ (getidx(arg, loc, idx - 1) ⊠ space.Δh_f2f[idx-1])), 2 ⊠ space.Δh_c2c[idx])
    a⁺ = getidx(arg, loc, idx + half)
    a⁻ = getidx(arg, loc, idx - half)
    RecursiveApply.rdiv(a⁺ ⊞ a⁻, 2)
end

boundary_width(op::InterpolateC2F, ::SetValue) = 1

function stencil_left_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end

function stencil_right_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    bc.val
end

boundary_width(op::InterpolateC2F, ::SetGradient) = 1

function stencil_left_boundary(::InterpolateC2F, bc::SetGradient, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half)
    a⁺ ⊟ (Δh_left_bf2c(space) ⊠ bc.val)
end

function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(arg, loc, idx - half)
    a⁻ ⊞ (Δh_right_c2bf(space) ⊠ bc.val)
end

boundary_width(op::InterpolateC2F, ::Extrapolate) = 1

function stencil_left_boundary(::InterpolateC2F, bc::Extrapolate, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    a⁺ = getidx(arg, loc, idx + half)
    a⁺
end

function stencil_right_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    a⁻ = getidx(arg, loc, idx - half)
    a⁻
end

"""
    WeightedInterpolateF2C()

Interpolate from face to center mesh weighted by face field `ω`.

    op = WeightedInterpolateF2C()
    center_field .= op(ω, face_field)

No boundary conditions are required (or supported).
"""
struct WeightedInterpolateF2C{BCS} <: InterpolationOperator
    bcs::BCS
end
WeightedInterpolateF2C(; kwargs...) = InterpolateF2C(NamedTuple(kwargs))

function return_space(
    ::WeightedInterpolateF2C,
    weight_space::Spaces.FaceFiniteDifferenceSpace,
    value_space::Spaces.FaceFiniteDifferenceSpace,
)
    Spaces.CenterFiniteDifferenceSpace(value_space)
end

function return_space(
    ::WeightedInterpolateF2C,
    weight_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    value_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
)
    Spaces.CenterExtrudedFiniteDifferenceSpace(value_space)
end

stencil_interior_width(::WeightedInterpolateF2C) = ((0, 0), (-half, half))

function stencil_interior(
    ::WeightedInterpolateF2C,
    loc,
    idx,
    weight_field,
    value_field,
)
    w⁺ = getidx(weight_field, loc, idx + half)
    w⁻ = getidx(weight_field, loc, idx - half)
    a⁺ = getidx(value_field, loc, idx + half)
    a⁻ = getidx(value_field, loc, idx - half)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (2 ⊠ (w⁺ ⊞ w⁻)))
end

"""
    WeightedInterpolateC2F(;boundaries..)

Interpolate from center to face weighted by weight field `ω`.

    op = WeightedInterpolateC2F(...)
    face_field .= op(ω, center_field)

Supported boundary conditions are:

- [`SetValue(val)`](@ref): set the value at the boundary face to be `val`.
- [`SetGradient`](@ref): set the value at the boundary such that the gradient is `val`.
- [`Extrapolate`](@ref): use the closest interior point as the boundary value

"""
struct WeightedInterpolateC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
WeightedInterpolateC2F(; kwargs...) = WeightedInterpolateC2F(NamedTuple(kwargs))

function return_space(
    ::WeightedInterpolateC2F,
    weight_space::Spaces.CenterFiniteDifferenceSpace,
    value_space::Spaces.CenterFiniteDifferenceSpace,
)
    Spaces.FaceFiniteDifferenceSpace(value_space)
end

function return_space(
    ::WeightedInterpolateC2F,
    weight_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    value_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    Spaces.FaceExtrudedFiniteDifferenceSpace(value_space)
end

stencil_interior_width(::WeightedInterpolateC2F) = ((0, 0), (0, 1))

function stencil_interior(
    ::WeightedInterpolateC2F,
    loc,
    idx,
    weight_field,
    value_field,
)
    w⁺ = getidx(weight_field, loc, idx + half)
    w⁻ = getidx(weight_field, loc, idx - half)
    a⁺ = getidx(value_field, loc, idx + half)
    a⁻ = getidx(value_field, loc, idx - half)
    RecursiveApply.rdiv((w⁺ ⊠ a⁺) ⊞ (w⁻ ⊠ a⁻), (2 ⊠ (w⁺ ⊞ w⁻)))
end

boundary_width(op::WeightedInterpolateC2F, ::SetValue) = 1

function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    weight_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_face_boundary_idx(space)
    bc.val
end

function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetValue,
    loc,
    idx,
    weight_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_face_boundary_idx(space)
    bc.val
end

boundary_width(op::WeightedInterpolateC2F, ::SetGradient) = 1

function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    weight_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half)
    a⁺ ⊟ (Δh_left_bf2c(space) ⊠ bc.val)
end

function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    weight_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(value_field, loc, idx - half)
    a⁻ ⊞ (Δh_right_c2bf(space) ⊠ bc.val)
end

boundary_width(op::WeightedInterpolateC2F, ::Extrapolate) = 1

function stencil_left_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    weight_field,
    value_field,
)
    @assert idx == left_face_boundary_idx(space)
    a⁺ = getidx(arg, loc, idx + half)
    a⁺
end

function stencil_right_boundary(
    ::WeightedInterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    weight_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = getidx(value_field, loc, idx - half)
    a⁻
end

"""
    LeftBiasedC2F(;boundaries)

Interpolate from the left. Only the left boundary condition should be set:
- [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
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

stencil_interior_width(::LeftBiasedC2F) = ((-half, -half),)

stencil_interior(::LeftBiasedC2F, loc, idx, arg) = getidx(arg, loc, idx - half)

boundary_width(op::LeftBiasedC2F, ::SetValue) = 1

function stencil_left_boundary(::LeftBiasedC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end

"""
    RightBiasedC2F(;boundaries)

Interpolate from the right. Only the right boundary condition should be set:
- [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
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

stencil_interior_width(::RightBiasedC2F) = ((half, half),)

stencil_interior(::RightBiasedC2F, loc, idx, arg) = getidx(arg, loc, idx + half)

boundary_width(op::RightBiasedC2F, ::SetValue) = 1

function stencil_right_boundary(::RightBiasedC2F, bc::SetValue, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    bc.val
end

abstract type AdvectionOperator <: FiniteDifferenceOperator end

return_eltype(::AdvectionOperator, velocity, arg) = eltype(arg)

"""
    UpwindBiasedProductC2F(;boundaries)

TODO:
"""
struct UpwindBiasedProductC2F{BCS} <: AdvectionOperator
    bcs::BCS
end
UpwindBiasedProductC2F(; kwargs...) = UpwindBiasedProductC2F(NamedTuple(kwargs))

return_eltype(::UpwindBiasedProductC2F, V, A) =
    Geometry.Contravariant3Vector{eltype(eltype(V))}

function return_space(
    ::UpwindBiasedProductC2F,
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    field_space::Spaces.CenterFiniteDifferenceSpace,
)
    velocity_space
end

function return_space(
    ::UpwindBiasedProductC2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    field_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    velocity_space
end

stencil_interior_width(::UpwindBiasedProductC2F) = ((0, 0), (-half, half))

function upwind_biased_product(v, a⁻, a⁺)
    RecursiveApply.rdiv(
        ((v ⊞ RecursiveApply.rmap(abs, v)) ⊠ a⁻) ⊞
        ((v ⊟ RecursiveApply.rmap(abs, v)) ⊠ a⁺),
        2,
    )
end

function stencil_interior(
    ::UpwindBiasedProductC2F,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, idx, value_field)
    a⁺ = stencil_interior(RightBiasedC2F(), loc, idx, value_field)
    vᶠ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, a⁺))
end

boundary_width(op::UpwindBiasedProductC2F, ::SetValue) = 1

function stencil_left_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_face_boundary_idx(space)
    aᴸᴮ = bc.val
    a⁺ = stencil_interior(RightBiasedC2F(), loc, idx, value_field)
    vᶠ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, aᴸᴮ, a⁺))
end

function stencil_right_boundary(
    ::UpwindBiasedProductC2F,
    bc::SetValue,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_face_boundary_idx(space)
    a⁻ = stencil_interior(LeftBiasedC2F(), loc, idx, value_field)
    aᴿᴮ = bc.val
    vᶠ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    return Geometry.Contravariant3Vector(upwind_biased_product(vᶠ, a⁻, aᴿᴮ))
end

boundary_width(op::UpwindBiasedProductC2F, ::Extrapolate) = 1

function stencil_left_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_face_boundary_idx(space)
    stencil_interior(op, loc, idx + 1, velocity_field, value_field)
end

function stencil_right_boundary(
    op::UpwindBiasedProductC2F,
    ::Extrapolate,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_face_boundary_idx(space)
    stencil_interior(op, loc, idx - 1, velocity_field, value_field)
end


struct AdvectionF2F{BCS} <: AdvectionOperator
    bcs::BCS
end

AdvectionF2F(; kwargs...) = AdvectionF2F(NamedTuple(kwargs))

stencil_interior_width(::AdvectionF2F) = ((0, 0), (-1, 1))

function return_space(
    ::AdvectionF2F,
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    field_space::Spaces.FaceFiniteDifferenceSpace,
)
    field_space
end

function return_space(
    ::AdvectionF2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    field_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
)
    field_space
end

function stencil_interior(::AdvectionF2F, loc, idx, velocity_field, value_field)
    space = axes(value_field)
    θ⁺ = getidx(value_field, loc, idx + 1)
    θ⁻ = getidx(value_field, loc, idx - 1)
    w³ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx),
        Geometry.LocalGeometry(space, idx),
    )
    ∂θ₃ = RecursiveApply.rdiv(θ⁺ ⊟ θ⁻, 2)
    return w³ ⊠ ∂θ₃
end


"""
    AdvectionC2C(;boundaries)

Advection operator at cell centers, for cell face velocity field `v` cell center variables `θ`:

    op = Advection(...)
    center_field .= op(v, θ)

It uses the following stencil
```math
A(v,θ)[i] = \\frac{1}{2} \\{ (θ[i+1] - θ[i]) v³[i+\\tfrac{1}{2}] + (θ[i] - θ[i-1])v³[i-\\tfrac{1}{2}]\\}
```
- [`SetValue(val)`](@ref): set the value of `θ` at the boundary face to be `val`.
- [`Extrapolate`](@ref): use the closest interior point as the boundary value
"""
struct AdvectionC2C{BCS} <: AdvectionOperator
    bcs::BCS
end

AdvectionC2C(; kwargs...) = AdvectionC2C(NamedTuple(kwargs))

stencil_interior_width(::AdvectionC2C) = ((-half, +half), (-1, 1))

function return_space(
    ::AdvectionC2C,
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    field_space::Spaces.CenterFiniteDifferenceSpace,
)
    field_space
end

function return_space(
    ::AdvectionC2C,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    field_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
)
    field_space
end


function stencil_interior(::AdvectionC2C, loc, idx, velocity_field, value_field)
    space = axes(value_field)
    θ⁺ = getidx(value_field, loc, idx + 1)
    θ = getidx(value_field, loc, idx)
    θ⁻ = getidx(value_field, loc, idx - 1)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

boundary_width(op::AdvectionC2C, ::SetValue) = 1

function stencil_left_boundary(
    ::AdvectionC2C,
    bc::SetValue,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(value_field, loc, idx + 1)
    θ = getidx(value_field, loc, idx)
    θ⁻ = bc.val
    w³⁺ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

function stencil_right_boundary(
    ::AdvectionC2C,
    bc::SetValue,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_center_boundary_idx(space)
    θ⁺ = bc.val
    θ = getidx(value_field, loc, idx)
    θ⁻ = getidx(value_field, loc, idx - 1)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    w³⁻ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    ∂θ₃⁻ = θ ⊟ θ⁻
    return RecursiveApply.rdiv((w³⁺ ⊠ ∂θ₃⁺) ⊞ (w³⁻ ⊠ ∂θ₃⁻), 2)
end

boundary_width(op::AdvectionC2C, ::Extrapolate) = 1

function stencil_left_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == left_center_boundary_idx(space)
    θ⁺ = getidx(value_field, loc, idx + 1)
    θ = getidx(value_field, loc, idx)
    w³⁺ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    ∂θ₃⁺ = θ⁺ ⊟ θ
    return (w³⁺ ⊠ ∂θ₃⁺)
end

function stencil_right_boundary(
    ::AdvectionC2C,
    ::Extrapolate,
    loc,
    idx,
    velocity_field,
    value_field,
)
    space = axes(value_field)
    @assert idx == right_center_boundary_idx(space)
    θ = getidx(value_field, loc, idx)
    θ⁻ = getidx(value_field, loc, idx - 1)
    w³⁻ = Geometry.contravariant3(
        getidx(velocity_field, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    ∂θ₃⁻ = θ ⊟ θ⁻
    return (w³⁻ ⊠ ∂θ₃⁻)
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
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end

function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    bc.val
end


abstract type GradientOperator <: FiniteDifferenceOperator end
return_eltype(::GradientOperator, arg) = Geometry.Covariant3Vector{eltype(arg)}


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

return_space(::GradientF2C, space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::GradientF2C) = ((-half, half),)

function stencil_interior(::GradientF2C, loc, idx, arg)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(arg, loc, idx + half) ⊟ getidx(arg, loc, idx - half))
end

boundary_width(op::GradientF2C, ::SetValue) = 1

function stencil_left_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    @assert idx == left_center_boundary_idx(arg)
    Geometry.Covariant3Vector(1) ⊗ (getidx(arg, loc, idx + half) ⊟ bc.val)
end

function stencil_right_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    @assert idx == right_center_boundary_idx(arg)
    Geometry.Covariant3Vector(1) ⊗ (bc.val ⊟ getidx(arg, loc, idx - half))
end

boundary_width(op::GradientF2C, ::Extrapolate) = 1

function stencil_left_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(arg)
    Geometry.transform(
        Geometery.Covariant3Axis(),
        stencil_interior(op, loc, idx + 1, arg),
        Geometry.LocalGeometry(space, idx),
    )
end
function stencil_right_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(arg)
    Geometry.transform(
        Geometry.Covariant3Axis(),
        stencil_interior(op, loc, idx - 1, arg),
        Geometry.LocalGeometry(space, idx),
    )
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

return_space(::GradientC2F, space::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::GradientC2F) = ((-half, half),)

function stencil_interior(::GradientC2F, loc, idx, arg)
    Geometry.Covariant3Vector(1) ⊗
    (getidx(arg, loc, idx + half) ⊟ getidx(arg, loc, idx - half))
end


boundary_width(op::GradientC2F, ::SetValue) = 1
function stencil_left_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    # ∂x[i] = 2(∂x[i + half] - val)
    Geometry.Covariant3Vector(2) ⊗ (getidx(arg, loc, idx + half) ⊟ bc.val)
end
function stencil_right_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    Geometry.Covariant3Vector(2) ⊗ (bc.val ⊟ getidx(arg, loc, idx - half))
end

# left / right SetGradient boundary conditions
boundary_width(op::GradientC2F, ::SetGradient) = 1
function stencil_left_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    space = axes(arg)
    # imposed flux boundary condition at left most face
    Geometry.transform(
        Geometry.Covariant3Axis(),
        bc.val,
        Geometry.LocalGeometry(space, idx),
    )
end
function stencil_right_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    space = axes(arg)
    # imposed flux boundary condition at right most face
    Geometry.transform(
        Geometry.Covariant3Axis(),
        bc.val,
        Geometry.LocalGeometry(space, idx),
    )
end

abstract type DivergenceOperator <: FiniteDifferenceOperator end
return_eltype(::DivergenceOperator, arg) =
    Geometry.divergence_result_type(eltype(arg))

"""
    DivergenceF2C(;boundaryname=boundarycondition...)

Centered-difference Divergence operator from a `FaceFiniteDifferenceSpace` to a
`CenterFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - by default, the current value at the boundary face will be used.
 - [`SetValue(val)`](@ref): calculate the Divergence assuming the value at the boundary is `val`.
 - [`Extrapolate()`](@ref): set the value at the center closest to the boundary
   to be the same as the neighbouring interior value.
"""
struct DivergenceF2C{BCS} <: DivergenceOperator
    bcs::BCS
end
DivergenceF2C(; kwargs...) = DivergenceF2C(NamedTuple(kwargs))

return_space(::DivergenceF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)

return_space(::DivergenceF2C, space::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    Spaces.CenterExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::DivergenceF2C) = ((-half, half),)

function stencil_interior(::DivergenceF2C, loc, idx, arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J)
end

boundary_width(op::DivergenceF2C, ::SetValue) = 1
function stencil_left_boundary(::DivergenceF2C, bc::SetValue, loc, idx, arg)
    @assert idx == left_center_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        bc.val,
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J)
end
function stencil_right_boundary(::DivergenceF2C, bc::SetValue, loc, idx, arg)
    @assert idx == right_center_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        bc.val,
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J)
end

boundary_width(op::DivergenceF2C, ::Extrapolate) = 1
function stencil_left_boundary(op::DivergenceF2C, ::Extrapolate, loc, idx, arg)
    @assert idx == left_center_boundary_idx(arg)
    stencil_interior(op, loc, idx + 1, arg)
end
function stencil_right_boundary(op::DivergenceF2C, ::Extrapolate, loc, idx, arg)
    @assert idx == right_center_boundary_idx(arg)
    stencil_interior(op, loc, idx - 1, arg)
end

"""
    DivergenceC2F(;boundaries)

Centered-difference Divergence operator from a `CenterFiniteDifferenceSpace` to a
`FaceFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - [`SetValue(val)`](@ref): calculate the Divergence assuming the value at the boundary is `val`.
 - [`SetDivergence(val)`](@ref): set the value of the Divergence at the boundary to be `val`.
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

stencil_interior_width(::DivergenceC2F) = ((-half, half),)

function stencil_interior(::DivergenceC2F, loc, idx, arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J)
end


boundary_width(op::DivergenceC2F, ::SetValue) = 1
function stencil_left_boundary(::DivergenceC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    # ∂x[i] = 2(∂x[i + half] - val)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx + half),
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        bc.val,
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J / 2)
end
function stencil_right_boundary(::DivergenceC2F, bc::SetValue, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    space = axes(arg)
    local_geometry = Geometry.LocalGeometry(space, idx)
    Ju³₊ = Geometry.Jcontravariant3(
        bc.val,
        Geometry.LocalGeometry(space, idx + half),
    )
    Ju³₋ = Geometry.Jcontravariant3(
        getidx(arg, loc, idx - half),
        Geometry.LocalGeometry(space, idx - half),
    )
    RecursiveApply.rdiv(Ju³₊ ⊟ Ju³₋, local_geometry.J / 2)
end

# left / right SetDivergence boundary conditions
boundary_width(op::DivergenceC2F, ::SetGradient) = 1
function stencil_left_boundary(::DivergenceC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    # imposed flux boundary condition at left most face
    bc.val
end
function stencil_right_boundary(::DivergenceC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == right_face_boundary_idx(arg)
    # imposed flux boundary condition at right most face
    bc.val
end

boundary_width(obj, loc) = 0

boundary_width(bc::Base.Broadcast.Broadcasted{StencilStyle}, loc) =
    has_boundary(bc.f, loc) ? boundary_width(bc.f, get_boundary(bc.f, loc)) : 0

function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = map(bc.args, stencil_interior_width(bc.f)) do arg, width
        left_interor_window_idx(arg, space, loc) - width[1]
    end
    return max(maximum(arg_idxs), left_idx(space) + boundary_width(bc, loc))
end

function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = map(bc.args, stencil_interior_width(bc.f)) do arg, width
        right_interor_window_idx(arg, space, loc) - width[2]
    end
    return min(minimum(arg_idxs), right_idx(space) - boundary_width(bc, loc))
end

function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    maximum(arg -> left_interor_window_idx(arg, space, loc), bc.args)
end
function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    minimum(arg -> right_interor_window_idx(arg, space, loc), bc.args)
end

function left_interor_window_idx(field::Field, _, loc::LeftBoundaryWindow)
    left_idx(axes(field))
end

function right_interor_window_idx(field::Field, _, loc::RightBoundaryWindow)
    right_idx(axes(field))
end

function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::LeftBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    left_idx(axes(bc))
end

function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::RightBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    right_idx(axes(bc))
end

function left_interor_window_idx(_, space, loc::LeftBoundaryWindow)
    left_idx(space)
end

function right_interor_window_idx(_, space, loc::RightBoundaryWindow)
    right_idx(space)
end

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::Interior,
    idx,
)
    stencil_interior(bc.f, loc, idx, bc.args...)
end

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::LeftBoundaryWindow,
    idx,
)
    op = bc.f
    space = axes(bc)
    if idx < left_idx(space) + boundary_width(bc, loc)
        stencil_left_boundary(op, get_boundary(op, loc), loc, idx, bc.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bc.args...)
    end
end

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::RightBoundaryWindow,
    idx,
)
    op = bc.f
    space = axes(bc)
    if idx > (right_idx(space) - boundary_width(bc, loc))
        stencil_right_boundary(op, get_boundary(op, loc), loc, idx, bc.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bc.args...)
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

function getidx(
    bc::Fields.CenterFiniteDifferenceField,
    ::Location,
    idx::Integer,
)
    Fields.field_values(bc)[idx]
end

function getidx(bc::Fields.FaceFiniteDifferenceField, ::Location, idx::PlusHalf)
    Fields.field_values(bc)[idx.i + 1]
end

getidx(field::Fields.Field, ::Location, idx) = error(
    "Invalid index type $(typeof(idx)) for field on space $(summary(axes(field)))",
)
getidx(field::Base.Broadcast.Broadcasted{StencilStyle}, ::Location, idx) =
    error(
        "Invalid index type $(typeof(idx)) for field on space $(summary(axes(field)))",
    )

getidx(scalar, ::Location, idx) = scalar

function getidx(bc::Base.Broadcast.Broadcasted, loc::Location, idx)
    args = map(arg -> getidx(arg, loc, idx), bc.args)
    bc.f(args...)
end

function setidx!(bc::Fields.CenterFiniteDifferenceField, idx::Integer, val)
    Fields.field_values(bc)[idx] = val
end

function setidx!(bc::Fields.FaceFiniteDifferenceField, idx::PlusHalf, val)
    Fields.field_values(bc)[idx.i + 1] = val
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    Base.Broadcast.broadcasted(StencilStyle(), op, args...)
end

function Base.Broadcast.broadcasted(
    ::StencilStyle,
    op::FiniteDifferenceOperator,
    args...,
)
    ax = return_space(op, map(axes, args)...)
    Base.Broadcast.Broadcasted{StencilStyle}(op, args, ax)
end

Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{StencilStyle}) = bc
Base.Broadcast._broadcast_getindex_eltype(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
) = eltype(bc)

# check that inferred output field space is equal to dest field space
@inline function Base.Broadcast.materialize!(
    ::Base.Broadcast.BroadcastStyle,
    dest::Fields.Field,
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractStencilStyle}
    dest_space, result_space = axes(dest), axes(bc)
    if result_space !== dest_space
        error(
            "dest space `$(summary(dest_space))` is not equal to the inferred broadcasted result space `$(summary(result_space))`",
        )
    end
    # the default Base behavior is to instantiate a Broadcasted object with the same axes as the dest
    return copyto!(
        dest,
        Base.Broadcast.instantiate(
            Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, dest_space),
        ),
    )
end

function column(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...,
) where {Style <: AbstractStencilStyle}
    _args = map(a -> column(a, inds...), bc.args)
    _axes = column(axes(bc), inds...)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

function Base.similar(
    bc::Base.Broadcast.Broadcasted{S},
    ::Type{Eltype},
) where {Eltype, S <: AbstractStencilStyle}
    sp = axes(bc)
    return Field(Eltype, sp)
end

function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
) where {S <: AbstractStencilStyle}
    space = axes(bc)
    local_geometry = Spaces.local_geometry_data(space)
    (Ni, Nj, _, _, Nh) = size(local_geometry)
    for h in 1:Nh, j in 1:Nj, i in 1:Ni
        column_field_out = column(field_out, i, j, h)
        column_bc = column(bc, i, j, h)
        apply_stencil!(column_field_out, column_bc)
    end
    return field_out
end

function apply_stencil!(field_out, bc)
    space = axes(bc)
    lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()
    li = left_idx(space)
    lw = left_interor_window_idx(bc, space, lbw)
    ri = right_idx(space)
    rw = right_interor_window_idx(bc, space, rbw)
    @assert li <= lw <= rw <= ri
    for idx in li:(lw - 1)
        setidx!(field_out, idx, getidx(bc, lbw, idx))
    end
    for idx in lw:rw
        setidx!(field_out, idx, getidx(bc, Interior(), idx))
    end
    for idx in (rw + 1):ri
        setidx!(field_out, idx, getidx(bc, rbw, idx))
    end
    return field_out
end
