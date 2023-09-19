# to be deprecated in future

# these were renamed
const LeftBiasedC2F = LeftBiased1stOrderC2F
const RightBiasedC2F = RightBiased1stOrderC2F
const FirstOrderOneSided = OneSided1stOrder
const ThirdOrderOneSided = OneSided3rdOrder

# F2C aren't used

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
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_interior(
        Upwind1stOrderC2F(),
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
end

boundary_width(::UpwindBiasedProductC2F, ::AbstractBoundaryCondition) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::UpwindBiasedProductC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_left_boundary(
        Upwind1stOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::UpwindBiasedProductC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_right_boundary(
        Upwind1stOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
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
- [`OneSided1stOrder(x₀)`](@ref): uses the first-order downwind scheme to compute `x` on the left boundary,
  and the first-order upwind scheme to compute `x` on the right boundary.
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
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_interior(
        Upwind3rdOrderC2F(),
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
end

boundary_width(::Upwind3rdOrderBiasedProductC2F, ::AbstractBoundaryCondition) =
    2


Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_left_boundary(
        Upwind3rdOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderBiasedProductC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    u = stencil_right_boundary(
        Upwind3rdOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        Geometry.Contravariant3Vector(v³),
        arg,
    )
    return Geometry.Contravariant3Vector(v³ * u)
end
