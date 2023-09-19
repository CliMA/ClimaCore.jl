"""
    L = LeftBiased1stOrderC2F(;boundaries)
    L.(x)

Interpolate a center-value field to a face-valued field from the left.
```math
L(x)[i] = x[i-\\tfrac{1}{2}]
```

Only the left boundary condition should be set. Currently supported are:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
L(x)[\\tfrac{1}{2}] = x_0
```
- [`Extrapolate()`](@ref): use the right-biased interior value.
```math
L(x)[\\tfrac{1}{2}] = x[1]
```
"""
struct LeftBiased1stOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
LeftBiased1stOrderC2F(; kwargs...) = LeftBiased1stOrderC2F(NamedTuple(kwargs))

return_space(
    ::LeftBiased1stOrderC2F,
    space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::LeftBiased1stOrderC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::LeftBiased1stOrderC2F, arg) = ((-half, -half),)
Base.@propagate_inbounds stencil_interior(
    ::LeftBiased1stOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx - half, hidx)

left_interior_idx(
    space::AbstractSpace,
    ::LeftBiased1stOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space) + 1
right_interior_idx(
    space::AbstractSpace,
    ::LeftBiased1stOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space)

Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased1stOrderC2F,
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
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased1stOrderC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    getidx(space, arg, loc, idx + half, hidx)
end



"""
    L = LeftBiased3rdOrderC2F(;boundaries)
    L.(x)

Interpolate a center-value field to a face-valued field from the left, using a
3rd-order reconstruction.
```math
L(x)[i] =  \\left(-2 x[i-\\tfrac{3}{2}] + 10 x[i-\\tfrac{1}{2}] + 4 x[i+\\tfrac{1}{2}] \\right) / 12
```

Only the left boundary condition should be set. Currently supported are:

- [`OneSided1stOrder()`](@ref): use the one-sided 1st-order reconstruction.
```math
L(x)[1+\\tfrac{1}{2}] = x[1]
```
- [`OneSided3rdOrder()`](@ref): use the [`RightBiased3rdOrderC2F`](@ref)
  reconstruction for ``L(x)[1+\\tfrac{1}{2}]``.
  
!!! note
    The boundary conditions only specify the first interior point. Actual boundary values
    remain undefined and need to be handled separately.
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
Base.@propagate_inbounds function stencil_interior(
    ::LeftBiased3rdOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    FT = Spaces.undertype(space)

    FT(4 / 12) ⊠ getidx(space, arg, loc, idx + half, hidx) ⊞
    FT(10 / 12) ⊠ getidx(space, arg, loc, idx - half, hidx) ⊟
    FT(2 / 12) ⊠ getidx(space, arg, loc, idx - half - 1, hidx)
end

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
    bc::OneSided1stOrder,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space) + 1
    stencil_interior(LeftBiased1stOrderC2F(), loc, space, idx, hidx, arg)
end
Base.@propagate_inbounds function stencil_left_boundary(
    ::LeftBiased3rdOrderC2F,
    bc::OneSided3rdOrder,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == left_face_boundary_idx(space) + 1
    stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
end


"""
    R = RightBiased1stOrderC2F(;boundaries)
    R.(x)

Interpolate a center-valued field to a face-valued field from the right.
```math
R(x)[i] = x[i+\\tfrac{1}{2}]
```

Only the right boundary condition should be set. Currently supported:
- [`SetValue(x₀)`](@ref): set the value to be `x₀` on the boundary.
```math
R(x)[n+\\tfrac{1}{2}] = x_0
```
- [`Extrapolate()`](@ref): use the left-biased interior value.
```math
R(x)[n+\\tfrac{1}{2}] = x[n]
```
"""
struct RightBiased1stOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
RightBiased1stOrderC2F(; kwargs...) = RightBiased1stOrderC2F(NamedTuple(kwargs))

return_space(
    ::RightBiased1stOrderC2F,
    space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(space)
return_space(
    ::RightBiased1stOrderC2F,
    space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(space)

stencil_interior_width(::RightBiased1stOrderC2F, arg) = ((half, half),)
Base.@propagate_inbounds stencil_interior(
    ::RightBiased1stOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
) = getidx(space, arg, loc, idx + half, hidx)

left_interior_idx(
    space::AbstractSpace,
    ::RightBiased1stOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = left_idx(space)
right_interior_idx(
    space::AbstractSpace,
    ::RightBiased1stOrderC2F,
    ::AbstractBoundaryCondition,
    arg,
) = right_idx(space) - 1

Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased1stOrderC2F,
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
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased1stOrderC2F,
    bc::Extrapolate,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    getidx(space, arg, loc, idx - half, hidx)
end

"""
    R = RightBiased3rdOrderC2F(;boundaries)
    R.(x)

Interpolate a center-valued field to a face-valued field from the right, using a 3rd-order reconstruction.
```math
R(x)[i] = \\left(4 x[i-\\tfrac{1}{2}] + 10 x[i+\\tfrac{1}{2}] -2 x[i+\\tfrac{3}{2}]  \\right) / 12
```

Only the left boundary condition should be set. Currently supported are:

- [`OneSided1stOrder()`](@ref): use the one-sided 1st-order reconstruction.
```math
R(x)[n-\\tfrac{1}{2}] = x[1]
```
- [`OneSided3rdOrder()`](@ref): use the [`RightBiased3rdOrderC2F`](@ref)
  reconstruction for ``R(x)[n-\\tfrac{1}{2}]``
  
!!! note
    The boundary conditions only specify the first interior point. Actual boundary values
    remain undefined and need to be handled separately.
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
Base.@propagate_inbounds function stencil_interior(
    ::RightBiased3rdOrderC2F,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    FT = Spaces.undertype(space)

    FT(4 / 12) ⊠ getidx(space, arg, loc, idx - half, hidx) ⊞
    FT(10 / 12) ⊠ getidx(space, arg, loc, idx + half, hidx) ⊟
    FT(2 / 12) ⊠ getidx(space, arg, loc, idx + half + 1, hidx)
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderC2F,
    bc::OneSided1stOrder,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space) - 1
    stencil_interior(RightBiased1stOrderC2F(), loc, space, idx, hidx, arg)
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::RightBiased3rdOrderC2F,
    bc::OneSided3rdOrder,
    loc,
    space,
    idx,
    hidx,
    arg,
)
    @assert idx == right_face_boundary_idx(space) - 1
    stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
end

"""
    U = Upwind1stOrderC2F(;boundaries)
    U.(v, x)

Interpolate the center-valued field `x` to cell faces by 1st-order upwinding `x`
according to the direction of `v`.

More precisely, it is computed based on the sign of the 3rd contravariant
component, and it returns a `Contravariant3Vector`:
```math
U(\\boldsymbol{v},x)[i] = \\begin{cases}
  x[i-\\tfrac{1}{2}]\\boldsymbol{e}_3 \\textrm{, if } v^3[i] > 0 \\\\
  x[i+\\tfrac{1}{2}]\\boldsymbol{e}_3 \\textrm{, if } v^3[i] < 0
  \\end{cases}
```
where ``\\boldsymbol{e}_3`` is the 3rd covariant basis vector.

Supported boundary conditions are:
- [`SetValue(x₀)`](@ref): set the value of `x` to be `x₀` in a hypothetical
  ghost cell on the other side of the boundary. On the left boundary the stencil
  is
  ```math
  U(\\boldsymbol{v},x)[\\tfrac{1}{2}] = \\begin{cases}
    x_0  \\boldsymbol{e}_3 \\textrm{, if }  v^3[\\tfrac{1}{2}] > 0 \\\\
    x[1] \\boldsymbol{e}_3 \\textrm{, if }  v^3[\\tfrac{1}{2}] < 0
    \\end{cases}
  ```
- [`Extrapolate()`](@ref): set the value of `x` to be the same as the closest
  interior point. On the left boundary, the stencil is
  ```math
  U(\\boldsymbol{v},x)[\\tfrac{1}{2}] = U(\\boldsymbol{v},x)[1 + \\tfrac{1}{2}]
  ```
"""
struct Upwind1stOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
Upwind1stOrderC2F(; kwargs...) = Upwind1stOrderC2F(NamedTuple(kwargs))

return_eltype(::Upwind1stOrderC2F, v, arg) = eltype(arg)

return_space(
    ::Upwind1stOrderC2F,
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = velocity_space
return_space(
    ::Upwind1stOrderC2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = velocity_space

stencil_interior_width(::Upwind1stOrderC2F, velocity, arg) =
    ((0, 0), (-half, half))


Base.@propagate_inbounds function stencil_interior(
    ::Upwind1stOrderC2F,
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
    a⁺ = stencil_interior(RightBiased1stOrderC2F(), loc, space, idx, hidx, arg)
    a⁻ = stencil_interior(LeftBiased1stOrderC2F(), loc, space, idx, hidx, arg)
    # signbit(v³) == true  if  v³ < 0
    return signbit(v³) ? a⁺ : a⁻
end

boundary_width(::Upwind1stOrderC2F, ::AbstractBoundaryCondition) = 1

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind1stOrderC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == left_face_boundary_idx(space)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ = stencil_interior(RightBiased1stOrderC2F(), loc, space, idx, hidx, arg)
    a⁻ = stencil_left_boundary(
        LeftBiased1stOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        arg,
    )
    return signbit(v³) ? a⁺ : a⁻
end

Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind1stOrderC2F,
    bc::AbstractBoundaryCondition,
    loc,
    space,
    idx,
    hidx,
    velocity,
    arg,
)
    @assert idx == right_face_boundary_idx(space)
    v³ = Geometry.contravariant3(
        getidx(space, velocity, loc, idx, hidx),
        Geometry.LocalGeometry(space, idx, hidx),
    )
    a⁺ = stencil_right_boundary(
        RightBiased1stOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        arg,
    )
    a⁻ = stencil_interior(LeftBiased1stOrderC2F(), loc, space, idx, hidx, arg)
    return signbit(v³) ? a⁺ : a⁻
end


"""
    U = Upwind3rdOrderC2F(;boundaries)
    U.(v, x)

Interpolate the center-valued field `x` to cell faces by 3rd-order upwinding `x`
according to the direction of `v`. The stencil is
```math
U(v,x)[i] = \\begin{cases}
  \\left(-2 x[i-\\tfrac{3}{2}] + 10 x[i-\\tfrac{1}{2}] + 4 x[i+\\tfrac{1}{2}] \\right) / 12  \\textrm{, if } v[i] > 0 \\\\
  \\left(4 x[i-\\tfrac{1}{2}] + 10 x[i+\\tfrac{1}{2}] -2 x[i+\\tfrac{3}{2}]  \\right) / 12  \\textrm{, if } v[i] < 0
  \\end{cases}
```
This stencil is based on [WickerSkamarock2002](@cite), eq. 4(a).

Supported boundary conditions are:
- [`OneSided1stOrder(x₀)`](@ref): uses the first-order downwind scheme to compute `x` on the left boundary,
  and the first-order upwind scheme to compute `x` on the right boundary.
- [`OneSided3rdOrder(x₀)`](@ref): uses the third-order downwind reconstruction to compute `x` on the left boundary,
and the third-order upwind reconstruction to compute `x` on the right boundary.

!!! note
    These boundary conditions do not define the value at the actual boundary faces, and so this operator cannot be materialized directly: it needs to be composed with another operator that does not make use of this value, e.g. a [`DivergenceF2C`](@ref) operator, with a [`SetValue`](@ref) boundary.
"""
struct Upwind3rdOrderC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
Upwind3rdOrderC2F(; kwargs...) = Upwind3rdOrderC2F(NamedTuple(kwargs))

return_eltype(::Upwind3rdOrderC2F, V, arg) = eltype(arg)

return_space(
    ::Upwind3rdOrderC2F,
    velocity_space::Spaces.FaceFiniteDifferenceSpace,
    arg_space::Spaces.CenterFiniteDifferenceSpace,
) = velocity_space
return_space(
    ::Upwind3rdOrderC2F,
    velocity_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    arg_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = velocity_space

stencil_interior_width(::Upwind3rdOrderC2F, velocity, arg) =
    ((0, 0), (-half - 1, half + 1))

Base.@propagate_inbounds function stencil_interior(
    ::Upwind3rdOrderC2F,
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
    a⁺ = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    a⁻ = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    return signbit(v³) ? a⁺ : a⁻
end

boundary_width(::Upwind3rdOrderC2F, ::AbstractBoundaryCondition) = 2

Base.@propagate_inbounds function stencil_left_boundary(
    ::Upwind3rdOrderC2F,
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
    a⁺ = stencil_interior(RightBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    a⁻ = stencil_left_boundary(
        LeftBiased3rdOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        arg,
    )
    return signbit(v³) ? a⁺ : a⁻
end
Base.@propagate_inbounds function stencil_right_boundary(
    ::Upwind3rdOrderC2F,
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
    a⁺ = stencil_right_boundary(
        RightBiased3rdOrderC2F(),
        bc,
        loc,
        space,
        idx,
        hidx,
        arg,
    )
    a⁻ = stencil_interior(LeftBiased3rdOrderC2F(), loc, space, idx, hidx, arg)
    return signbit(v³) ? a⁺ : a⁻
end
