(AxisVector{T, A, SVector{1, T}} where {T})(
    a::Real,
    ::LocalGeometry,
) where {A} = AxisVector(A.instance, SVector(a))

# standard conversions
ContravariantVector(u::ContravariantVector, ::LocalGeometry) = u
CovariantVector(u::CovariantVector, ::LocalGeometry) = u
LocalVector(u::LocalVector, ::LocalGeometry) = u

ContravariantVector(
    u::LocalVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.∂ξ∂x * u
LocalVector(
    u::ContravariantVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.∂x∂ξ * u
CovariantVector(
    u::LocalVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.∂x∂ξ' * u
LocalVector(
    u::CovariantVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.∂ξ∂x' * u

# Converting to specific dimension types
(::Type{<:ContravariantVector{<:Any, I}})(
    u::ContravariantVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

(::Type{<:ContravariantVector{<:Any, I}})(
    u::ContravariantVector,
    ::LocalGeometry,
) where {I} = transform(ContravariantAxis{I}(), u)

(::Type{<:ContravariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} =
    transform(ContravariantAxis{I}(), ContravariantVector(u, local_geometry))

(::Type{<:CovariantVector{<:Any, I}})(
    u::CovariantVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

(::Type{<:CovariantVector{<:Any, I}})(
    u::CovariantVector,
    ::LocalGeometry,
) where {I} = transform(CovariantAxis{I}(), u)

(::Type{<:CovariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = transform(CovariantAxis{I}(), CovariantVector(u, local_geometry))

(::Type{<:LocalVector{<:Any, I}})(
    u::LocalVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

(::Type{<:LocalVector{<:Any, I}})(u::LocalVector, ::LocalGeometry) where {I} =
    transform(LocalAxis{I}(), u)

(::Type{<:LocalVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = transform(LocalAxis{I}(), LocalVector(u, local_geometry))

# Generic N-axis conversion functions,
# Convert to specific local geometry dimension then convert vector type
LocalVector(u::CovariantVector, local_geometry::LocalGeometry{I}) where {I} =
    transform(LocalAxis{I}(), transform(CovariantAxis{I}(), u), local_geometry)

LocalVector(
    u::ContravariantVector,
    local_geometry::LocalGeometry{I},
) where {I} = transform(
    LocalAxis{I}(),
    transform(ContravariantAxis{I}(), u),
    local_geometry,
)

CovariantVector(u::LocalVector, local_geometry::LocalGeometry{I}) where {I} =
    transform(CovariantAxis{I}(), transform(LocalAxis{I}(), u), local_geometry)

CovariantVector(
    u::ContravariantVector,
    local_geometry::LocalGeometry{I},
) where {I} = transform(
    CovariantAxis{I}(),
    transform(ContravariantAxis{I}(), u),
    local_geometry,
)

ContravariantVector(
    u::LocalVector,
    local_geometry::LocalGeometry{I},
) where {I} = transform(
    ContravariantAxis{I}(),
    transform(LocalAxis{I}(), u),
    local_geometry,
)

ContravariantVector(
    u::CovariantVector,
    local_geometry::LocalGeometry{I},
) where {I} = transform(
    ContravariantAxis{I}(),
    transform(CovariantAxis{I}(), u),
    local_geometry,
)

# In order to make curls and cross products work in 2D, we define the 3rd
# dimension to be orthogonal to the exisiting dimensions, and have unit length
# (so covariant, contravariant and local axes are equal)
ContravariantVector(u::LocalVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(Contravariant3Axis(), components(u))
ContravariantVector(u::CovariantVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(Contravariant3Axis(), components(u))

CovariantVector(u::LocalVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(Covariant3Axis(), components(u))
CovariantVector(u::ContravariantVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(Covariant3Axis(), components(u))

LocalVector(u::CovariantVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(WAxis(), components(u))
LocalVector(u::ContravariantVector{<:Any, (3,)}, ::LocalGeometry{(1, 2)}) =
    AxisVector(WAxis(), components(u))


@inline covariant1(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₁
@inline covariant2(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₂
@inline covariant3(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₃

@inline contravariant1(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant1Axis(), u, local_geometry)[1]
@inline contravariant2(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant2Axis(), u, local_geometry)[1]
@inline contravariant3(u::AxisVector, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant3Axis(), u, local_geometry)[1]

@inline contravariant1(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant1Axis(), u, local_geometry)[1, :]
@inline contravariant2(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant2Axis(), u, local_geometry)[1, :]
@inline contravariant3(u::Axis2Tensor, local_geometry::LocalGeometry) =
    @inbounds project(Contravariant3Axis(), u, local_geometry)[1, :]

Base.@propagate_inbounds Jcontravariant3(
    u::AxisTensor,
    local_geometry::LocalGeometry,
) = local_geometry.J * contravariant3(u, local_geometry)

# required for curl-curl
@inline covariant3(
    u::Contravariant3Vector,
    local_geometry::LocalGeometry{(1, 2)},
) = contravariant3(u, local_geometry)

# workarounds for using a Covariant12Vector/Covariant123Vector in a UW space:
@inline function LocalVector(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, v, u₃ = components(vector)
    vector2 = Covariant13Vector(u₁, u₃)
    u, w = components(transform(LocalAxis{(1, 3)}(), vector2, local_geometry))
    return UVWVector(u, v, w)
end
@inline function contravariant1(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, _, u₃ = components(vector)
    vector2 = Covariant13Vector(u₁, u₃)
    return transform(Contravariant13Axis(), vector2, local_geometry).u¹
end
@inline function contravariant3(
    vector::CovariantVector{<:Any, (1, 2)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, _ = components(vector)
    vector2 = Covariant13Vector(u₁, zero(u₁))
    return transform(Contravariant13Axis(), vector2, local_geometry).u³
end
@inline function ContravariantVector(
    vector::CovariantVector{<:Any, (1, 2)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, v = components(vector)
    vector2 = Covariant1Vector(u₁)
    vector3 = transform(
        ContravariantAxis{(1, 3)}(),
        transform(CovariantAxis{(1, 3)}(), vector2),
        local_geometry,
    )
    u¹, u³ = components(vector3)
    return Contravariant123Vector(u¹, v, u³)
end

"""
    transform(axis, V[, local_geometry])

Transform the first axis of the vector or tensor `V` to `axis`. This will throw
an error if the conversion is not exact.

The conversion rules are defined as:

- `v::Covariant` => `Local`:     `∂ξ∂x' * v`
- `v::Local` => `Contravariant`: `∂ξ∂x  * v`
- `v::Contravariant` => `Local`: `∂x∂ξ  * v`
- `v::Local` => `Covariant`:     `∂x∂ξ' * v`
- `v::Covariant` => `Contravariant`:  `∂ξ∂x * (∂ξ∂x' * v) = gⁱʲ * v`
- `v::Contravariant` => `Covariant`:  `∂x∂ξ' * ∂x∂ξ * v   = gᵢⱼ * v`

# Example
Consider the conversion from a  `Covariant12Vector` to a `Contravariant12Axis`.
Mathematically, we can write this as

```
[ v¹ ]   [g¹¹  g¹²  g¹³ ]   [ v₁ ]
[ v² ] = [g²¹  g²²  g²³ ] * [ v₂ ]
[ v³ ]   [g³¹  g³²  g³³ ]   [ 0  ]
```

`project` will drop v³ term no matter what the value is, i.e. it returns

```
[ v¹ ]   [g¹¹ v₁  + g¹² v₂ ]
[ v² ] = [g²¹ v₁  + g²² v₂ ]
[ 0  ]   [<drops this>]
```

`transform` will drop the v³ term, but throw an error if it is non-zero (i.e. if
the conversion is not exact)

```
[ v¹ ]   [g¹¹ v₁  + g¹² v₂ ]
[ v² ] = [g²¹ v₁  + g²² v₂ ]
[ 0  ]   [<asserts g²³ v₁  + g²³ v₂ == 0>]
```

"""
function transform end

"""
    project(axis, V[, local_geometry])

Project the first axis component of the vector or tensor `V` to `axis`

This is equivalent to [`transform`](@ref), but will not throw an error if the
conversion is not exact.
"""
function project end

for op in (:transform, :project)
    @eval begin
        # Covariant <-> Cartesian
        @inline $op(
            ax::CartesianAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' * $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' * $op(dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )
        @inline $op(
            ax::LocalAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' * $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::LocalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' * $op(dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )

        # Contravariant <-> Cartesian
        @inline $op(
            ax::ContravariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x * $op(dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )
        @inline $op(
            ax::CartesianAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ * $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )
        @inline $op(
            ax::ContravariantAxis,
            v::LocalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x * $op(dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )

        @inline $op(
            ax::LocalAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ * $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        # Covariant <-> Contravariant
        @inline $op(
            ax::ContravariantAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            local_geometry.∂ξ∂x' *
            $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        @inline $op(
            ax::CovariantAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' *
            local_geometry.∂x∂ξ *
            $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        @inline $op(ato::CovariantAxis, v::CovariantTensor, ::LocalGeometry) =
            $op(ato, v)
        @inline $op(
            ato::ContravariantAxis,
            v::ContravariantTensor,
            ::LocalGeometry,
        ) = $op(ato, v)
        @inline $op(ato::CartesianAxis, v::CartesianTensor, ::LocalGeometry) =
            $op(ato, v)
        @inline $op(ato::LocalAxis, v::LocalTensor, ::LocalGeometry) =
            $op(ato, v)
    end
end


"""
    divergence_result_type(V)

The return type when taking the divergence of a field of `V`.

Required for statically infering the result type of the divergence operation for `AxisVector` subtypes.
"""
@inline divergence_result_type(::Type{V}) where {V <: AxisVector} = eltype(V)

# this isn't quite right: it only is true when the Christoffel symbols are zero
@inline divergence_result_type(
    ::Type{Axis2Tensor{FT, Tuple{A1, A2}, S}},
) where {FT, A1, A2 <: LocalAxis, S <: StaticMatrix{S1, S2}} where {S1, S2} =
    AxisVector{FT, A2, SVector{S2, FT}}

"""
    gradient_result_type(Val(I), V)

The return type when taking the gradient along dimension `I` of a field `V`.

Required for statically infering the result type of the gradient operation for `AxisVector` subtypes.
    """
@inline function gradient_result_type(
    ::Val{I},
    ::Type{V},
) where {I, V <: Number}
    N = length(I)
    AxisVector{V, CovariantAxis{I}, SVector{N, V}}
end
@inline function gradient_result_type(
    ::Val{I},
    ::Type{V},
) where {I, V <: Geometry.AxisVector{T, A, SVector{N, T}}} where {T, A, N}
    M = length(I)
    Axis2Tensor{T, Tuple{CovariantAxis{I}, A}, SMatrix{M, N, T, M * N}}
end

"""
    curl_result_type(Val(I), Val(L), V)

The return type when taking the curl along dimensions `I` of a field of eltype `V`, defined on dimensions `L`

Required for statically infering the result type of the curl operation for `AxisVector` subtypes.
Curl is only defined for `CovariantVector`` field input types.

| Input Vector | Operator direction | Curl output vector |
| ------------ | ------------------ | -------------- |
|  Covariant12Vector | (1,2) | Contravariant3Vector |
|  Covariant3Vector | (1,2) | Contravariant12Vector |
|  Covariant123Vector | (1,2) | Contravariant123Vector |
|  Covariant1Vector | (1,) | Contravariant1Vector |
|  Covariant2Vector | (1,) | Contravariant3Vector |
|  Covariant3Vector | (1,) | Contravariant2Vector |
|  Covariant12Vector | (3,) | Contravariant12Vector |
|  Covariant1Vector | (3,) | Contravariant2Vector |
|  Covariant2Vector | (3,) | Contravariant1Vector |
|  Covariant3Vector | (3,) | Contravariant3Vector |
"""
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Covariant3Vector{FT}},
) where {FT} = Contravariant12Vector{FT}
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Covariant12Vector{FT}},
) where {FT} = Contravariant3Vector{FT}
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Covariant123Vector{FT}},
) where {FT} = Contravariant123Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant1Vector{FT}}) where {FT} =
    Contravariant1Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant2Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(
    ::Val{(3,)},
    ::Type{Covariant12Vector{FT}},
) where {FT} = Contravariant12Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant1Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant2Vector{FT}}) where {FT} =
    Contravariant1Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}

_norm_sqr(x, local_geometry::LocalGeometry) =
    sum(x -> _norm_sqr(x, local_geometry), x)
_norm_sqr(x::Number, ::LocalGeometry) = LinearAlgebra.norm_sqr(x)
_norm_sqr(x::AbstractArray, ::LocalGeometry) = LinearAlgebra.norm_sqr(x)

function _norm_sqr(uᵢ::CovariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(LocalVector(uᵢ, local_geometry))
end

function _norm_sqr(uᵢ::ContravariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(LocalVector(uᵢ, local_geometry))
end

_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(1,)}) =
    LinearAlgebra.norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(3,)}) =
    LinearAlgebra.norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(1, 3)}) =
    LinearAlgebra.norm_sqr(u.u²)

_norm_sqr(u::Contravariant3Vector, ::LocalGeometry{(1,)}) =
    LinearAlgebra.norm_sqr(u.u³)
_norm_sqr(u::Contravariant3Vector, ::LocalGeometry{(1, 2)}) =
    LinearAlgebra.norm_sqr(u.u³)



_norm(u::AxisVector, local_geometry::LocalGeometry) =
    sqrt(_norm_sqr(u, local_geometry))

_cross(u::AxisVector, v::AxisVector, local_geometry::LocalGeometry) = _cross(
    ContravariantVector(u, local_geometry),
    ContravariantVector(v, local_geometry),
    local_geometry,
)

_cross(
    x::ContravariantVector,
    y::ContravariantVector,
    local_geometry::LocalGeometry,
) =
    local_geometry.J * Covariant123Vector(
        x.u² * y.u³ - x.u³ * y.u²,
        x.u³ * y.u¹ - x.u¹ * y.u³,
        x.u¹ * y.u² - x.u² * y.u¹,
    )
_cross(
    x::Contravariant12Vector,
    y::Contravariant12Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant3Vector(x.u¹ * y.u² - x.u² * y.u¹)
_cross(
    x::Contravariant2Vector,
    y::Contravariant1Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant3Vector(-x.u² * y.u¹)
_cross(
    x::Contravariant12Vector,
    y::Contravariant3Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant12Vector(x.u² * y.u³, -x.u¹ * y.u³)
_cross(
    x::Contravariant3Vector,
    y::Contravariant12Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant12Vector(-x.u³ * y.u², x.u³ * y.u¹)

_cross(
    x::Contravariant2Vector,
    y::Contravariant3Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant1Vector(x.u² * y.u³)


_cross(u::CartesianVector, v::CartesianVector, ::LocalGeometry) =
    LinearAlgebra.cross(u, v)
_cross(u::LocalVector, v::LocalVector, ::LocalGeometry) =
    LinearAlgebra.cross(u, v)
