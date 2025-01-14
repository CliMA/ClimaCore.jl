(AxisVector{T, A, SVector{1, T}} where {T})(
    a::Real,
    ::LocalGeometry,
) where {A} = AxisVector(A.instance, SVector(a))

# standard conversions
ContravariantVector(u::ContravariantVector, ::LocalGeometry) = u
CovariantVector(u::CovariantVector, ::LocalGeometry) = u
LocalVector(u::LocalVector, ::LocalGeometry) = u
# conversions between Covariant/Contravariant vectors and local vectors
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

# conversions between Covariant and Contravariant vectors
Contravariant123Vector(
    u::CovariantVector{T, (1, 2)},
    local_geometry::LocalGeometry{(1, 2, 3)},
) where {T} = local_geometry.gⁱʲ * Covariant123Vector(u[1], u[2], zero(u[1]))

Contravariant123Vector(
    u::CovariantVector{T, (3,)},
    local_geometry::LocalGeometry{(1, 2, 3)},
) where {T} =
    local_geometry.gⁱʲ * Covariant123Vector(zero(u[1]), zero(u[1]), u[1])


ContravariantVector(
    u::CovariantVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.gⁱʲ * u

CovariantVector(
    u::ContravariantVector{T, I},
    local_geometry::LocalGeometry{I},
) where {T, I} = local_geometry.gᵢⱼ * u

# Converting to specific dimension types
@inline (::Type{<:ContravariantVector{<:Any, I}})(
    u::ContravariantVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

@inline (::Type{<:ContravariantVector{<:Any, I}})(
    u::ContravariantVector,
    ::LocalGeometry,
) where {I} = project(ContravariantAxis{I}(), u)

@inline (::Type{<:ContravariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} =
    project(ContravariantAxis{I}(), ContravariantVector(u, local_geometry))

@inline (::Type{<:CovariantVector{<:Any, I}})(
    u::CovariantVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

@inline (::Type{<:CovariantVector{<:Any, I}})(
    u::CovariantVector,
    ::LocalGeometry,
) where {I} = project(CovariantAxis{I}(), u)

@inline (::Type{<:CovariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = project(CovariantAxis{I}(), CovariantVector(u, local_geometry))

@inline (::Type{<:LocalVector{<:Any, I}})(
    u::LocalVector{<:Any, I},
    ::LocalGeometry{I},
) where {I} = u

@inline (::Type{<:LocalVector{<:Any, I}})(
    u::LocalVector,
    ::LocalGeometry,
) where {I} = project(LocalAxis{I}(), u)

@inline (::Type{<:LocalVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = project(LocalAxis{I}(), LocalVector(u, local_geometry))

# Generic N-axis conversion functions,
# Convert to specific local geometry dimension then convert vector type
@inline LocalVector(
    u::CovariantVector,
    local_geometry::LocalGeometry{I},
) where {I} =
    project(LocalAxis{I}(), project(CovariantAxis{I}(), u), local_geometry)

@inline LocalVector(
    u::ContravariantVector,
    local_geometry::LocalGeometry{I},
) where {I} =
    project(LocalAxis{I}(), project(ContravariantAxis{I}(), u), local_geometry)

@inline CovariantVector(
    u::LocalVector,
    local_geometry::LocalGeometry{I},
) where {I} =
    project(CovariantAxis{I}(), project(LocalAxis{I}(), u), local_geometry)

@inline CovariantVector(
    u::ContravariantVector,
    local_geometry::LocalGeometry{I},
) where {I} = project(
    CovariantAxis{I}(),
    project(ContravariantAxis{I}(), u),
    local_geometry,
)

@inline ContravariantVector(
    u::LocalVector,
    local_geometry::LocalGeometry{I},
) where {I} =
    project(ContravariantAxis{I}(), project(LocalAxis{I}(), u), local_geometry)

@inline ContravariantVector(
    u::CovariantVector,
    local_geometry::LocalGeometry{I},
) where {I} = project(
    ContravariantAxis{I}(),
    project(CovariantAxis{I}(), u),
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

@inline Jcontravariant3(u::AxisTensor, local_geometry::LocalGeometry) =
    local_geometry.J * contravariant3(u, local_geometry)

# required for curl-curl
@inline covariant3(
    u::Contravariant3Vector,
    local_geometry::LocalGeometry{(1, 2)},
) = contravariant3(u, local_geometry)

# workarounds for using a Covariant12Vector/Covariant123Vector in a UW space:
function LocalVector(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, v, u₃ = components(vector)
    vector2 = Covariant13Vector(u₁, u₃)
    u, w = components(project(LocalAxis{(1, 3)}(), vector2, local_geometry))
    return UVWVector(u, v, w)
end
function contravariant1(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, _, u₃ = components(vector)
    vector2 = Covariant13Vector(u₁, u₃)
    return project(Contravariant13Axis(), vector2, local_geometry).u¹
end
function contravariant3(
    vector::CovariantVector{<:Any, (1, 2)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, _ = components(vector)
    vector2 = Covariant13Vector(u₁, zero(u₁))
    return project(Contravariant13Axis(), vector2, local_geometry).u³
end
function ContravariantVector(
    vector::CovariantVector{<:Any, (1, 2)},
    local_geometry::LocalGeometry{(1, 3)},
)
    u₁, v = components(vector)
    vector2 = Covariant1Vector(u₁)
    vector3 = project(
        ContravariantAxis{(1, 3)}(),
        project(CovariantAxis{(1, 3)}(), vector2),
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
        $op(
            ax::CartesianAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x' * $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        $op(
            ax::CovariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ' * $op(dual(axes(local_geometry.∂x∂ξ, 1)), v),
        )
        $op(ax::LocalAxis, v::CovariantTensor, local_geometry::LocalGeometry) =
            $op(
                ax,
                local_geometry.∂ξ∂x' *
                $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
            )
        $op(ax::CovariantAxis, v::LocalTensor, local_geometry::LocalGeometry) =
            $op(
                ax,
                local_geometry.∂x∂ξ' *
                $op(dual(axes(local_geometry.∂x∂ξ, 1)), v),
            )

        # Contravariant <-> Cartesian
        $op(
            ax::ContravariantAxis,
            v::CartesianTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x * $op(dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )
        $op(
            ax::CartesianAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ * $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )
        $op(
            ax::ContravariantAxis,
            v::LocalTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x * $op(dual(axes(local_geometry.∂ξ∂x, 2)), v),
        )

        $op(
            ax::LocalAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂x∂ξ * $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        # Covariant <-> Contravariant
        #=
        $op(
            ax::ContravariantAxis,
            v::CovariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.∂ξ∂x *
            local_geometry.∂ξ∂x' *
            $op(dual(axes(local_geometry.∂ξ∂x, 1)), v),
        )
        =#
        $op(
            ax::CovariantAxis,
            v::ContravariantTensor,
            local_geometry::LocalGeometry,
        ) = $op(
            ax,
            local_geometry.gᵢⱼ * $op(dual(axes(local_geometry.∂x∂ξ, 2)), v),
        )

        $op(ato::CovariantAxis, v::CovariantTensor, ::LocalGeometry) =
            $op(ato, v)
        $op(ato::ContravariantAxis, v::ContravariantTensor, ::LocalGeometry) =
            $op(ato, v)
        $op(ato::CartesianAxis, v::CartesianTensor, ::LocalGeometry) =
            $op(ato, v)
        $op(ato::LocalAxis, v::LocalTensor, ::LocalGeometry) = $op(ato, v)
    end
end

transform(
    ax::ContravariantAxis,
    v::CovariantTensor,
    local_geometry::LocalGeometry,
) = project(
    ax,
    local_geometry.gⁱʲ * project(dual(axes(local_geometry.∂ξ∂x, 1)), v),
)

@generated function project(
    ax::ContravariantAxis{Ito},
    v::CovariantVector{T, Ifrom},
    local_geometry::LocalGeometry{J},
) where {T, Ito, Ifrom, J}
    Nfrom = length(Ifrom)
    Nto = length(Ito)
    NJ = length(J)

    vals = []
    for i in Ito
        if i ∈ J
            # e.g. i = 2, J = (1,2,3)
            IJ = intersect(J, Ifrom)
            if isempty(IJ)
                val = 0
            else
                niJ = findfirst(==(i), J)
                val = Expr(
                    :call,
                    :+,
                    [
                        :(
                            local_geometry.gⁱʲ[$niJ, $(findfirst(==(j), J))] * v[$(findfirst(==(j), Ifrom))]
                        ) for j in IJ
                    ]...,
                )
            end
        elseif i ∈ Ifrom
            # e.g. i = 2, J = (1,3), Ifrom = (2,)
            ni = findfirst(==(i), Ifrom)
            val = :(v[$ni])
        else
            # e.g. i = 2, J = (1,3), Ifrom = (1,)
            val = 0
        end
        push!(vals, val)
    end
    quote
        Base.@_propagate_inbounds_meta
        AxisVector(ContravariantAxis{$Ito}(), SVector{$Nto, $T}($(vals...)))
    end
end
@generated function project(
    ax::ContravariantAxis{Ito},
    v::Contravariant2Tensor{T, Tuple{CovariantAxis{Ifrom}, A}},
    local_geometry::LocalGeometry{J},
) where {T, Ito, Ifrom, A, J}
    Nfrom = length(Ifrom)
    Nto = length(Ito)
    NJ = length(J)
    NA = length(A.instance)

    vals = []
    for na in 1:NA
        for i in Ito
            if i ∈ J
                # e.g. i = 2, J = (1,2,3)
                IJ = intersect(J, Ifrom)
                if isempty(IJ)
                    val = 0
                else
                    niJ = findfirst(==(i), J)
                    val = Expr(
                        :call,
                        :+,
                        [
                            :(
                                local_geometry.gⁱʲ[
                                    $niJ,
                                    $(findfirst(==(j), J)),
                                ] * v[$(findfirst(==(j), Ifrom)), $na]
                            ) for j in IJ
                        ]...,
                    )
                end
            elseif i ∈ Ifrom
                # e.g. i = 2, J = (1,3), Ifrom = (2,)
                ni = findfirst(==(i), Ifrom)
                val = :(v[$ni, $na])
            else
                # e.g. i = 2, J = (1,3), Ifrom = (1,)
                val = 0
            end
            push!(vals, val)
        end
    end
    quote
        Base.@_propagate_inbounds_meta
        AxisTensor(
            (ContravariantAxis{$Ito}(), A.instance),
            SMatrix{$Nto, $NA, $T, $(Nto * NA)}($(vals...)),
        )
    end
end

# A few other expensive ones:
#! format: off
function project(
    ax::ContravariantAxis{(1,)},
    v::AxisTensor{FT,2,Tuple{LocalAxis{(1, 2)},LocalAxis{(1, 2)}},SMatrix{2,2,FT,4}},
    lg::LocalGeometry{(1, 2, 3),XYZPoint{FT},FT,SMatrix{3,3,FT,9}}
) where {FT}
    AxisTensor(
        (ContravariantAxis{(1,)}(), LocalAxis{(1, 2)}()),
        @inbounds @SMatrix [
            lg.∂ξ∂x[1, 1]*v[1, 1]+lg.∂ξ∂x[1, 2]*v[2, 1] lg.∂ξ∂x[1, 1]*v[1, 2]+lg.∂ξ∂x[1, 2]*v[2, 2]
        ])
end
function project(
    ax::ContravariantAxis{(2,)},
    v::AxisTensor{FT,2,Tuple{LocalAxis{(1,2)},LocalAxis{(1,2)}},SMatrix{2,2,FT,4}},
    lg::LocalGeometry{(1,2,3),XYZPoint{FT},FT,SMatrix{3,3,FT,9}}
) where {FT}
    AxisTensor(
        (ContravariantAxis{(2,)}(), LocalAxis{(1, 2)}()),
        @inbounds @SMatrix [
            lg.∂ξ∂x[2, 1]*v[1, 1]+lg.∂ξ∂x[2, 2]*v[2, 1] lg.∂ξ∂x[2, 1]*v[1, 2]+lg.∂ξ∂x[2, 2]*v[2, 2]
        ]
    )
end
function project(
    ax::ContravariantAxis{(3,)},
    v::AxisTensor{FT,2,Tuple{LocalAxis{(3,)},LocalAxis{(1,2)}},SMatrix{1,2,FT,2}},
    lg::LocalGeometry{(1,2,3),XYZPoint{FT},FT,SMatrix{3,3,FT,9}}
) where {FT}
    AxisTensor(
        (ContravariantAxis{(3,)}(), LocalAxis{(1, 2)}()),
        @inbounds @SMatrix [lg.∂ξ∂x[3, 3]*v[1, 1] lg.∂ξ∂x[3, 3]*v[1, 2]]
    )
end
#! format: on


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
When `I` is an empty tuple, the gradient result type is assumed to be a Covariant12Vector.
"""
@inline function gradient_result_type(
    ::Val{()},
    ::Type{V},
) where {V <: Number}
    AxisVector{V, CovariantAxis{(1,2)}, SVector{2, V}}
end
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

# TODO: this is wrong, but replicates behavior of if a column represented by a box
@inline curl_result_type(
    ::Val{()},
    ::Type{Covariant3Vector{FT}},
) where {FT} = Contravariant12Vector{FT}
@inline curl_result_type(
    ::Val{()},
    ::Type{Covariant12Vector{FT}},
) where {FT} = Contravariant3Vector{FT}
@inline curl_result_type(
    ::Val{()},
    ::Type{Covariant123Vector{FT}},
) where {FT} = Contravariant123Vector{FT}


@inline curl_result_type(::Val{(1,)}, ::Type{Covariant1Vector{FT}}) where {FT} =
    Contravariant1Vector{FT} # not strictly correct: should be a zero Vector
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant2Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(
    ::Val{(1,)},
    ::Type{Covariant13Vector{FT}},
) where {FT} = Contravariant2Vector{FT}
@inline curl_result_type(
    ::Val{(1,)},
    ::Type{Covariant123Vector{FT}},
) where {FT} = Contravariant23Vector{FT}

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


_norm_sqr(
    u::Axis2Tensor{T, A, S},
    local_geometry::LocalGeometry,
) where {T, A <: Tuple{LocalAxis, LocalAxis}, S} =
    LinearAlgebra.norm_sqr(components(u))
_norm_sqr(
    u::Axis2Tensor{T, A, S},
    local_geometry::LocalGeometry,
) where {T, A <: Tuple{CartesianAxis, CartesianAxis}, S} =
    LinearAlgebra.norm_sqr(components(u))

_norm(u::AxisTensor, local_geometry::LocalGeometry) =
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
