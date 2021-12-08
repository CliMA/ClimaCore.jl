(AxisVector{T, A, SVector{1, T}} where {T})(
    a::Real,
    ::LocalGeometry,
) where {A} = AxisVector(A.instance, SVector(a))

ContravariantVector(u::ContravariantVector, local_geometry::LocalGeometry) = u
CovariantVector(u::CovariantVector, local_geometry::LocalGeometry) = u
LocalVector(u::LocalVector, local_geometry::LocalGeometry) = u

# convert contravariant <-> local <-> covariant
ContravariantVector(u::CovariantVector, local_geometry::LocalGeometry) =
    ContravariantVector(LocalVector(u, local_geometry), local_geometry)
CovariantVector(u::ContravariantVector, local_geometry::LocalGeometry) =
    CovariantVector(LocalVector(u, local_geometry), local_geometry)

# standard conversions
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

# In order to make curls and cross products work in 2D, we define the 3rd
# dimension to be orthogonal to the exisiting dimensions, and have unit length
# (so covariant, contravariant and local axes are equal)
ContravariantVector(
    u::LocalVector{T, (3,)},
    local_geometry::LocalGeometry{(1, 2)},
) where {T} = AxisVector(Contravariant3Axis(), components(u))
LocalVector(
    u::ContravariantVector{T, (3,)},
    local_geometry::LocalGeometry{(1, 2)},
) where {T} = AxisVector(WAxis(), components(u))
CovariantVector(
    u::LocalVector{T, (3,)},
    local_geometry::LocalGeometry{(1, 2)},
) where {T} = AxisVector(Covariant3Axis(), components(u))
LocalVector(
    u::CovariantVector{T, (3,)},
    local_geometry::LocalGeometry{(1, 2)},
) where {T} = AxisVector(WAxis(), components(u))

# Converting to specific dimension types
(::Type{<:ContravariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} =
    transform(ContravariantAxis{I}(), ContravariantVector(u, local_geometry))
(::Type{<:CovariantVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = transform(CovariantAxis{I}(), CovariantVector(u, local_geometry))
(::Type{<:LocalVector{<:Any, I}})(
    u::AxisVector,
    local_geometry::LocalGeometry,
) where {I} = transform(LocalAxis{I}(), LocalVector(u, local_geometry))



covariant1(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₁
covariant2(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₂
covariant3(u::AxisVector, local_geometry::LocalGeometry) =
    CovariantVector(u, local_geometry).u₃

# TODO: specialize?
contravariant1(u::AxisVector, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry).u¹
contravariant2(u::AxisVector, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry).u²
contravariant3(u::AxisVector, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry).u³

contravariant1(u::Axis2Tensor, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry)[1, :]
contravariant2(u::Axis2Tensor, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry)[2, :]
contravariant3(u::Axis2Tensor, local_geometry::LocalGeometry) =
    transform(Contravariant123Axis(), u, local_geometry)[3, :]

Jcontravariant3(u::AxisTensor, local_geometry::LocalGeometry) =
    local_geometry.J * contravariant3(u, local_geometry)


# required for curl-curl
covariant3(u::Contravariant3Vector, local_geometry::LocalGeometry{(1, 2)}) =
    contravariant3(u, local_geometry)

"""
    transform(axis, V[, local_geometry])

Transform the first axis of the vector or tensor `V` to `axis`.
"""
function transform end

# Covariant <-> Cartesian
function transform(
    ax::CartesianAxis,
    v::CovariantTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂ξ∂x' * transform(dual(axes(local_geometry.∂ξ∂x, 1)), v),
    )
end
function transform(
    ax::CovariantAxis,
    v::CartesianTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂x∂ξ' * transform(dual(axes(local_geometry.∂x∂ξ, 1)), v),
    )
end
function transform(
    ax::LocalAxis,
    v::CovariantTensor,
    local_geometry::LocalGeometry,
)
    @show typeof(v)
    transform(
        ax,
        local_geometry.∂ξ∂x' * transform(dual(axes(local_geometry.∂ξ∂x, 1)), v),
    )
end
function transform(
    ax::CovariantAxis,
    v::LocalTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂x∂ξ' * transform(dual(axes(local_geometry.∂x∂ξ, 1)), v),
    )
end

# Contravariant <-> Cartesian
function transform(
    ax::ContravariantAxis,
    v::CartesianTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂ξ∂x * transform(dual(axes(local_geometry.∂ξ∂x, 2)), v),
    )
end
function transform(
    ax::CartesianAxis,
    v::ContravariantTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂x∂ξ * transform(dual(axes(local_geometry.∂x∂ξ, 2)), v),
    )
end
function transform(
    ax::ContravariantAxis,
    v::LocalTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂ξ∂x * transform(dual(axes(local_geometry.∂ξ∂x, 2)), v),
    )
end
function transform(
    ax::LocalAxis,
    v::ContravariantTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂x∂ξ * transform(dual(axes(local_geometry.∂x∂ξ, 2)), v),
    )
end

# Covariant <-> Contravariant
function transform(
    ax::ContravariantAxis,
    v::CovariantTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂ξ∂x *
        local_geometry.∂ξ∂x' *
        transform(dual(axes(local_geometry.∂ξ∂x, 1)), v),
    )
end
function transform(
    ax::CovariantAxis,
    v::ContravariantTensor,
    local_geometry::LocalGeometry,
)
    transform(
        ax,
        local_geometry.∂x∂ξ' *
        local_geometry.∂x∂ξ *
        transform(dual(axes(local_geometry.∂x∂ξ, 2)), v),
    )
end

transform(ato::CovariantAxis, v::CovariantTensor, ::LocalGeometry) =
    transform(ato, v)
transform(ato::ContravariantAxis, v::ContravariantTensor, ::LocalGeometry) =
    transform(ato, v)
transform(ato::CartesianAxis, v::CartesianTensor, ::LocalGeometry) =
    transform(ato, v)
transform(ato::LocalAxis, v::LocalTensor, ::LocalGeometry) = transform(ato, v)



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
"""
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Covariant12Vector{FT}},
) where {FT} = Contravariant3Vector{FT}
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Covariant3Vector{FT}},
) where {FT} = Contravariant12Vector{FT}
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

# these are only true in 2D
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Contravariant12Vector{FT}},
) where {FT} = Contravariant3Vector{FT}
@inline curl_result_type(::Val{(1, 2)}, ::Type{UVVector{FT}}) where {FT} =
    Contravariant3Vector{FT}
@inline curl_result_type(
    ::Val{(1, 2)},
    ::Type{Contravariant3Vector{FT}},
) where {FT} = Contravariant12Vector{FT}
@inline curl_result_type(::Val{(1, 2)}, ::Type{WVector{FT}}) where {FT} =
    Contravariant12Vector{FT}


_norm_sqr(x, local_geometry) = sum(x -> _norm_sqr(x, local_geometry), x)
_norm_sqr(x::Number, local_geometry) = LinearAlgebra.norm_sqr(x)
_norm_sqr(x::AbstractArray, local_geometry) = LinearAlgebra.norm_sqr(x)

function _norm_sqr(uᵢ::CovariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(LocalVector(uᵢ, local_geometry))
end

function _norm_sqr(uᵢ::ContravariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(LocalVector(uᵢ, local_geometry))
end

_norm_sqr(u::Contravariant2Vector, local_geometry::LocalGeometry{(1,)}) =
    LinearAlgebra.norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, local_geometry::LocalGeometry{(3,)}) =
    LinearAlgebra.norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, local_geometry::LocalGeometry{(1, 3)}) =
    LinearAlgebra.norm_sqr(u.u²)

_norm_sqr(u::Contravariant3Vector, local_geometry::LocalGeometry{(1,)}) =
    LinearAlgebra.norm_sqr(u.u³)
_norm_sqr(u::Contravariant3Vector, local_geometry::LocalGeometry{(1, 2)}) =
    LinearAlgebra.norm_sqr(u.u³)



_norm(u::AxisVector, local_geometry) = sqrt(_norm_sqr(u, local_geometry))

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
    x::Contravariant12Vector,
    y::Contravariant3Vector,
    local_geometry::LocalGeometry,
) = local_geometry.J * Covariant12Vector(x.u² * y.u³, -x.u¹ * y.u³)


_cross(u::CartesianVector, v::CartesianVector, local_geometry::LocalGeometry) =
    LinearAlgebra.cross(u, v)
_cross(u::LocalVector, v::LocalVector, local_geometry::LocalGeometry) =
    LinearAlgebra.cross(u, v)
