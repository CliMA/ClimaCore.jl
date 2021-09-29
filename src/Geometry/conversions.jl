(AxisVector{T, A, SVector{1, T}} where {T})(
    a::Real,
    ::LocalGeometry,
) where {A} = AxisVector(A.instance, SVector(a))

ContravariantVector(u::ContravariantVector, local_geometry::LocalGeometry) = u
ContravariantVector(u::CartesianVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x * u
ContravariantVector(u::CovariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x * local_geometry.∂ξ∂x' * u

CovariantVector(u::CovariantVector, local_geometry::LocalGeometry) = u
CovariantVector(u::CartesianVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ' * u
CovariantVector(u::ContravariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ' * local_geometry.∂x∂ξ * u

CartesianVector(u::CartesianVector, local_geometry::LocalGeometry) = u
CartesianVector(u::CovariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x' * u
CartesianVector(u::ContravariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ * u



# These are for compatibility, and should be removed

Contravariant12Vector(u::ContravariantVector, local_geometry::LocalGeometry) = u
Contravariant12Vector(u::CartesianVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x * u
Contravariant12Vector(u::CovariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x * local_geometry.∂ξ∂x' * u

Covariant12Vector(u::CovariantVector, local_geometry::LocalGeometry) = u
Covariant12Vector(u::CartesianVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ' * u
Covariant12Vector(u::ContravariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ' * local_geometry.∂x∂ξ * u

Cartesian12Vector(u::CartesianVector, local_geometry::LocalGeometry) = u
Cartesian12Vector(u::CovariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂ξ∂x' * u
Cartesian12Vector(u::ContravariantVector, local_geometry::LocalGeometry) =
    local_geometry.∂x∂ξ * u


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

# conversions
function Covariant3Vector(
    uⁱ::Contravariant3Vector,
    local_geometry::LocalGeometry,
)
    # Not true generally, but is in 2D
    Covariant3Vector(uⁱ.u³)
end

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

Required for statically infering the result type of the divergence operation for StaticArray.FieldVector subtypes.
"""
divergence_result_type(::Type{V}) where {V <: AxisVector} = eltype(V)
divergence_result_type(
    ::Type{Axis2Tensor{FT, Tuple{A1, A2}, S}},
) where {
    FT,
    A1,
    A2 <: CartesianAxis,
    S <: StaticMatrix{S1, S2},
} where {S1, S2} = AxisVector{FT, A2, SVector{S2, FT}}

"""
    gradient_result_type(axes, V)

The return type when taking the gradient over `axes` of a field `V`.

Required for statically infering the result type of the gradient operator for StaticArray.FieldVector subtypes.
"""
function gradient_result_type(::Val{I}, ::Type{V}) where {I, V <: Number}
    N = length(I)
    AxisVector{V, CovariantAxis{I}, SVector{N, V}}
end
function gradient_result_type(
    ::Val{I},
    ::Type{V},
) where {I, V <: Geometry.AxisVector{T, A, SVector{N, T}}} where {T, A, N}
    M = length(I)
    Axis2Tensor{T, Tuple{CovariantAxis{I}, A}, SMatrix{M, N, T, M * N}}
end

"""
    curl_result_type(V)

The return type when taking the curl of a field of `V`.

Required for statically infering the result type of the divergence operation for StaticArray.FieldVector subtypes.
"""
curl_result_type(::Type{V}) where {V <: Covariant12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}
curl_result_type(::Type{V}) where {V <: Cartesian12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}
# TODO: not generally true that Contravariant3Vector => Covariant3Vector, but is for our 2D case
# curl of Covariant3Vector -> Contravariant12Vector
curl_result_type(::Type{V}) where {V <: Covariant3Vector{FT}} where {FT} =
    Contravariant12Vector{FT}

_norm_sqr(x, local_geometry) = sum(x -> _norm_sqr(x, local_geometry), x)
_norm_sqr(x::Number, local_geometry) = LinearAlgebra.norm_sqr(x)
_norm_sqr(x::AbstractArray, local_geometry) = LinearAlgebra.norm_sqr(x)

function _norm_sqr(u::Contravariant3Vector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(u.u³)
end

function _norm_sqr(uᵢ::CovariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(CartesianVector(uᵢ, local_geometry))
end

function _norm_sqr(uᵢ::ContravariantVector, local_geometry::LocalGeometry)
    LinearAlgebra.norm_sqr(CartesianVector(uᵢ, local_geometry))
end

_norm(u::AxisVector, local_geometry) = sqrt(_norm_sqr(u, local_geometry))

_cross(u::AxisVector, v::AxisVector, local_geometry) = LinearAlgebra.cross(
    ContravariantVector(u, local_geometry),
    ContravariantVector(v, local_geometry),
)
_cross(u::CartesianVector, v::CartesianVector, local_geometry) =
    LinearAlgebra.cross(u, v)
