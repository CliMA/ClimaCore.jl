#=
(AxisTensor{T, 1, Tuple{A}, SVector{1, T}} where {T})(
    u::Real,
    local_geometry::LocalGeometry,
) where {A} = (AxisTensor{T, 1, Tuple{A}, SVector{1, T}} where {T})(u)
=#


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
3
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



"""
    divergence_result_type(V)

The return type when taking the divergence of a field of type `V`.

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

curl_result_type(::Type{V}) where {V <: Covariant12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}
curl_result_type(::Type{V}) where {V <: Cartesian12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}

# not generally true that Contravariant3Vector => Covariant3Vector, but is for our 2D case
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




#=

function contravariant1(
    A::Tensor{Contravariant12Vector{FT}, V},
    local_geometry::LocalGeometry,
) where {FT, V}
    V(A.matrix[1, :]...)
end
function contravariant2(
    A::Tensor{Contravariant12Vector{FT}, V},
    local_geometry::LocalGeometry,
) where {FT, V}
    V(A.matrix[2, :]...)
end
function contravariant1(
    A::Tensor{Cartesian12Vector{FT}, V},
    local_geometry::LocalGeometry,
) where {FT, V}
    V((local_geometry.∂ξ∂x[1, :]' * A.matrix)...)
end
function contravariant2(
    A::Tensor{Cartesian12Vector{FT}, V},
    local_geometry::LocalGeometry,
) where {FT, V}
    V((local_geometry.∂ξ∂x[2, :]' * A.matrix)...)
end
=#
#=


"""
    SphericalCartesianVector

Representation of a vector in spherical cartesian coordinates.
"""
struct SphericalCartesianVector{FT <: Number}
    "zonal (eastward) component"
    u::FT
    "meridional (northward) component"
    v::FT
    "radial (upward) component"
    w::FT
end


function spherical_cartesian_basis(geom::LocalGeometry)
    x = geom.x

    r12 = hypot(x[2], x[1])
    r = hypot(x[3], r12)

    (
        û = SVector(-x[2] / r12, x[1] / r12, 0),
        v̂ = SVector(
            -(x[3] / r) * (x[1] / r12),
            -(x[3] / r) * (x[2] / r12),
            r12 / r,
        ),
        ŵ = x ./ r,
    )
end

function SphericalCartesianVector(v::CartesianVector, geom::LocalGeometry)
    b = spherical_cartesian_basis(geom)
    SphericalCartesianVector(dot(b.û, v), dot(b.v̂, v), dot(b.ŵ, v))
end

function CartesianVector(s::SphericalCartesianVector, geom::LocalGeometry)
    b = spherical_cartesian_basis(geom)
    return s.u .* b.û .+ s.v .* b.v̂ .+ s.w .* b.ŵ
end

=#
