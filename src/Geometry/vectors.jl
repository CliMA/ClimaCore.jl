import StaticArrays
using LinearAlgebra



"""
    AbstractAxis

These are the axes of our custom vector types. Their main purpose is to prevent
accidental conversion of one vector type to another when broadcasting.
"""
abstract type AbstractAxis end


Base.length(ax::AbstractAxis) = length(ax.range)
Base.iterate(ax::AbstractAxis) = iterate(ax.range)
Base.iterate(ax::AbstractAxis, i) = iterate(ax.range, i)
Base.getindex(ax::AbstractAxis, i) = getindex(ax.range, i)
Base.unitrange(ax::AbstractAxis) = Base.unitrange(ax.range)
Base.checkindex(::Type{Bool}, ax::AbstractAxis, i) =
    Base.checkindex(Bool, ax.range, i)
Base.lastindex(ax::AbstractAxis) = Base.lastindex(ax.range)

function Base.Broadcast.broadcast_shape(
    ax1::A,
    ax2::A,
) where {A <: AbstractAxis}
    @assert ax1 == ax2
    return ax1
end

Base.LinearIndices(axs::NTuple{N, AbstractAxis}) where {N} =
    LinearIndices(map(ax -> ax.range, axs))

struct CovariantAxis{R} <: AbstractAxis
    range::R
end
struct ContravariantAxis{R} <: AbstractAxis
    range::R
end
dual(cov::CovariantAxis) = ContravariantAxis(cov.range)
dual(con::ContravariantAxis) = CovariantAxis(con.range)

struct CartesianAxis{R} <: AbstractAxis
    range::R
end
dual(cart::CartesianAxis) = cart


iscontractible(::AbstractAxis, ::AbstractAxis) = false
iscontractible(axcov::CovariantAxis, axcon::ContravariantAxis) =
    axcov.range == axcon.range
iscontractible(axcov::ContravariantAxis, axcon::CovariantAxis) =
    axcov.range == axcon.range
iscontractible(ax1::CartesianAxis, ax2::CartesianAxis) = ax1.range == ax2.range

function check_iscontractible(ax1::AbstractAxis, ax2::AbstractAxis)
    iscontractible(ax1, ax2) || throw(DimensionMismatch("incompatible axis"))
    true
end

# Vectors

abstract type CustomAxisFieldVector{N, FT} <: StaticArrays.FieldVector{N, FT} end

(::Type{SA})(a::StaticArrays.StaticArray) where {SA <: CustomAxisFieldVector} =
    error("Cannot convert without local geometry information")

Base.axes(::CV) where {CV <: CustomAxisFieldVector} = Base.axes(CV)


# TODO: figure out which linear algebra operations make sense here:
#  - dot(a,b)


# All vectors are subtypes of StaticArrays.FieldVector, but with custom axes

"""
    components(u::StaticArrays.FieldVector)

Returns an `SVector` containing the components of `u` in its stored basis.
"""
components(u::CustomAxisFieldVector) =
    SVector(ntuple(i -> getfield(u, i), nfields(u)))



abstract type AbstractCovariantVector{N, FT} <: CustomAxisFieldVector{N, FT} end

function LinearAlgebra.dot(u::CustomAxisFieldVector, v::CustomAxisFieldVector)
    check_iscontractible(axes(u, 1), axes(v, 1))
    LinearAlgebra.dot(components(u), components(v))
end



"""
    Covariant12Vector

A vector point value represented as the first two covariant coordinates.
"""
struct Covariant12Vector{FT} <: AbstractCovariantVector{2, FT}
    u₁::FT
    u₂::FT
end
# Axes wrappers
Base.axes(::Type{Covariant12Vector{FT}}) where {FT} =
    (CovariantAxis(StaticArrays.SOneTo(2)),)

struct Covariant3Vector{FT} <: AbstractCovariantVector{2, FT}
    u₃::FT
end
# Axes wrappers
Base.axes(::Type{Covariant3Vector{FT}}) where {FT} =
    (CovariantAxis(StaticArrays.SUnitRange(3, 3)),)



abstract type AbstractContravariantVector{N, FT} <: CustomAxisFieldVector{N, FT} end

"""
    Contravariant12Vector

A vector point value represented as the first two contravariant coordinates.
"""
struct Contravariant12Vector{FT} <: AbstractContravariantVector{2, FT}
    u¹::FT
    u²::FT
end
Base.axes(::Type{Contravariant12Vector{FT}}) where {FT} =
    (ContravariantAxis(StaticArrays.SOneTo(2)),)


"""
    Contravariant3Vector

A vector point value represented as the third contravariant coordinates.
"""
struct Contravariant3Vector{FT} <: AbstractContravariantVector{1, FT}
    u³::FT
end
Contravariant3Vector{FT}(tup::Tuple{FT}) where {FT} =
    Contravariant3Vector{FT}(tup[1])

Base.axes(::Type{Contravariant3Vector{FT}}) where {FT} =
    (ContravariantAxis(StaticArrays.SUnitRange(3, 3)),)

#Base.:(*)(u::Contravariant12Vector,)
# Sphere:
#  Spherical in lat/long
#  Spherical as unit 3 vector
#  Spherical as local on a cubed-sphere face

abstract type AbstractCartesianVector{N, FT} <: CustomAxisFieldVector{N, FT} end

"""
    Cartesian12Vector

A vector point value represented by its first 2 cartesian coordinates.
"""
struct Cartesian12Vector{FT} <: AbstractCartesianVector{2, FT}
    u1::FT
    u2::FT
end
Base.axes(::Type{Cartesian12Vector{FT}}) where {FT} =
    (CartesianAxis(StaticArrays.SOneTo(2)),)

function contravariant1(x::Cartesian12Vector, local_geometry::LocalGeometry)
    LinearAlgebra.dot(local_geometry.∂ξ∂x[1, :], x)
end
function contravariant2(x::Cartesian12Vector, local_geometry::LocalGeometry)
    LinearAlgebra.dot(local_geometry.∂ξ∂x[2, :], x)
end
function contravariant1(x::Contravariant12Vector, local_geometry::LocalGeometry)
    x.u¹
end
function contravariant2(x::Contravariant12Vector, local_geometry::LocalGeometry)
    x.u²
end


function covariant1(x::Cartesian12Vector, local_geometry::LocalGeometry)
    (local_geometry.∂ξ∂x \ components(x))[1]
end

function covariant2(x::Cartesian12Vector, local_geometry::LocalGeometry)
    (local_geometry.∂ξ∂x \ components(x))[2]
end

# need to rethink
covariant3(x::Cartesian12Vector, local_geometry::LocalGeometry) =
    zero(eltype(x))


covariant1(x::Covariant3Vector, local_geometry::LocalGeometry) = zero(x.u₃)
covariant2(x::Covariant3Vector, local_geometry::LocalGeometry) = zero(x.u₃)
covariant3(x::Covariant3Vector, local_geometry::LocalGeometry) = x.u₃

# conversions

# uⁱ(∂ξ∂x, u, i) = sum(j->∂ξ∂x[i,j] * u[j], 1:size(∂ξ∂x,2))
function Cartesian12Vector(u::Covariant12Vector, local_geometry::LocalGeometry)
    # u[j] = ∂ξ∂x[i,j] * uᵢ
    Cartesian12Vector((local_geometry.∂ξ∂x' * components(u))...)
end



function Contravariant12Vector(
    u::Cartesian12Vector,
    local_geometry::LocalGeometry,
)
    # uⁱ = ∂ξ∂x[i,j] * u[j]
    Contravariant12Vector((local_geometry.∂ξ∂x * components(u))...)
end
function Cartesian12Vector(
    uⁱ::Contravariant12Vector,
    local_geometry::LocalGeometry,
)
    # u[j] = ∂ξ∂x[i,j] \ uⁱ
    Cartesian12Vector((local_geometry.∂ξ∂x \ components(uⁱ))...)
end

"""
    divergence_result_type(V)

The return type when taking the divergence of a field of type `V`.

Required for statically infering the result type of the divergence operation for StaticArray.FieldVector subtypes.
"""
divergence_result_type(::Type{V}) where {V <: CustomAxisFieldVector} = eltype(V)

curl_result_type(::Type{V}) where {V <: Covariant12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}
curl_result_type(::Type{V}) where {V <: Cartesian12Vector{FT}} where {FT} =
    Contravariant3Vector{FT}

# not generally true that Contravariant3Vector => Covariant3Vector, but is for our 2D case
# curl of Covariant3Vector -> Contravariant12Vector
curl_result_type(::Type{V}) where {V <: Covariant3Vector{FT}} where {FT} =
    Contravariant12Vector{FT}

function norm²(uᵢ::Covariant12Vector, local_geometry::LocalGeometry)
    u = Cartesian12Vector(uᵢ, local_geometry)
    norm²(u, local_geometry)
end

function norm²(u::Cartesian12Vector, local_geometry::LocalGeometry)
    abs2(u.u1) + abs2(u.u2)
end

LinearAlgebra.norm(u::CustomAxisFieldVector, local_geometry::LocalGeometry) =
    sqrt(norm²(u, local_geometry))

function LinearAlgebra.cross(
    uⁱ::Contravariant12Vector,
    v::Contravariant3Vector,
    local_geometry::LocalGeometry,
)
    Covariant12Vector(uⁱ.u² * v.u³, -uⁱ.u¹ * v.u³)
end

function LinearAlgebra.cross(
    u::Cartesian12Vector,
    v::Contravariant3Vector,
    local_geometry::LocalGeometry,
)
    uⁱ = Contravariant12Vector(u, local_geometry)
    LinearAlgebra.cross(uⁱ, v, local_geometry)
end

# tensors

struct Tensor{U, V, N, M, FT, A} <: StaticArrays.StaticMatrix{N, M, FT}
    matrix::A

    function Tensor{U, V}(
        matrix::A,
    ) where {U, V, A <: StaticArrays.StaticMatrix}
        N = length(U)
        M = length(V)
        FT = eltype(matrix)
        @assert size(matrix) == (N, M)
        new{U, V, N, M, FT, A}(matrix)
    end
end

Base.axes(::Type{T}) where {T <: Tensor{U, V}} where {U, V} =
    (axes(U)[1], axes(V)[1])
Base.axes(::T) where {T <: Tensor} = Base.axes(T)

Base.getindex(r::Tensor, i::Int...) = Base.getindex(r.matrix, i...)

function ⊗(
    u::U,
    v::V,
) where {U <: CustomAxisFieldVector, V <: CustomAxisFieldVector}
    Tensor{U, V}(components(u) * components(v)')
end

function Base.:(*)(A::Tensor{U, V}, v::CustomAxisFieldVector) where {U, V}
    check_iscontractible(axes(A, 2), axes(v, 1))
    U(A.matrix * components(v))
end
function Base.adjoint(A::Tensor{U, V}) where {U, V}
    Tensor{V, U}(adjoint(A.matrix))
end

function Base.:(+)(
    A::Tensor{U, V},
    b::LinearAlgebra.UniformScaling,
) where {U, V}
    check_iscontractible(axes(A)...)
    Tensor{U, V}(A.matrix + b)
end
function Base.:(+)(
    b::LinearAlgebra.UniformScaling,
    A::Tensor{U, V},
) where {U, V}
    check_iscontractible(axes(A)...)
    Tensor{U, V}(b + A.matrix)
end
function Base.:(-)(
    A::Tensor{U, V},
    b::LinearAlgebra.UniformScaling,
) where {U, V}
    check_iscontractible(axes(A)...)
    Tensor{U, V}(A.matrix - b)
end
function Base.:(-)(
    b::LinearAlgebra.UniformScaling,
    A::Tensor{U, V},
) where {U, V}
    check_iscontractible(axes(A)...)
    Tensor{U, V}(b - A.matrix)
end


divergence_result_type(::Type{T}) where {T <: Tensor{U, V}} where {U, V} = V

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
