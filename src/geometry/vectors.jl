import StaticArrays


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
Base.checkindex(::Type{Bool}, ax::AbstractAxis, i) = Base.checkindex(Bool, ax.range, i)

function Base.Broadcast.broadcast_shape(ax1::A, ax2::A) where {A<:AbstractAxis}
    @assert ax1 == ax2
    return ax1
end

struct CovariantAxis{R} <: AbstractAxis
    range::R
end
struct ContravariantAxis{R} <: AbstractAxis
    range::R
end
struct CartesianAxis{R} <: AbstractAxis
    range::R
end

iscontractible(::AbstractAxis, ::AbstractAxis) = false
iscontractible(axcov::CovariantAxis, axcon::ContravariantAxis) = axcov.range ==  axcon.range
iscontractible(axcov::ContravariantAxis, axcon::CovariantAxis) = axcov.range ==  axcon.range
iscontractible(ax1::CartesianAxis, ax2::CartesianAxis) = ax1.range == ax2.range

function check_iscontractible(ax1::AbstractAxis,ax2::AbstractAxis)
    iscontractible(ax1,ax2) || throw(DimensionMismatch("incompatible axis"))
    true
end

# Vectors

abstract type CustomAxisFieldVector{N,FT} <: StaticArrays.FieldVector{N,FT} end

Base.axes(::CV) where {CV<:CustomAxisFieldVector} = Base.axes(CV)


# TODO: figure out which linear algebra operations make sense here:
#  - dot(a,b)


# All vectors are subtypes of StaticArrays.FieldVector, but with custom axes

"""
    components(u::StaticArrays.FieldVector)

Returns an `SVector` containing the components of `u` in its stored basis.
"""
components(u::CustomAxisFieldVector) =
    SVector(ntuple(i -> getfield(u, i), nfields(u)))



abstract type AbstractCovariantVector{N,FT} <: CustomAxisFieldVector{N,FT} end


"""
    Covariant12Vector

A vector point value represented as the first two covariant coordinates.
"""
struct Covariant12Vector{FT} <: AbstractCovariantVector{2,FT}
    u₁::FT
    u₂::FT
end
# Axes wrappers
Base.axes(::Type{Covariant12Vector{FT}}) where {FT} = (CovariantAxis(StaticArrays.SOneTo(2)),)



abstract type AbstractContravariantVector{N,FT} <: CustomAxisFieldVector{N,FT} end

"""
    Contravariant12Vector

A vector point value represented as the first two contavariant coordinates.
"""
struct Contravariant12Vector{FT} <: AbstractContravariantVector{2,FT}
    u¹::FT
    u²::FT
end
Base.axes(::Type{Contravariant12Vector{FT}}) where {FT} = (ContravariantAxis(StaticArrays.SOneTo(2)),)


#Base.:(*)(u::Contravariant12Vector,)
# Sphere:
#  Spherical in lat/long
#  Spherical as unit 3 vector
#  Spherical as local on a cubed-sphere face

abstract type AbstractCartesianVector{N,FT} <: CustomAxisFieldVector{N,FT} end

"""
    Cartesian12Vector

A vector point value represented by its first 2 cartesian coordinates.
"""
struct Cartesian12Vector{FT} <: AbstractCartesianVector{2,FT}
    u1::FT
    u2::FT
end
Base.axes(::Type{Cartesian12Vector{FT}}) where {FT} = (CartesianAxis(StaticArrays.SOneTo(2)),)


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


# tensors

struct Tensor{U,V,N,M,FT,A} <: StaticArrays.StaticMatrix{N,M,FT}
    matrix::A

    function Tensor{U,V}(matrix::A) where {U,V,A<:StaticArrays.StaticMatrix}
        N = length(U)
        M = length(V)
        FT = eltype(matrix)
        @assert size(matrix) == (N,M)
        new{U,V,N,M,FT,A}(matrix)
    end
end

Base.axes(::Type{T}) where {T<:Tensor{U,V}} where {U,V} = (axes(U)[1], axes(V)[1])
Base.axes(::T) where {T<:Tensor} = Base.axes(T)

Base.getindex(r::Tensor, i::Int...) = Base.getindex(r.matrix, i...)

function ⊗(u::U, v::V) where {U<:CustomAxisFieldVector, V<:CustomAxisFieldVector}
    Tensor{U,V}(components(u) * components(v)')
end

function Base.:(*)(A::Tensor{U,V}, v::CustomAxisFieldVector) where {U,V}
    check_iscontractible(axes(A,2), axes(v,1))
    U(A.matrix * components(v))
end