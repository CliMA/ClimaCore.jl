



abstract type Abstract2DVectorPoint end

"""
    components(u::AbstactVectorPoint)

Returns an `SVector` containing the components of `u` in its stored basis.
"""
components(u::Abstract2DVectorPoint) =
    SVector(ntuple(i -> getfield(u, i), nfields(u)))



"""
    Covariant12Vector

A vector point value represented as the first two covariant coordinates.
"""
struct Covariant12Vector{FT} <: Abstract2DVectorPoint
    u₁::FT
    u₂::FT
end

Base.eltype(::Covariant12Vector{FT}) where {FT} = FT


"""
    Contravariant12Vector

A vector point value represented as the first two contavariant coordinates.
"""
struct Contravariant12Vector{FT} <: Abstract2DVectorPoint
    u¹::FT
    u²::FT
end
Base.eltype(::Contravariant12Vector{FT}) where {FT} = FT


#Base.:(*)(u::Contravariant12Vector,)
# Sphere:
#  Spherical in lat/long
#  Spherical as unit 3 vector
#  Spherical as local on a cubed-sphere face

"""
    Cartesian2DVector

A vector point value represented in the local reference cartesian coordinate system.
"""
struct Cartesian2DVector{FT} <: Abstract2DVectorPoint
    u1::FT
    u2::FT
end

# uⁱ(∂ξ∂x, u, i) = sum(j->∂ξ∂x[i,j] * u[j], 1:size(∂ξ∂x,2))
function Cartesian2DVector(u::Covariant12Vector, local_geometry::LocalGeometry)
    # u[j] = ∂ξ∂x[i,j] * uᵢ
    Cartesian2DVector((local_geometry.∂ξ∂x' * components(u))...)
end
function Contravariant12Vector(
    u::Cartesian2DVector,
    local_geometry::LocalGeometry,
)
    # uⁱ = ∂ξ∂x[i,j] * u[j]
    Contravariant12Vector((local_geometry.∂ξ∂x * components(u))...)
end

Base.:(+)(u::Cartesian2DVector, v::Cartesian2DVector) =
    Cartesian2DVector(u.u1 + v.u1, u.u2 + v.u2)
Base.:(*)(w::Number, u::Cartesian2DVector) =
    Cartesian2DVector(w * u.u1, w * u.u2)

Base.eltype(::Cartesian2DVector{FT}) where {FT} = FT
Base.eltype(::Type{Cartesian2DVector{FT}}) where {FT} = FT
