import LinearAlgebra: issymmetric

isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T <: AbstractFloat} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, TMet <: Metric, TG1}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` (reference space) to `x` (physical space)"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "Canonical metric, wrapping ∂x/∂ξ (Orthonormal row × Covariant column)"
    metric::TMet
    "Contravariant metric tensor gⁱʲ = (∂ξ/∂x)(∂ξ/∂x)ᵀ. Kept precomputed because
    Cov -> Contra (via gⁱʲ) is the hottest conversion. Other forms (∂ξ/∂x, gᵢⱼ) are
    derived on demand through `getproperty`."
    gⁱʲ::TG1
end

@inline function Base.getproperty(lg::LocalGeometry, name::Symbol)
    return if name === :invJ
        inv(getfield(lg, :J))
    elseif name === :∂x∂ξ
        getfield(lg, :metric).tensor
    elseif name === :∂ξ∂x
        inv(getfield(lg, :metric).tensor)
    elseif name === :gᵢⱼ
        inv(getfield(lg, :gⁱʲ))
    else
        getfield(lg, name)
    end
end

# Primary constructor: accepts a Tensor{2} with Orthonormal/Covariant bases
@inline function LocalGeometry(
    coordinates::C,
    J::FT,
    WJ::FT,
    ∂x∂ξ::Tensor{2},
) where {C, FT}
    names = basis_vector_names(axes(∂x∂ξ, 1))
    ∂ξ∂x = inv(∂x∂ξ)
    gⁱʲ = ∂ξ∂x * ∂ξ∂x'
    isapproxsymmetric(parent(gⁱʲ)) || error("gⁱʲ is not symmetric.")
    @assert isapproxsymmetric(parent(∂x∂ξ' * ∂x∂ξ)) "gᵢⱼ is not symmetric."
    metric = Metric(∂x∂ξ)
    return LocalGeometry{names, C, FT, typeof(metric), typeof(gⁱʲ)}(
        coordinates, J, WJ, metric, gⁱʲ,
    )
end

"""
    LocalGeometryType(::Type{C}, ::Type{FT}, I)

Compute the concrete `LocalGeometry` type for coordinate type `C`, float type `FT`,
and index tuple `I`. Useful for pre-allocating DataLayouts with the correct element type.
"""
function LocalGeometryType(::Type{C}, ::Type{FT}, I::Tuple) where {C <: AbstractPoint, FT}
    N = length(I)
    _∂x∂ξ_bases = (Basis{Orthonormal, I}(), Basis{Covariant, I}())
    gⁱʲ_bases = (Basis{Contravariant, I}(), Basis{Contravariant, I}())
    TX = Tensor{2, FT, typeof(_∂x∂ξ_bases), SMatrix{N, N, FT, N * N}}
    TMet = Metric{TX}
    TG1 = Tensor{2, FT, typeof(gⁱʲ_bases), SMatrix{N, N, FT, N * N}}
    return LocalGeometry{I, C, FT, TMet, TG1}
end

"""
    SurfaceGeometry

The necessary local metric information defined at each node on each surface.
"""
struct SurfaceGeometry{FT, N}
    "surface Jacobian determinant, multiplied by the surface quadrature weight"
    sWJ::FT
    "surface outward pointing normal vector"
    normal::N
end

"""
    CoordinateOnlyGeometry

The necessary coordinates information defined at each node.

This is currently used for constructing spaces with pressure as the vertical
coordinate.
"""
struct CoordinateOnlyGeometry{C <: AbstractPoint}
    "Coordinates of the current point"
    coordinates::C
end

undertype(::Type{<:LocalGeometry{I, C, FT}}) where {I, C, FT} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
undertype(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = eltype(C)
