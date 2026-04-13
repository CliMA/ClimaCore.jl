import LinearAlgebra: issymmetric

isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, TX, TG1, TG2}
    "Coordinates of the current point"
    coordinates::C
    "Jacobian determinant of the transformation `ξ` (reference space) to `x` (physical space)"
    J::FT
    "Metric terms: `J` multiplied by the quadrature weights"
    WJ::FT
    "∂x/∂ξ tensor (Orthonormal row × Covariant column bases)"
    ∂x∂ξ::TX
    "Contravariant metric tensor gⁱʲ = (∂ξ/∂x)(∂ξ/∂x)ᵀ"
    gⁱʲ::TG1
    "Covariant metric tensor gᵢⱼ = (∂x/∂ξ)ᵀ(∂x/∂ξ)"
    gᵢⱼ::TG2
end
# TODO: Benchmark which properties should be stored, which should be computed on the fly

@inline function Base.getproperty(lg::LocalGeometry, name::Symbol)
    return if name === :invJ
        inv(getfield(lg, :J))
    elseif name === :∂ξ∂x
        inv(getfield(lg, :∂x∂ξ))
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
    # Use A'A and BB' forms (where B = A⁻¹) to guarantee exact symmetry in
    # floating point, unlike change_of_basis_tensor which computes inv(A'A).
    ∂ξ∂x = inv(∂x∂ξ)
    _gᵢⱼ = ∂x∂ξ' * ∂x∂ξ
    _gⁱʲ = ∂ξ∂x * ∂ξ∂x'
    isapproxsymmetric(parent(_gⁱʲ)) || error("gⁱʲ is not symmetric.")
    isapproxsymmetric(parent(_gᵢⱼ)) || error("gᵢⱼ is not symmetric.")
    return LocalGeometry{names, C, FT, typeof(∂x∂ξ), typeof(_gⁱʲ), typeof(_gᵢⱼ)}(
        coordinates, J, WJ, ∂x∂ξ, _gⁱʲ, _gᵢⱼ,
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
    _gⁱʲ_bases = (Basis{Contravariant, I}(), Basis{Contravariant, I}())
    _gᵢⱼ_bases = (Basis{Covariant, I}(), Basis{Covariant, I}())
    TX = Tensor{2, FT, typeof(_∂x∂ξ_bases), SMatrix{N, N, FT, N * N}}
    TG1 = Tensor{2, FT, typeof(_gⁱʲ_bases), SMatrix{N, N, FT, N * N}}
    TG2 = Tensor{2, FT, typeof(_gᵢⱼ_bases), SMatrix{N, N, FT, N * N}}
    return LocalGeometry{I, C, FT, TX, TG1, TG2}
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

"""
    blockmat(a, b[, c])

Construct a block-diagonal (or block-lower-triangular) Tensor{2} from sub-blocks.
Uses `combine_bases` and `reshape` to zero-fill missing components.
"""
function blockmat(a::Tensor{2}, b::Tensor{2}, ::Nothing = nothing)
    new_bases = (
        combine_bases(axes(a, 1), axes(b, 1)),
        combine_bases(axes(a, 2), axes(b, 2)),
    )
    return reshape(a, new_bases) + reshape(b, new_bases)
end

function blockmat(a::Tensor{2}, b::Tensor{2}, c::Tensor{2})
    new_bases = (
        combine_bases(axes(a, 1), axes(b, 1)),
        combine_bases(axes(a, 2), axes(b, 2)),
    )
    return reshape(a, new_bases) + reshape(b, new_bases) + reshape(c, new_bases)
end
