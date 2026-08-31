isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T <: AbstractFloat} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry

The necessary local metric information defined at each node.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, M, G}
    """
    Coordinates of the current point
    """
    coordinates::C
    """
    Jacobian determinant of the transformation `ξ` (reference space) to `x` (physical space)
    """
    J::FT
    """
    Metric terms: `J` multiplied by the quadrature weights
    """
    WJ::FT
    """
    Canonical metric ∂x/∂ξ. Identity-padded to full
    (UVWAxis, Covariant123Axis) shape so a single matvec covers every
    conversion regardless of `I`.
    """
    ∂x∂ξ::M
    """
    Contravariant metric tensor gⁱʲ. Identity-padded to full
    (Contravariant123, Contravariant123) shape.
    """
    gⁱʲ::G
end

@inline function Base.getproperty(lg::LocalGeometry, name::Symbol)
    return if name === :invJ
        inv(getfield(lg, :J))
    elseif name === :∂ξ∂x
        inv(getfield(lg, :∂x∂ξ))
    elseif name === :gᵢⱼ
        inv(getfield(lg, :gⁱʲ))
    else
        getfield(lg, name)
    end
end

# Primary constructor: accepts a Tensor{2} with Orthonormal/Covariant bases
# of any size; pads to full 3×3 internally.
@inline function LocalGeometry(
    coordinates::C,
    J::FT,
    WJ::FT,
    ∂x∂ξ::Tensor{2},
) where {C, FT}
    names = component_names(axes(∂x∂ξ, 1))
    padded = pad_metric_tensor(∂x∂ξ)
    ∂ξ∂x = inv(padded)
    gⁱʲ = ∂ξ∂x * ∂ξ∂x'
    isapproxsymmetric(parent(gⁱʲ)) || error("gⁱʲ is not symmetric.")
    @assert isapproxsymmetric(parent(padded' * padded)) "gᵢⱼ is not symmetric."
    return LocalGeometry{names, C, FT, typeof(padded), typeof(gⁱʲ)}(
        coordinates, J, WJ, padded, gⁱʲ,
    )
end

const Padded∂x∂ξ{FT} =
    Tensor{2, FT, Tuple{UVWAxis, Covariant123Axis}, SMatrix{3, 3, FT, 9}}
const PaddedContravariantMetric{FT} =
    Tensor{2, FT, Tuple{Contravariant123Axis, Contravariant123Axis}, SMatrix{3, 3, FT, 9}}

"""
    LocalGeometryType(::Type{C}, ::Type{FT}, I)

Compute the concrete `LocalGeometry` type for coordinate type `C`, float type `FT`,
and index tuple `I`. Useful for pre-allocating DataLayouts with the correct element type.
"""
function LocalGeometryType(::Type{C}, ::Type{FT}, I::Tuple) where {C <: AbstractPoint, FT}
    return LocalGeometry{
        I, C, FT,
        Padded∂x∂ξ{FT},
        PaddedContravariantMetric{FT},
    }
end

"""
    SurfaceGeometry

The necessary local metric information defined at each node on each surface.
"""
struct SurfaceGeometry{FT, N}
    """
    surface Jacobian determinant, multiplied by the surface quadrature weight
    """
    sWJ::FT
    """
    surface outward pointing normal vector
    """
    normal::N
end

"""
    CoordinateOnlyGeometry

The necessary coordinates information defined at each node.

This is currently used for constructing spaces with pressure as the vertical
coordinate.
"""
struct CoordinateOnlyGeometry{C <: AbstractPoint}
    """
    Coordinates of the current point
    """
    coordinates::C
end

undertype(::Type{<:LocalGeometry{I, C, FT}}) where {I, C, FT} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
undertype(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = eltype(C)

"""
    coordinate_type(geom)

The `AbstractPoint` type of the coordinates of the geometry `geom`, which can
be either an object or a type.
"""
coordinate_type(::Type{<:LocalGeometry{I, C}}) where {I, C} = C
coordinate_type(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = C
coordinate_type(geom::Union{LocalGeometry, CoordinateOnlyGeometry}) =
    coordinate_type(typeof(geom))
