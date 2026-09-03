isapproxsymmetric(A::AbstractMatrix{T}; rtol = 10 * eps(T)) where {T <: AbstractFloat} =
    Base.isapprox(A, A'; rtol)

"""
    LocalGeometry{I, C, FT, M, G}
    LocalGeometry(coordinates, J, WJ, ∂x∂ξ::Tensor{2})

Local metric information defined at each node.

`I` is the tuple of component names of the reference-space axes that the node's grid
spans, `C` the coordinate type, `FT` the float type, and `M` and `G` the padded
tensor types of `∂x∂ξ` and `gⁱʲ`. The constructor accepts a `∂x∂ξ` `Tensor{2}` with
orthonormal and covariant bases of any size and pads it to the full 3×3 shape.

# Fields

  - `coordinates`: coordinates of the node.
  - `J`: Jacobian determinant of the transformation from reference space `ξ` to physical
    space `x`.
  - `WJ`: `J` multiplied by the quadrature weight.
  - `∂x∂ξ`: canonical metric ∂x/∂ξ, identity-padded to the full
    `(UVWAxis, Covariant123Axis)` shape so that a single matrix-vector product covers
    every conversion regardless of `I`.
  - `gⁱʲ`: contravariant metric tensor, identity-padded to the full
    `(Contravariant123Axis, Contravariant123Axis)` shape.

The derived properties `invJ`, `∂ξ∂x`, and `gᵢⱼ` are computed on access as the inverses
of `J`, `∂x∂ξ`, and `gⁱʲ`.
"""
struct LocalGeometry{I, C <: AbstractPoint, FT, M, G}
    coordinates::C
    J::FT
    WJ::FT
    ∂x∂ξ::M
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

Return the concrete `LocalGeometry` type for coordinate type `C`, float type `FT`, and
component-name tuple `I`, for pre-allocating data layouts with the correct element type.
"""
function LocalGeometryType(::Type{C}, ::Type{FT}, I::Tuple) where {C <: AbstractPoint, FT}
    return LocalGeometry{
        I, C, FT,
        Padded∂x∂ξ{FT},
        PaddedContravariantMetric{FT},
    }
end

"""
    SurfaceGeometry{FT, N}

Local metric information defined at each node on each element surface.

# Fields

  - `sWJ`: surface Jacobian determinant multiplied by the surface quadrature weight.
  - `normal`: outward-pointing surface normal vector.
"""
struct SurfaceGeometry{FT, N}
    sWJ::FT
    normal::N
end

"""
    CoordinateOnlyGeometry{C}

Coordinate information defined at each node, without metric terms.

Used for constructing spaces with pressure as the vertical coordinate.

# Fields

  - `coordinates`: coordinates of the node.
"""
struct CoordinateOnlyGeometry{C <: AbstractPoint}
    coordinates::C
end

undertype(::Type{<:LocalGeometry{I, C, FT}}) where {I, C, FT} = FT
undertype(::Type{SurfaceGeometry{FT, N}}) where {FT, N} = FT
undertype(::Type{<:CoordinateOnlyGeometry{C}}) where {C} = eltype(C)
