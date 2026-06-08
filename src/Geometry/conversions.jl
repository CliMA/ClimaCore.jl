## Components-type conversion helpers (private)

# Metrics are identity-padded to full (1,2,3) shape on `LocalGeometry`, so
# every conversion is a single matvec — names outside `lg`'s geometry `I`
# ride the identity block of the padded matrix automatically. Same-type
# pairs are explicit no-ops; cross-type pairs pick the appropriate cached
# matrix.
@inline _to_components_type(::Contravariant, v::ContravariantTensor, ::LocalGeometry) = v
@inline _to_components_type(::Covariant, v::CovariantTensor, ::LocalGeometry) = v
@inline _to_components_type(::Orthonormal, v::OrthonormalTensor, ::LocalGeometry) = v
@inline _to_components_type(::Contravariant, v::CovariantTensor, lg::LocalGeometry) =
    lg.gⁱʲ * v
@inline _to_components_type(::Covariant, v::ContravariantTensor, lg::LocalGeometry) =
    lg.gᵢⱼ * v
@inline _to_components_type(::Contravariant, v::OrthonormalTensor, lg::LocalGeometry) =
    lg.∂ξ∂x * v
@inline _to_components_type(::Covariant, v::OrthonormalTensor, lg::LocalGeometry) =
    lg.∂x∂ξ' * v
@inline _to_components_type(::Orthonormal, v::ContravariantTensor, lg::LocalGeometry) =
    lg.∂x∂ξ * v
@inline _to_components_type(::Orthonormal, v::CovariantTensor, lg::LocalGeometry) =
    lg.∂ξ∂x' * v

## project(basis, v, local_geometry)  — 3-argument form using metric

"""
    project(basis, V, local_geometry)

Project the first axis of vector or tensor `V` onto `basis`, performing a
change of basis type via the metric if necessary.  Missing components are
zero-filled; extra components are dropped (no error even if they are nonzero).
Identity-metric passthrough on dimensions orthogonal to `local_geometry`'s
geometry (e.g. dim 3 in a horizontal `(1,2)` `LocalGeometry`) is handled by
`_to_components_type`, which preserves every source name in the converted result; the
final `reshape` then zero-fills any remaining destination names that aren't
in the source.
"""
@inline project(b::Components{BT}, v::AbstractTensor, lg::LocalGeometry) where {BT} =
    reshape(_to_components_type(BT(), v, lg), (b, Base.tail(axes(v))...))

"""
    transform(basis, V, local_geometry)

Like `project(basis, V, local_geometry)`, but throws an `InexactError` if any
dropped component is nonzero.
"""
@inline transform(b::Components{BT}, v::AbstractTensor, lg::LocalGeometry) where {BT} =
    transform(b, _to_components_type(BT(), v, lg))

## Vector type constructors with LocalGeometry

# Standard same-dimension conversions: forward to the private `_to_components_type`.
@inline ContravariantVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_components_type(Contravariant(), u, lg)
@inline CovariantVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_components_type(Covariant(), u, lg)
@inline LocalVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_components_type(Orthonormal(), u, lg)

## Scalar constructor for 1D vectors (e.g. WVector(1.0, lg))

# A 1D vector type built from a scalar doesn't need the metric — drop the lg.
@inline (::Type{T})(a::Real, ::LocalGeometry) where {T <: Tensor{1}} = T(a)

## Callable type constructors (e.g. Contravariant1Vector(u, lg))

for (BT, VecType) in (
    (Covariant, :CovariantVector),
    (Contravariant, :ContravariantVector),
    (Orthonormal, :LocalVector),
)
    # General: convert to full basis type, then project to requested dimensions
    @eval @inline (::Type{<:$VecType{<:Any, I}})(
        u::AbstractTensor{1}, lg::LocalGeometry,
    ) where {I} = project(Components{$BT, I}(), $VecType(u, lg))

    # Identity: already in the right basis type and dimension
    @eval @inline (::Type{<:$VecType{<:Any, I}})(
        u::$VecType{<:Any, I}, ::LocalGeometry{I},
    ) where {I} = u
end

## Scalar component extractors

for n in 1:3
    @eval @inline $(Symbol(:covariant, n))(u::AbstractTensor{1}, lg::LocalGeometry) =
        project($(Symbol(:Covariant, n, :Axis))(), u, lg)[1]
    @eval @inline $(Symbol(:contravariant, n))(u::AbstractTensor{1}, lg::LocalGeometry) =
        project($(Symbol(:Contravariant, n, :Axis))(), u, lg)[1]
    @eval @inline $(Symbol(:contravariant, n))(u::AbstractTensor{2}, lg::LocalGeometry) =
        project($(Symbol(:Contravariant, n, :Axis))(), u, lg)[1, :]
end

@inline Jcontravariant3(u::AbstractTensor, lg::LocalGeometry) =
    lg.J * contravariant3(u, lg)

## Operator result types

"""
    divergence_result_type(V)


Return type when taking the divergence of a field of `V`.
"""
@inline divergence_result_type(::Type{V}) where {V <: AbstractTensor{1}} = eltype(V)
@inline divergence_result_type(
    ::Type{Tensor{2, FT, Tuple{A1, A2}, S}},
) where {FT, A1, A2 <: AbstractComponents, S <: StaticMatrix{S1, S2}} where {S1, S2} =
    Tensor{1, FT, Tuple{A2}, SVector{S2, FT}}

"""
    gradient_result_type(Val(I), V)

Return type when taking the gradient along dimension `I` of a field of type `V`.
"""
@inline function gradient_result_type(::Val{I}, ::Type{V}) where {I, V <: Number}
    N = length(I)
    CovariantVector{V, I, SVector{N, V}}
end
@inline function gradient_result_type(
    ::Val{I},
    ::Type{Tensor{1, T, Tuple{A}, SVector{N, T}}},
) where {I, T, A, N}
    M = length(I)
    Tensor{2, T, Tuple{Components{Covariant, I}, A}, SMatrix{M, N, T, M * N}}
end

"""
    curl_result_type(Val(I), V)

Return type when taking the curl along dimensions `I` of a field of type `V`.
Always returns the full `Contravariant123Vector`; dimensions outside the
actual curl range carry zeros, consistent with the identity-padded metric
convention used throughout `LocalGeometry`.
"""
@inline curl_result_type(_, ::Type{<:CovariantVector{FT}}) where {FT} =
    Contravariant123Vector{FT}

## Norm and cross-product (used in broadcast.jl)
##
## These are metric-aware versions of norm and cross that take a LocalGeometry.
## broadcast.jl routes `norm(field)` and `cross(field1, field2)` here so that
## the correct geometric magnitude is computed regardless of what basis the
## vectors are stored in. Unlike LinearAlgebra.norm / LinearAlgebra.cross, these
## convert to the local Orthonormal frame (or Contravariant for cross) first.

_norm_sqr(x, lg::LocalGeometry) = sum(x -> _norm_sqr(x, lg), x)
_norm_sqr(x::Number, ::LocalGeometry) = norm_sqr(x)
_norm_sqr(x::AbstractArray, ::LocalGeometry) = norm_sqr(x)
_norm_sqr(uᵢ::AbstractTensor{1}, lg::LocalGeometry) =
    norm_sqr(LocalVector(uᵢ, lg))
_norm_sqr(uᵢ::OrthonormalTensor, ::LocalGeometry) = norm_sqr(uᵢ)

_norm(u::AbstractTensor, lg::LocalGeometry) = sqrt(_norm_sqr(u, lg))

# TODO: Determine if this 3D general method impacts performance
function _cross(u::AbstractTensor{1}, v::AbstractTensor{1}, lg::LocalGeometry)
    x = ContravariantVector(u, lg)
    y = ContravariantVector(v, lg)
    return lg.J * Covariant123Vector(
        x.u² * y.u³ - x.u³ * y.u²,
        x.u³ * y.u¹ - x.u¹ * y.u³,
        x.u¹ * y.u² - x.u² * y.u¹,
    )
end

_cross(u::OrthonormalTensor, v::OrthonormalTensor, ::LocalGeometry) = cross(u, v)
