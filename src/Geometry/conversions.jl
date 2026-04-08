## Basis-type conversion helpers (private)

# Relabel the first-axis basis type in place. For dimensions orthogonal to
# `lg`'s geometry, the metric is identity by convention (u_i ≡ u^i ≡ u), so
# only the label changes — component values stay put.
@inline function _relabel(::BT, v::AbstractTensor) where {BT <: BasisType}
    names = basis_vector_names(axes(v, 1))
    new_bases = (Basis(BT(), names), Base.tail(axes(v))...)
    return Tensor(parent(v), new_bases)
end

# Pick the change-of-basis tensor that performs each (target ← source)
# conversion. `gⁱʲ` is cached on `LocalGeometry`; the others are derived via
# `getproperty` (`∂x∂ξ` is a free field read, `∂ξ∂x` and `gᵢⱼ` invoke
# `inv` on the cached SMatrix — small constant cost the JIT typically
# hoists when the result feeds a sparse matvec). The single-matvec form is
# kept because rewriting these as two matvecs explodes FLOPs for partial-dim
# inputs (e.g. `WVector` in 3D LG): the second matvec is dense even when
# the first is sparse, doubling cost on hot paths like vertical advection.
@inline _apply_metric(::Contravariant, v::CovariantTensor, lg::LocalGeometry) =
    lg.gⁱʲ * v
@inline _apply_metric(::Covariant, v::ContravariantTensor, lg::LocalGeometry) =
    lg.gᵢⱼ * v
@inline _apply_metric(::Contravariant, v::OrthonormalTensor, lg::LocalGeometry) =
    lg.∂ξ∂x * v
@inline _apply_metric(::Covariant, v::OrthonormalTensor, lg::LocalGeometry) =
    lg.∂x∂ξ' * v
@inline _apply_metric(::Orthonormal, v::ContravariantTensor, lg::LocalGeometry) =
    lg.∂x∂ξ * v
@inline _apply_metric(::Orthonormal, v::CovariantTensor, lg::LocalGeometry) =
    lg.∂ξ∂x' * v

# Convert the first axis of `v` to `target` basis type. Source names that fall
# inside `lg`'s geometry dims `I` go through the cached metric tensor via
# `_apply_metric`; names outside `I` are along dimensions orthogonal to `lg`
# (identity metric u_i ≡ u^i ≡ u), so they pass through with only a basis-type
# relabel. Mixed cases split `v` along axis 1, convert each piece, and sum in
# the shared target basis.
@inline function _to_basis_type(
    target::BasisType, v::AbstractTensor, lg::LocalGeometry{I},
) where {I}
    src_basis = axes(v, 1)
    src_bt = basis_type(src_basis)
    src_bt === target && return v  # already in target basis
    # Partition source names by whether `lg`'s metric covers them.
    src_names = basis_vector_names(src_basis)
    metric_names = unrolled_filter(n -> unrolled_in(n, I), src_names)
    passthrough_names = unrolled_filter(n -> !unrolled_in(n, I), src_names)
    # Pure passthrough - no name is in `lg`'s geometry, just relabel.
    isempty(metric_names) && return _relabel(target, v)
    # Pure metric - every name is in `lg`'s geometry, apply the metric to all of `v`.
    isempty(passthrough_names) && return _apply_metric(target, v, lg)
    # Mixed - split `v` along axis 1, convert each piece, sum in the target basis.
    tail_axes = Base.tail(axes(v))
    v_metric = reshape(v, (Basis(src_bt, metric_names), tail_axes...))
    v_passthrough = reshape(v, (Basis(src_bt, passthrough_names), tail_axes...))
    return _apply_metric(target, v_metric, lg) + _relabel(target, v_passthrough)
end

## project(basis, v, local_geometry)  — 3-argument form using metric

"""
    project(basis, V, local_geometry)

Project the first axis of vector or tensor `V` onto `basis`, performing a
change of basis type via the metric if necessary.  Missing components are
zero-filled; extra components are dropped (no error even if they are nonzero).
Identity-metric passthrough on dimensions orthogonal to `local_geometry`'s
geometry (e.g. dim 3 in a horizontal `(1,2)` `LocalGeometry`) is handled by
`_to_basis_type`, which preserves every source name in the converted result; the
final `reshape` then zero-fills any remaining destination names that aren't
in the source.
"""
@inline project(b::Basis{BT}, v::AbstractTensor, lg::LocalGeometry) where {BT} =
    reshape(_to_basis_type(BT(), v, lg), (b, Base.tail(axes(v))...))

"""
    transform(basis, V, local_geometry)

Like `project(basis, V, local_geometry)`, but throws an `InexactError` if any
dropped component is nonzero.
"""
@inline transform(b::Basis{BT}, v::AbstractTensor, lg::LocalGeometry) where {BT} =
    transform(b, _to_basis_type(BT(), v, lg))

## Vector type constructors with LocalGeometry

# Standard same-dimension conversions: forward to the private `_to_basis_type`.
@inline ContravariantVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_basis_type(Contravariant(), u, lg)
@inline CovariantVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_basis_type(Covariant(), u, lg)
@inline LocalVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    _to_basis_type(Orthonormal(), u, lg)

## Scalar constructor for 1D vectors (e.g. WVector(1.0, lg))

# 1D vector types can be constructed from a scalar + LocalGeometry.
# The LocalGeometry is ignored — the scalar is wrapped directly.
for I in [(1,), (2,), (3,)]
    strI = string(I[1])
    strUVW = string([:U, :V, :W][I[1]])
    for sym in [Symbol(:Covariant, strI, :Vector),
        Symbol(:Contravariant, strI, :Vector),
        Symbol(strUVW, :Vector)]
        @eval @inline $sym(a::Real, ::LocalGeometry) = $sym(a)
    end
end

## Callable type constructors (e.g. Contravariant1Vector(u, lg))

for (BT, VecType, fn) in (
    (Covariant, :CovariantVector, :CovariantVector),
    (Contravariant, :ContravariantVector, :ContravariantVector),
    (Orthonormal, :LocalVector, :LocalVector),
)
    # General: convert to full basis type, then project to requested dimensions
    @eval @inline (::Type{<:$VecType{<:Any, I}})(
        u::AbstractTensor{1}, lg::LocalGeometry,
    ) where {I} = project(Basis{$BT, I}(), $fn(u, lg))

    # Identity: already in the right basis type and dimension
    @eval @inline (::Type{<:$VecType{<:Any, I}})(
        u::$VecType{<:Any, I}, ::LocalGeometry{I},
    ) where {I} = u
end

## Scalar component extractors

for (n, cov_sym, con_sym) in ((1, :u₁, :u¹), (2, :u₂, :u²), (3, :u₃, :u³))
    @eval @inline $(Symbol(:covariant, n))(u::AbstractTensor{1}, lg::LocalGeometry) =
        CovariantVector(u, lg).$cov_sym
    @eval @inline $(Symbol(:contravariant, n))(u::AbstractTensor{1}, lg::LocalGeometry) =
        project($(Symbol(:Contravariant, n, :Axis))(), u, lg)[1]
    @eval @inline $(Symbol(:contravariant, n))(u::Tensor{2}, lg::LocalGeometry) =
        project($(Symbol(:Contravariant, n, :Axis))(), u, lg)[1, :]
end

@inline Jcontravariant3(u::AbstractTensor, lg::LocalGeometry) =
    lg.J * contravariant3(u, lg)

# required for curl-curl
@inline covariant3(u::Contravariant3Vector, lg::LocalGeometry{(1, 2)}) =
    contravariant3(u, lg)

## Operator result types

"""
    divergence_result_type(V)

Return type when taking the divergence of a field of `V`.
"""
@inline divergence_result_type(::Type{V}) where {V <: AbstractTensor{1}} = eltype(V)
@inline divergence_result_type(
    ::Type{Tensor{2, FT, Tuple{A1, A2}, S}},
) where {FT, A1, A2 <: AbstractBasis, S <: StaticMatrix{S1, S2}} where {S1, S2} =
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
    Tensor{2, T, Tuple{Basis{Covariant, I}, A}, SMatrix{M, N, T, M * N}}
end

"""
    curl_result_type(Val(I), Val(L), V)

Return type when taking the curl along dimensions `I` of a field of type `V`.
"""
@inline curl_result_type(::Val{(1, 2)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant12Vector{FT}
@inline curl_result_type(::Val{(1, 2)}, ::Type{Covariant12Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}
@inline curl_result_type(::Val{(1, 2)}, ::Type{Covariant123Vector{FT}}) where {FT} =
    Contravariant123Vector{FT}

@inline curl_result_type(::Val{(1,)}, ::Type{Covariant1Vector{FT}}) where {FT} =
    ContravariantNullVector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant2Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant13Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(::Val{(1,)}, ::Type{Covariant123Vector{FT}}) where {FT} =
    Contravariant23Vector{FT}

@inline curl_result_type(::Val{(3,)}, ::Type{Covariant12Vector{FT}}) where {FT} =
    Contravariant12Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant1Vector{FT}}) where {FT} =
    Contravariant2Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant2Vector{FT}}) where {FT} =
    Contravariant1Vector{FT}
@inline curl_result_type(::Val{(3,)}, ::Type{Covariant3Vector{FT}}) where {FT} =
    Contravariant3Vector{FT}

@inline curl_result_type(_, ::Type{<:CovariantVector{FT}}) where {FT} =
    ContravariantNullVector{FT}

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
    norm_sqr(parent(LocalVector(uᵢ, lg)))
_norm_sqr(uᵢ::OrthonormalTensor, ::LocalGeometry) =
    norm_sqr(parent(uᵢ))

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
