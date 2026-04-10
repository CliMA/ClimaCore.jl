###############################################################################
## Basis-type conversion helpers (private)
###############################################################################

# Union type aliases like CovariantTensor use Basis{Covariant} without the
# `names` type parameter, so functions dispatching on them lose `names` at
# compile time. Any method that performs a matrix-vector product must instead
# dispatch on Tensor{1, <:Any, <:Tuple{Basis{BT, names}}} where {names} so
# that Julia captures `names` as a type parameter and avoids runtime dispatch
# in basis overlap/dual computations. 

# Convert any Tensor{1} to the Contravariant basis of lg
@inline _to_contravariant(v::ContravariantTensor, ::LocalGeometry) = v
@inline _to_contravariant(
    v::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal, names}}}, lg::LocalGeometry,
) where {names} = lg.∂ξ∂x * v
@inline _to_contravariant(
    v::Tensor{1, <:Any, <:Tuple{Basis{Covariant, names}}}, lg::LocalGeometry,
) where {names} = lg.gⁱʲ * v

# Convert any Tensor{1} to the Covariant basis of lg
@inline _to_covariant(v::CovariantTensor, ::LocalGeometry) = v
@inline _to_covariant(
    v::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal, names}}}, lg::LocalGeometry,
) where {names} = lg.∂x∂ξ' * v
@inline _to_covariant(
    v::Tensor{1, <:Any, <:Tuple{Basis{Contravariant, names}}}, lg::LocalGeometry,
) where {names} = lg.gᵢⱼ * v

# Convert any Tensor{1} to the Orthonormal (local) basis of lg
@inline _to_local(v::OrthonormalTensor, ::LocalGeometry) = v
@inline _to_local(
    v::Tensor{1, <:Any, <:Tuple{Basis{Contravariant, names}}}, lg::LocalGeometry,
) where {names} = lg.∂x∂ξ * v
@inline _to_local(
    v::Tensor{1, <:Any, <:Tuple{Basis{Covariant, names}}}, lg::LocalGeometry,
) where {names} = lg.∂ξ∂x' * v

###############################################################################
## project(basis, v, local_geometry)  — 3-argument form using metric
###############################################################################

"""
    project(basis, V, local_geometry)

Project the first axis of vector or tensor `V` onto `basis`, performing a
change of basis type via the metric if necessary.  Missing components are
zero-filled; extra components are dropped (no error even if they are nonzero).
"""
@inline function project(
    b::Basis{Contravariant}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Contravariant(), _to_contravariant(v, lg), v)
end
@inline function project(
    b::Basis{Covariant}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Covariant(), _to_covariant(v, lg), v)
end
@inline function project(
    b::Basis{Orthonormal}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Orthonormal(), _to_local(v, lg), v)
end

# 2-tensor projections: change basis type for the first axis
@inline function project(
    b::Basis{Contravariant}, v::Tensor{2}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Contravariant(), _to_contravariant_2t(v, lg), v)
end
@inline function project(
    b::Basis{Covariant}, v::Tensor{2}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Covariant(), _to_covariant_2t(v, lg), v)
end
@inline function project(
    b::Basis{Orthonormal}, v::Tensor{2}, lg::LocalGeometry,
)
    return _project_with_passthrough(b, Orthonormal(), _to_local_2t(v, lg), v)
end

# Helper: reshape `converted` to `b`, but for components in `b` that are
# missing from the metric conversion yet present in `source`, pass them through
# directly.
#
# This arises in extruded geometries: the horizontal `LocalGeometry` has a 2D
# metric (dimensions 1,2 only), so `gⁱʲ * Covariant123` yields `Contravariant12`
# — the metric knows nothing about dimension 3. But the vertical direction is
# orthogonal with unit arc-length by construction (reference and physical
# coordinates coincide there), giving g^{33} = g_{33} = 1. Therefore
# u³ = u₃ = u (all three representations are equal), and projecting a
# `Covariant123` source to `Contravariant123` should copy component 3 as-is
# rather than zero-filling it.
@inline function _project_with_passthrough(
    b::Basis{BT}, ::BT, converted::AbstractTensor{N}, source::AbstractTensor{N},
) where {BT <: BasisType, N}
    converted_names = basis_vector_names(axes(converted, 1))
    src_names = basis_vector_names(axes(source, 1))
    dest_names = basis_vector_names(b)
    extra_names = unrolled_filter(
        n -> !unrolled_in(n, converted_names) && unrolled_in(n, src_names),
        dest_names,
    )
    other_converted_axes = Base.tail(axes(converted))
    result = reshape(converted, (b, other_converted_axes...))
    isempty(extra_names) && return result
    # Extract source components for extra names (covariant/contravariant values
    # are equal for identity-metric dimensions) and relabel as target basis type.
    src_extra_basis = Basis(basis_type(axes(source, 1)), extra_names)
    other_source_axes = Base.tail(axes(source))
    src_extra = reshape(source, (src_extra_basis, other_source_axes...))
    # Relabel as target basis type (same values, g^{ii}=1 assumption)
    tgt_extra = Tensor(parent(src_extra), (Basis{BT, extra_names}(), other_source_axes...))
    other_result_axes = Base.tail(axes(result))
    return result + reshape(tgt_extra, (b, other_result_axes...))
end

# 2-tensor basis-type conversion for the first axis
@inline _to_contravariant_2t(v::ContravariantTensor, ::LocalGeometry) = v
@inline _to_contravariant_2t(v::OrthonormalTensor, lg::LocalGeometry) = lg.∂ξ∂x * v
@inline _to_contravariant_2t(v::CovariantTensor, lg::LocalGeometry) = lg.gⁱʲ * v

@inline _to_covariant_2t(v::CovariantTensor, ::LocalGeometry) = v
@inline _to_covariant_2t(v::OrthonormalTensor, lg::LocalGeometry) = lg.∂x∂ξ' * v
@inline _to_covariant_2t(v::ContravariantTensor, lg::LocalGeometry) = lg.gᵢⱼ * v

@inline _to_local_2t(v::OrthonormalTensor, ::LocalGeometry) = v
@inline _to_local_2t(v::ContravariantTensor, lg::LocalGeometry) = lg.∂x∂ξ * v
@inline _to_local_2t(v::CovariantTensor, lg::LocalGeometry) = lg.∂ξ∂x' * v

"""
    transform(basis, V, local_geometry)

Like `project(basis, V, local_geometry)`, but throws an `InexactError` if any
dropped component is nonzero.
"""
@inline function transform(
    b::Basis{Contravariant}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return transform(b, _to_contravariant(v, lg))
end
@inline function transform(
    b::Basis{Covariant}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return transform(b, _to_covariant(v, lg))
end
@inline function transform(
    b::Basis{Orthonormal}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    return transform(b, _to_local(v, lg))
end

###############################################################################
## Vector type constructors with LocalGeometry
###############################################################################

# Standard same-dimension conversions:

# Use explicit `where {names}` on matrix-product methods so Julia captures the
# names type parameter and can infer concrete return types (see comment at top).
@inline ContravariantVector(u::ContravariantTensor, ::LocalGeometry) = u
@inline ContravariantVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal, names}}}, lg::LocalGeometry,
) where {names} = lg.∂ξ∂x * u
@inline ContravariantVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Covariant, names}}}, lg::LocalGeometry,
) where {names} = lg.gⁱʲ * u

@inline CovariantVector(u::CovariantTensor, ::LocalGeometry) = u
@inline CovariantVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal, names}}}, lg::LocalGeometry,
) where {names} = lg.∂x∂ξ' * u
@inline CovariantVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Contravariant, names}}}, lg::LocalGeometry,
) where {names} = lg.gᵢⱼ * u

@inline LocalVector(u::OrthonormalTensor, ::LocalGeometry) = u
@inline LocalVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Contravariant, names}}}, lg::LocalGeometry,
) where {names} = lg.∂x∂ξ * u
@inline LocalVector(
    u::Tensor{1, <:Any, <:Tuple{Basis{Covariant, names}}}, lg::LocalGeometry,
) where {names} = lg.∂ξ∂x' * u

# UVWVector(u, lg) — convert any vector to the UVW (full-3D Orthonormal) basis.
# LocalVector is identity for OrthonormalTensor, so no specialization needed.
@inline UVWVector(u::AbstractTensor{1}, lg::LocalGeometry) =
    reshape(LocalVector(u, lg), (UVWAxis(),))

# 2D cross-dimension conversions for (3,) vectors in a (1,2) geometry.
# The 3rd dimension is orthogonal with unit length: covariant ≡ contravariant ≡ local,
# so conversion is just a relabeling of components (no metric needed).
const _dest_info = [
    (Covariant, :CovariantVector, :Covariant3Vector),
    (Contravariant, :ContravariantVector, :Contravariant3Vector),
    (Orthonormal, :LocalVector, :WVector),
]
for (dest_T, fn, constructor) in _dest_info
    for src_T in (Covariant, Contravariant, Orthonormal)
        if src_T === dest_T
            @eval @inline $fn(
                u::Tensor{1, <:Any, Tuple{Basis{$src_T, (3,)}}},
                ::LocalGeometry{(1, 2)},
            ) = u
        else
            @eval @inline $fn(
                u::Tensor{1, <:Any, Tuple{Basis{$src_T, (3,)}}},
                ::LocalGeometry{(1, 2)},
            ) = $constructor(parent(u)...)
        end
    end
end


###############################################################################
## Scalar constructor for 1D vectors (e.g. WVector(1.0, lg))
###############################################################################

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

###############################################################################
## Callable type constructors (e.g. Contravariant1Vector(u, lg))
###############################################################################

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

###############################################################################
## Scalar component extractors
###############################################################################

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

# Specialized scalar FMA-friendly versions (avoids extra mat-vec allocation on CUDA)
for (geom_I, mat_idx) in (((3,), 1), ((1, 2, 3), 3))
    @eval @inline Jcontravariant3(u::Covariant3Vector, lg::LocalGeometry{$geom_I}) =
        @inbounds lg.J * lg.gⁱʲ[$mat_idx, $mat_idx] * parent(u)[1]
    @eval @inline Jcontravariant3(u::WVector, lg::LocalGeometry{$geom_I}) =
        @inbounds lg.J * lg.∂ξ∂x[$mat_idx, $mat_idx] * parent(u)[1]
end

# required for curl-curl
@inline covariant3(u::Contravariant3Vector, lg::LocalGeometry{(1, 2)}) =
    contravariant3(u, lg)

###############################################################################
## Special workarounds for mixed-dimension geometries
###############################################################################

# Covariant123Vector in a UW (1,3) geometry: treat v (index 2) as orthonormal
function LocalVector(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    lg::LocalGeometry{(1, 3)},
)
    u₁, v, u₃ = parent(vector)
    vector2 = Covariant13Vector(u₁, u₃)
    u, w = parent(project(Basis{Orthonormal, (1, 3)}(), vector2, lg))
    return UVWVector(u, v, w)
end

function contravariant1(
    vector::CovariantVector{<:Any, (1, 2, 3)},
    lg::LocalGeometry{(1, 3)},
)
    pv = parent(vector)
    vector2 = Covariant13Vector(pv[1], pv[3])
    return project(Contravariant13Axis(), vector2, lg).u¹
end

function contravariant3(
    vector::CovariantVector{<:Any, (1, 2)},
    lg::LocalGeometry{(1, 3)},
)
    u₁ = parent(vector)[1]
    vector2 = Covariant13Vector(u₁, zero(u₁))
    return project(Contravariant13Axis(), vector2, lg).u³
end

function ContravariantVector(
    vector::CovariantVector{<:Any, (1, 2)},
    lg::LocalGeometry{(1, 3)},
)
    u₁, v = parent(vector)
    vector2 = Covariant1Vector(u₁)
    vector3 = project(Contravariant13Axis(), project(Covariant13Axis(), vector2), lg)
    u¹, u³ = parent(vector3)
    return Contravariant123Vector(u¹, v, u³)
end

###############################################################################
## Operator result types
###############################################################################

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

###############################################################################
## Norm and cross-product (used in broadcast.jl)
##
## These are metric-aware versions of norm and cross that take a LocalGeometry.
## broadcast.jl routes `norm(field)` and `cross(field1, field2)` here so that
## the correct geometric magnitude is computed regardless of what basis the
## vectors are stored in. Unlike LinearAlgebra.norm / LinearAlgebra.cross, these
## convert to the local Orthonormal frame (or Contravariant for cross) first.
###############################################################################

_norm_sqr(x, lg::LocalGeometry) = sum(x -> _norm_sqr(x, lg), x)
_norm_sqr(x::Number, ::LocalGeometry) = norm_sqr(x)
_norm_sqr(x::AbstractArray, ::LocalGeometry) = norm_sqr(x)

function _norm_sqr(uᵢ::Union{CovariantVector, ContravariantVector}, lg::LocalGeometry)
    norm_sqr(parent(LocalVector(uᵢ, lg)))
end
function _norm_sqr(uᵢ::OrthonormalTensor, ::LocalGeometry)
    norm_sqr(parent(uᵢ))
end

# Specialized for common 1D cases to avoid unnecessary matrix operations
_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(1,)}) = norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(3,)}) = norm_sqr(u.u²)
_norm_sqr(u::Contravariant2Vector, ::LocalGeometry{(1, 3)}) = norm_sqr(u.u²)
_norm_sqr(u::Contravariant3Vector, ::LocalGeometry{(1,)}) = norm_sqr(u.u³)
_norm_sqr(u::Contravariant3Vector, ::LocalGeometry{(1, 2)}) = norm_sqr(u.u³)

function _norm_sqr(
    u::Tensor{2, T, <:Tuple{Basis{Orthonormal, <:Any}, Basis{Orthonormal, <:Any}}},
    ::LocalGeometry,
) where {T}
    norm_sqr(parent(u))
end

_norm(u::AbstractTensor, lg::LocalGeometry) = sqrt(_norm_sqr(u, lg))

function _cross(
    u::AbstractTensor{1}, v::AbstractTensor{1}, lg::LocalGeometry,
)
    _cross(ContravariantVector(u, lg), ContravariantVector(v, lg), lg)
end

function _cross(
    x::ContravariantVector, y::ContravariantVector, lg::LocalGeometry,
)
    lg.J * Covariant123Vector(
        x.u² * y.u³ - x.u³ * y.u²,
        x.u³ * y.u¹ - x.u¹ * y.u³,
        x.u¹ * y.u² - x.u² * y.u¹,
    )
end
function _cross(
    x::Contravariant12Vector, y::Contravariant12Vector, lg::LocalGeometry,
)
    lg.J * Covariant3Vector(x.u¹ * y.u² - x.u² * y.u¹)
end
function _cross(
    x::Contravariant2Vector, y::Contravariant1Vector, lg::LocalGeometry,
)
    lg.J * Covariant3Vector(-x.u² * y.u¹)
end
function _cross(
    x::Contravariant12Vector, y::Contravariant3Vector, lg::LocalGeometry,
)
    lg.J * Covariant12Vector(x.u² * y.u³, -x.u¹ * y.u³)
end
function _cross(
    x::Contravariant3Vector, y::Contravariant12Vector, lg::LocalGeometry,
)
    lg.J * Covariant12Vector(-x.u³ * y.u², x.u³ * y.u¹)
end
function _cross(
    x::Contravariant2Vector, y::Contravariant3Vector, lg::LocalGeometry,
)
    lg.J * Covariant1Vector(x.u² * y.u³)
end

_cross(u::OrthonormalTensor, v::OrthonormalTensor, ::LocalGeometry) =
    cross(u, v)
