##############################################
## Basis Vectors in Generalized Coordinates ##
##############################################

abstract type BasisType end

# FIXME: Swap Covariant and Contravariant definitions in future breaking release
# (current definition is based on how components transform, not basis vectors)
struct Covariant <: BasisType end     # Basis vector i is given by eⁱ = ∇ξⁱ
struct Contravariant <: BasisType end # Basis vector i is given by eᵢ = ∂r/∂ξⁱ
struct Orthonormal <: BasisType end   # Any basis of orthogonal unit vectors
struct OneScalar <: BasisType end     # Basis for scalar field of a vector space

dual_basis_type(::Covariant) = Contravariant()
dual_basis_type(::Contravariant) = Covariant()
dual_basis_type(::Orthonormal) = Orthonormal()
dual_basis_type(::OneScalar) = OneScalar()

abstract type AbstractBasis end

"""
    Basis{T <: BasisType, names}()
    Basis(basis_type::BasisType, names::Tuple)

Type-level description of a single tensor axis. The parameter `T` selects the
kind of basis ([`Covariant`](@ref), [`Contravariant`](@ref),
[`Orthonormal`](@ref), or [`OneScalar`](@ref)), and `names` is a tuple of
identifiers for the basis vectors along the axis (typically dimension indices
like `(1, 3)` for the ξ¹/ξ³ directions, or `(nothing,)` for the scalar row
of a covector; see [`ScalarBasis`](@ref)).

A `Basis` is a singleton: all information lives in the type parameters, so
instances are free at runtime and available for multiple dispatch. Named aliases
such as `Covariant13Axis`, `UWAxis`, and `ScalarBasis` are defined for every
supported dimension combination.

# Role in `reshape`

`Basis` objects (as opposed to bare `BasisType`s) carry the basis-vector
names, so they are what lets `reshape` reorder, drop, or zero-fill components
along an axis. Passing `Bases` to `reshape` cannot change the underlying
`BasisType` of a concrete `Tensor`; for that, use the basis-conversion
helpers in `conversions.jl` (e.g. `project`, `transform`), which apply the
appropriate metric from a `LocalGeometry`.

# Examples
```julia
julia> Covariant13Axis()
Covariant13Basis()

julia> length(Covariant13Axis())
2

julia> dual(Covariant13Axis())
Contravariant13Basis()

# reshape(tensor, (Basis, ...)) reorders and zero-fills by `names`:
julia> v = Covariant12Vector(1.0, 2.0);

julia> reshape(v, (Covariant123Axis(),))         # zero-fill the missing u₃
Tensor([1.0, 2.0, 0.0], (Covariant123Basis(),))

julia> reshape(v, (Covariant2Axis(),))           # drop u₁, keep u₂
Tensor([2.0], (Covariant2Basis(),))
```
"""
struct Basis{T <: BasisType, names} <: AbstractBasis end
const ScalarBasis = Basis{OneScalar, (nothing,)} # Used in row axes of covectors

Basis(::T, names) where {T} =
    unrolled_allunique(names) ? Basis{T, names}() :
    throw(ArgumentError("Basis vector names are not all unique: $names"))
Basis(::OneScalar, names) =
    names == (nothing,) ? ScalarBasis() :
    throw(ArgumentError("OneScalar basis must contain a single unnamed scalar"))

basis_type(::Basis{T}) where {T} = T()
basis_vector_names(::Basis{<:Any, names}) where {names} = names

dual(b::Basis) = Basis(dual_basis_type(basis_type(b)), basis_vector_names(b))

Base.length(b::Basis) = length(basis_vector_names(b))

# Extend internal Base.unitrange to support the default show(io, mime, ::Tensor)
Base.unitrange(b::Basis) = Base.OneTo(length(b))

Base.show(io::IO, b::Basis) =
    print(io, typeof(basis_type(b)), join(basis_vector_names(b)), "Basis()")
Base.show(io::IO, ::ScalarBasis) = print(io, "ScalarBasis()")

no_metric_error(T1, T2) =
    throw(DimensionMismatch("Metric is needed for change of basis: $T1 vs $T2"))
scalar_error(T) =
    throw(DimensionMismatch("Incompatible bases: one scalar vs $T vectors"))

check_same_type(::T1, ::T2) where {T1, T2} = T1 == T2 || no_metric_error(T1, T2)
check_same_type(::OneScalar, ::T) where {T} = scalar_error(T)
check_same_type(::T, ::OneScalar) where {T} = scalar_error(T)
check_same_type(::OneScalar, ::OneScalar) = true

combine_bases(b::B, ::B) where {B <: Basis} = b
combine_bases(b1::Basis, b2::Basis) =
    check_same_type(basis_type(b1), basis_type(b2)) &&
    Basis(
        basis_type(b1),
        unrolled_unique((basis_vector_names(b1)..., basis_vector_names(b2)...)),
    )
combine_bases(b::Basis, bases::Basis...) =
    unrolled_reduce(combine_bases, bases; init = b)
function overlap_bases(b1::Basis, b2::Basis)
    check_same_type(basis_type(b1), basis_type(b2))
    names2 = basis_vector_names(b2)
    overlap = unrolled_filter(n -> unrolled_in(n, names2), basis_vector_names(b1))
    return Basis(basis_type(b1), overlap)
end

# Indices of vectors in src_basis matching the vectors in dest_basis, with
# `nothing` denoting vectors present in dest_basis but missing from src_basis
function matching_basis_vector_indices(dest_basis, src_basis)
    check_same_type(basis_type(dest_basis), basis_type(src_basis))
    return unrolled_map(
        Base.Fix2(unrolled_findfirst, basis_vector_names(src_basis)) ∘ ==,
        basis_vector_names(dest_basis),
    )
end

########################################
## Tensors in Generalized Coordinates ##
########################################

const AbstractBases{N} = NTuple{N, AbstractBasis}
const Bases{N} = NTuple{N, Basis}
const BasisTypes{N} = NTuple{N, BasisType}

# Generic tensor whose components can be expressed in different bases; stores
# its bases as a type parameter to facilitate basis-dependent multiple dispatch
abstract type AbstractTensor{N, T, B <: AbstractBases{N}} <: AbstractArray{T, N} end

"""
    Tensor(components, bases::NTuple{N, Basis})
    Tensor(s::UniformScaling, bases::NTuple{2, Basis})

`N`-dimensional tensor whose entries in `components` are interpreted with
respect to the given `bases`. Each entry of `bases` is a [`Basis`](@ref), and
the shape of `components` must match `length.(bases)`. The bases are
stored as a type parameter so that operations can dispatch on the kind of
basis (covariant, contravariant, orthonormal, or scalar) at compile time.

Use `parent(x)` to get the component array and `axes(x)` to get the bases.
Scalar indexing `x[i, j, ...]` reads directly from the component array;
colon-indexing yields a smaller `Tensor` over the remaining non-colon axes.

# Shapes that `Tensor` takes in practice

- `Tensor{1}` (a vector): `components::SVector`, `bases::Tuple{Basis}`.
- `Tensor{2}` covector (a row-vector): the first axis is [`ScalarBasis`](@ref)
  and `components::Adjoint{T, SVector}`. This is what `v'` produces for a
  `Tensor{1}` `v`.
- `Tensor{2}` square tensor: `components::SMatrix`.

The `UniformScaling` constructor is a convenience that converts
`s = λ * I` (where `λ = s.λ` is the scalar stored in Julia's
`LinearAlgebra.UniformScaling`) into an `SMatrix` of the appropriate size —
e.g., `Tensor(2I, (b, b))` builds a diagonal tensor with `2` on the diagonal.

# Role in `reshape`

`reshape(x::Tensor, bases::NTuple{N, Basis})` changes basis-vector
*names* along each axis, zero-filling gaps. Changing the `BasisType` of a
concrete `Tensor` is not possible through `reshape` alone; attempting it
throws a `DimensionMismatch`:

```julia
julia> reshape(Covariant12Vector(1.0, 2.0), (Contravariant12Axis(),))
ERROR: DimensionMismatch: Metric is needed for change of basis: Covariant vs Contravariant
```

To change basis types, use `project` / `transform` from `conversions.jl`,
which apply the appropriate metric from a `LocalGeometry`.

# Examples
```julia
julia> v = Covariant12Vector(1.0, 2.0)
Tensor([1.0, 2.0], (Covariant12Basis(),))

julia> parent(v)
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 2.0

julia> axes(v)
(Covariant12Basis(),)

julia> v[1], v.u₁                                # indexed and named access
(1.0, 1.0)

julia> reshape(v, (Covariant123Axis(),))         # names-only reshape
Tensor([1.0, 2.0, 0.0], (Covariant123Basis(),))
```
"""
# Tensor represented by its components in a specific set of bases, which can be
# reshaped to have new basis_vector_names, but cannot be given new BasisTypes
struct Tensor{N, T, B <: Bases{N}, C} <: AbstractTensor{N, T, B}
    components::C
    bases::B
end

Tensor(components::C, bases::B) where {C, B} =
    size(components) == unrolled_map(length, bases) ?
    Tensor{ndims(C), eltype(C), B, C}(components, bases) :
    throw(DimensionMismatch("Tensor component array size, $(size(components)), \
                             does not match dimensions of tensor \
                             bases, $(unrolled_map(length, bases))"))

# Allow UniformScaling as components for square 2-tensors (converted into SMatrix)
function Tensor(s::UniformScaling, bases::NTuple{2, Basis})
    N1 = length(bases[1])
    N2 = length(bases[2])
    N1 == N2 ||
        throw(DimensionMismatch("UniformScaling requires square tensor, got $(N1)×$(N2)"))
    T = typeof(s.λ)
    Tensor(SMatrix{N1, N2, T}(s), bases)
end

Base.parent(x::Tensor) = x.components
Base.axes(x::Tensor) = x.bases

@inline _unwrap(t::UnionAll) = _unwrap(t.body)
@inline _unwrap(t::DataType) = t

@inline tensor_bases(::Type{T}) where {T <: Tensor} =
    _unwrap(T).parameters[3].instance


Base.zero(x::Tensor) = zero(typeof(x))
Base.one(x::Tensor) = one(typeof(x))

Base.zero(::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(zero(C), B.instance)
# The Covector representation (Tensor{2} with ScalarBasis row) stores its
# components as an `Adjoint{T, SVector}`; since that type has no type-level
# zero, we unwrap to the SVector, zero it, then re-wrap with adjoint.
Base.zero(::Type{Tensor{N, T, B, Adjoint{T, P}}}) where {N, T, B, P} =
    Tensor(adjoint(zero(P)), B.instance)
Base.one(::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(one(C), B.instance)
Base.convert(::Type{Tensor{N, T, B, C}}, x::AbstractTensor) where {N, T, B, C} =
    Tensor(convert(C, parent(reshape(x, B.instance))), B.instance)
Random.rand(rng::Random.AbstractRNG, ::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(rand(rng, C), B.instance)

Base.show(io::IO, x::Tensor) =
    print(io, "Tensor(", parent(x), ", ", axes(x), ")")

Base.size(x::Tensor) = unrolled_map(length, axes(x))

Base.@propagate_inbounds Base.getindex(x::Tensor, indices::Integer...) =
    parent(x)[indices...]
Base.@propagate_inbounds Base.view(x::Tensor, indices::Integer...) =
    view(parent(x), indices...)
Base.@propagate_inbounds Base.isassigned(x::Tensor, indices::Integer...) =
    isassigned(parent(x), indices...)
Base.@propagate_inbounds Base.setindex!(x::Tensor, v, indices::Integer...) =
    parent(x)[indices...] = v
Base.@propagate_inbounds Base.setindex(x::Tensor, v, indices::Integer...) =
    Tensor(Base.setindex(parent(x), v, indices...), axes(x))

const TensorIndex = Union{Colon, Integer}

function bases_at_colons(bases, indices)
    basis_index_pairs = unrolled_map(tuple, bases, indices)
    basis_colon_pairs = unrolled_filter(==(Colon()) ∘ last, basis_index_pairs)
    return unrolled_map(first, basis_colon_pairs)
end

Base.@propagate_inbounds Base.getindex(x::Tensor, indices::TensorIndex...) =
    Tensor(parent(x)[indices...], bases_at_colons(axes(x), indices))
Base.@propagate_inbounds Base.view(x::Tensor, indices::TensorIndex...) =
    Tensor(view(parent(x), indices...), bases_at_colons(axes(x), indices))

#############################################
## Metric Terms in Generalized Coordinates ##
#############################################

"""
    Metric(tensor::AbstractTensor{2})

Storage wrapper around the canonical local metric tensor `∂x/∂ξ`
(`Orthonormal` rows × `Covariant` columns). Held as a field of
[`LocalGeometry`](@ref); read directly via `lg.∂x∂ξ`. The wrapped tensor is
identity-padded to full `(UVWAxis, Covariant123Axis)` shape regardless of
the source geometry's `I`, so a single matvec covers every conversion case
— directions outside `I` ride the identity block. See [`pad_metric_tensor`](@ref).
"""
struct Metric{T <: AbstractTensor{2}}
    tensor::T
end

Base.zero(g::Metric) = zero(typeof(g))
Base.one(g::Metric) = one(typeof(g))

Base.zero(::Type{Metric{T}}) where {T} = Metric(zero(T))
Base.one(::Type{Metric{T}}) where {T} = Metric(one(T))
Base.convert(::Type{Metric{T}}, g::Metric) where {T} =
    Metric(convert(T, g.tensor))

Base.show(io::IO, g::Metric) = print(io, "Metric(", g.tensor, ")")

"""
    pad_metric_tensor(∂x∂ξ::Tensor{2})

Pads an N×N metric tensor with axes `(Basis{Orthonormal, I}, Basis{Covariant, I})`
to a full 3×3 tensor with axes `(UVWAxis, Covariant123Axis)`, putting `1`
on diagonal entries for dimensions outside `I` and `0` on cross-coupling
entries. The padded form encodes the "identity metric in directions
orthogonal to `I`" convention as actual matrix entries, so a single matvec
`padded_M * v` covers all source-name configurations without partition
logic. Idempotent for `I == (1, 2, 3)`.
"""
function pad_metric_tensor(∂x∂ξ::Tensor{2})
    src_names = basis_vector_names(axes(∂x∂ξ, 1))
    src_names == (1, 2, 3) && return ∂x∂ξ
    full_bases = (UVWAxis(), Covariant123Axis())
    # `reshape` to the full bases zero-fills rows/cols whose name isn't in
    # `src_names`. We then add `1` on diagonal entries at dims not in
    # `src_names` to recover the identity-padding convention.
    padded_zeros = reshape(∂x∂ξ, full_bases)
    iso = _orthogonal_identity(Val(src_names), eltype(∂x∂ξ))
    return Tensor(parent(padded_zeros) + iso, full_bases)
end

# 3×3 SMatrix with `1` on the diagonal at positions outside `src_names`,
# `0` elsewhere. `unrolled_in` is compile-time foldable for tuple
# arguments, so each diagonal entry resolves to a literal at the call
# site and the SMatrix is built without runtime branching.
@inline function _orthogonal_identity(
    ::Val{src_names}, ::Type{FT},
) where {src_names, FT}
    z, o = zero(FT), one(FT)
    d1 = unrolled_in(1, src_names) ? z : o
    d2 = unrolled_in(2, src_names) ? z : o
    d3 = unrolled_in(3, src_names) ? z : o
    SMatrix{3, 3, FT, 9}(d1, z, z, z, d2, z, z, z, d3)
end

#################################
## Covector and Vector Aliases ##
#################################

const AbstractCovector =
    AbstractTensor{2, <:Any, <:Tuple{ScalarBasis, AbstractBasis}}
const Covector = Tensor{2, <:Any, <:Tuple{ScalarBasis, Basis}}

#####################################
## Modifying the Bases of a Tensor ##
#####################################

check_ndims(x, N) =
    ndims(x) == N ||
    throw(DimensionMismatch("Cannot reshape $(ndims(x))-tensor into $N-tensor"))

# Change the basis_vector_names, and, if possible, change the BasisTypes as well
function Base.reshape(x::Tensor, bases::Bases)
    check_ndims(x, length(bases))
    axes(x) == bases && return x
    components_constructor = SArray{Tuple{unrolled_map(length, bases)...}, eltype(x)}
    component_indices = unrolled_product(
        unrolled_map(matching_basis_vector_indices, bases, axes(x))...,
    )
    component_values = unrolled_map(component_indices) do indices
        unrolled_any(isnothing, indices) ? zero(eltype(x)) : x[indices...]
    end
    return Tensor(components_constructor(component_values), bases)
end
Base.reshape(x::AbstractTensor, bases::Basis...) = reshape(x, bases)

# Change the BasisTypes without constraining the basis_vector_names
Base.reshape(x::Tensor, types::BasisTypes) =
    check_ndims(x, length(types)) &&
    unrolled_map(basis_type, axes(x)) == types ? x :
    throw(DimensionMismatch("Metric is needed for change of basis: \
                             $(unrolled_map(basis_type, axes(x))) vs $types"))
Base.reshape(x::AbstractTensor, types::BasisType...) = reshape(x, types)

# Change all bases to a single BasisType
Base.reshape(x::AbstractTensor, type::BasisType) =
    reshape(x, ntuple(Returns(type), Val(ndims(x))))

reshape_for_norm(x::AbstractTensor) = reshape(x, Orthonormal())
reshape_for_norm(x::AbstractCovector) = reshape_for_norm(x')

########################################
## Other Generic Tensor Manipulations ##
########################################

apply_f(f, x::Tensor) = Tensor(f(parent(x)), axes(x))

function reshape_and_apply_f(f::F, args...) where {F}
    unrolled_foreach(Base.Fix2(check_ndims, ndims(args[1])), args)
    bases = unrolled_map(combine_bases, unrolled_map(axes, args)...)
    components = f(unrolled_map(x -> parent(reshape(x, bases)), args)...)
    components isa AbstractArray || return components  # return scalar values
    return Tensor(components, bases)               # add bases to non-scalars
end


Base.map(f::F, args::AbstractTensor...) where {F} =
    reshape_and_apply_f((xs...) -> map(f, xs...), args...)

# The contracted-axis basis for `x * y`.
new_basis_for_product(x, y) = overlap_bases(axes(x, ndims(x)), dual(axes(y, 1)))

x_and_y_bases_for_product(x, y) = (
    Base.setindex(axes(x), new_basis_for_product(x, y), ndims(x)),
    Base.setindex(axes(y), dual(new_basis_for_product(x, y)), 1),
)

###################################
## Math Operations on One Tensor ##
###################################

Base.:-(x::AbstractTensor) = apply_f(-, x)
Base.:*(x::AbstractTensor, a::Number) = apply_f(Base.Fix2(*, a), x)
Base.:/(x::AbstractTensor, a::Number) = apply_f(Base.Fix2(/, a), x)
Base.:*(a::Number, x::AbstractTensor) = apply_f(Base.Fix1(*, a), x)
Base.:\(a::Number, x::AbstractTensor) = apply_f(Base.Fix1(\, a), x)

# Covector storage uses `Adjoint{T, SVector}`. LinearAlgebra's
# `Number * Adjoint{<:Any, <:AbstractVector}` path produces a type whose
# storage layout doesn't match what broadcast eltype inference computes,
# triggering dynamic dispatch on GPU when filling a typed destination
# (e.g. `BidiagonalMatrixRow{Tensor{2, T, _, Adjoint{T, SVector}}}`). Route
# through the underlying `SVector` so the result re-wraps as `Adjoint{Tnew, SVector{Tnew}}`.
Base.:*(a::Number, x::Tensor{N, T, B, <:Adjoint}) where {N, T, B} =
    Tensor(adjoint(a * adjoint(parent(x))), axes(x))
Base.:*(x::Tensor{N, T, B, <:Adjoint}, a::Number) where {N, T, B} =
    Tensor(adjoint(adjoint(parent(x)) * a), axes(x))
Base.:/(x::Tensor{N, T, B, <:Adjoint}, a::Number) where {N, T, B} =
    Tensor(adjoint(adjoint(parent(x)) / a), axes(x))
Base.:\(a::Number, x::Tensor{N, T, B, <:Adjoint}) where {N, T, B} =
    Tensor(adjoint(a \ adjoint(parent(x))), axes(x))

Base.adjoint(x::Covector) = Tensor(parent(x)', (axes(x, 2),))
Base.adjoint(x::Tensor{1}) = Tensor(parent(x)', (ScalarBasis(), axes(x, 1)))
Base.adjoint(x::Tensor{2}) = Tensor(parent(x)', (axes(x, 2), axes(x, 1)))

Base.inv(x::Tensor{2}) =
    Tensor(inv(parent(x)), (dual(axes(x, 2)), dual(axes(x, 1))))

function Base.:^(x::AbstractTensor{2}, p::Integer)
    p == 1 && return x
    p < 0 && return inv(x)^(-p)
    p == 0 && return Tensor(one(parent(x)), (dual(axes(x, 2)), axes(x, 2)))
    bases = (dual(new_basis_for_product(x, x)), new_basis_for_product(x, x))
    return apply_f(Base.Fix2(^, p), reshape(x, bases))
end

norm(x::AbstractTensor, p::Real = 2) = norm(parent(reshape_for_norm(x)), p)
norm_sqr(x::AbstractTensor) = norm_sqr(parent(reshape_for_norm(x)))

#########################################
## Math Operations on Multiple Tensors ##
#########################################


Base.:+(args::AbstractTensor...) = reshape_and_apply_f(_add_components, args...)
Base.:-(x::AbstractTensor, y::AbstractTensor) =
    reshape_and_apply_f(_sub_components, x, y)

# Apply +/- componentwise without going through Base's
# `+(::AbstractArray, ::AbstractArray)`, which calls `promote_shape` whose
# error branch is GPU-incompatible (string formatting & exception machinery).
# `SArray + SArray` already bypasses `promote_shape`, but covector storage
# (`Adjoint{T, SVector}`) inherits from AbstractMatrix and falls into Base's
# path. Route those through the underlying SVector to keep kernels clean.
@inline _add_components(xs::SArray...) = +(xs...)
@inline _sub_components(x::SArray, y::SArray) = x - y
@inline _add_components(xs::Adjoint{<:Any, <:SVector}...) =
    adjoint(+(unrolled_map(adjoint, xs)...))
@inline _sub_components(x::Adjoint{<:Any, <:SVector}, y::Adjoint{<:Any, <:SVector}) =
    adjoint(adjoint(x) - adjoint(y))
Base.:(==)(x::AbstractTensor, y::AbstractTensor) = reshape_and_apply_f(==, x, y)
Base.isapprox(x::AbstractTensor, y::AbstractTensor; kwargs...) =
    reshape_and_apply_f((x, y) -> isapprox(x, y; kwargs...), x, y)

Base.:*(::AbstractTensor{1}, ::AbstractTensor) =
    throw(ArgumentError("Adjoint is needed to multiply a vector by a tensor"))
function Base.:*(x::AbstractTensor{2}, y::AbstractTensor)
    (x_bases, y_bases) = x_and_y_bases_for_product(x, y)
    components = parent(reshape(x, x_bases)) * parent(reshape(y, y_bases))
    return Tensor(components, Base.setindex(axes(y), axes(x, 1), 1))
end

Base.:*(x::AbstractCovector, y::AbstractTensor{1}) = dot(x', y)
function dot(x::AbstractTensor{1}, y::AbstractTensor{1})
    (x_bases, y_bases) = x_and_y_bases_for_product(x, y)
    return dot(parent(reshape(x, x_bases)), parent(reshape(y, y_bases)))
end

Base.:*(x::Tensor{1}, y::Covector) =
    Tensor(parent(x) * parent(y), (axes(x, 1), axes(y, 2)))

Base.:/(x::AbstractTensor, y::AbstractTensor{2}) = x * inv(y)
Base.:\(y::AbstractTensor{2}, x::AbstractTensor) = inv(y) * x

###############################################
## Concrete Aliases, Constructors, & Extras ##
###############################################

# coordinate_axis: maps point types to basis vector name tuples
coordinate_axis(::Type{<:XPoint}) = (1,)
coordinate_axis(::Type{<:YPoint}) = (2,)
coordinate_axis(::Type{<:ZPoint}) = (3,)
coordinate_axis(::Type{<:XYPoint}) = (1, 2)
coordinate_axis(::Type{<:XZPoint}) = (1, 3)
coordinate_axis(::Type{<:XYZPoint}) = (1, 2, 3)
coordinate_axis(::Type{<:Cartesian1Point}) = (1,)
coordinate_axis(::Type{<:Cartesian2Point}) = (2,)
coordinate_axis(::Type{<:Cartesian3Point}) = (3,)
coordinate_axis(::Type{<:Cartesian123Point}) = (1, 2, 3)
coordinate_axis(::Type{<:LatLongZPoint}) = (1, 2, 3)
coordinate_axis(::Type{<:Cartesian13Point}) = (1, 3)
coordinate_axis(::Type{<:LatLongPoint}) = (1, 2)
coordinate_axis(coord::AbstractPoint) = coordinate_axis(typeof(coord))

# Generic vector/tensor type aliases
const CovariantVector{T, I, S} = Tensor{1, T, Tuple{Basis{Covariant, I}}, S}
const ContravariantVector{T, I, S} = Tensor{1, T, Tuple{Basis{Contravariant, I}}, S}
const LocalVector{T, I, S} = Tensor{1, T, Tuple{Basis{Orthonormal, I}}, S}

# Union types for dispatch
const CovariantTensor = Union{
    Tensor{1, <:Any, <:Tuple{Basis{Covariant}}},
    Tensor{2, <:Any, <:Tuple{Basis{Covariant}, <:AbstractBasis}},
}
const ContravariantTensor = Union{
    Tensor{1, <:Any, <:Tuple{Basis{Contravariant}}},
    Tensor{2, <:Any, <:Tuple{Basis{Contravariant}, <:AbstractBasis}},
}
const OrthonormalTensor = Union{
    Tensor{1, <:Any, <:Tuple{Basis{Orthonormal}}},
    Tensor{2, <:Any, <:Tuple{Basis{Orthonormal}, <:AbstractBasis}},
}

# Concrete axis and vector aliases for all dimension combinations
for I in [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    strI = isempty(I) ? "Null" : join(I)
    N = length(I)
    strUVW = isempty(I) ? "Null" : join(map(i -> [:U, :V, :W][i], I))

    # Axis aliases (Basis singletons)
    @eval const $(Symbol(:Covariant, strI, :Axis)) = Basis{Covariant, $I}
    @eval const $(Symbol(:Contravariant, strI, :Axis)) = Basis{Contravariant, $I}
    @eval const $(Symbol(strUVW, :Axis)) = Basis{Orthonormal, $I}

    # Vector aliases. Splatted constructors like `Covariant12Vector(1.0, 2.0)`
    # are handled by the generic `(::Type{T})(args::Number...)` defined below,
    # which uses the `@generated tensor_bases(T)` to extract the basis tuple
    # even when `T` is a UnionAll with free eltype/storage parameters.
    @eval const $(Symbol(:Covariant, strI, :Vector)){T} =
        CovariantVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Contravariant, strI, :Vector)){T} =
        ContravariantVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(strUVW, :Vector)){T} =
        LocalVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Cartesian, strI, :Vector)){T} =
        LocalVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Cartesian, strI, :Axis)) = Basis{Orthonormal, $I}
end

# Named property access for vectors (e.g., v.u₁, v.u², v.u)
_symbols(::Covariant) = (:u₁, :u₂, :u₃)
_symbols(::Contravariant) = (:u¹, :u², :u³)
_symbols(::Orthonormal) = (:u, :v, :w)

Base.propertynames(x::Tensor{1}) = _symbols(basis_type(x.bases[1]))

# `unrolled_findfirst` walks the compile-time `I` tuple, so the search
# inlines to a flat chain of `name === :u_k` comparisons that fold to a
# direct `parent(x)[idx]` read for the matching name or `zero(T)`.
@inline function Base.getproperty(
    x::Tensor{1, T, <:Tuple{Basis{BT, I}}}, name::Symbol,
) where {T, BT, I}
    name === :components && return getfield(x, :components)
    name === :bases && return getfield(x, :bases)
    syms = _symbols(BT())
    idx = unrolled_findfirst(I) do dim
        dim <= 3 && name === syms[dim]
    end
    return idx === nothing ? zero(T) :
           @inbounds getfield(x, :components)[idx]
end

(::Type{T})(args::Number...) where {T <: Tensor{1}} =
    Tensor(SVector(args...), tensor_bases(T))

"""
    project(basis, v)
    project(basis, v, local_geometry)

Project the first axis of vector/tensor `v` onto `basis`, zero-filling
components not present in the source. When `local_geometry` is provided,
performs a change of basis type (e.g., Covariant → Contravariant) via the metric.
"""
@inline project(b::Basis, v::AbstractTensor{1}) = reshape(v, (b,))
@inline project(b::Basis, v::AbstractTensor{2}) = reshape(v, (b, axes(v, 2)))
@inline function project(b::Basis, v::AbstractTensor{2}, b2::Basis)
    reshape(v, (b, b2))
end
@inline project(v::AbstractTensor{2}, b::Basis) = reshape(v, (axes(v, 1), b))

"""
    transform(basis, v)

Transform the first axis of vector or 2-tensor `v` to `basis`. Unlike
`project`, throws an `InexactError` if any dropped component is nonzero.
"""
@inline function transform(b::Basis, v::AbstractTensor{1})
    result = reshape(v, (b,))
    # Check that no nonzero components were dropped. `unrolled_all` over a
    # static index tuple lets the compiler elide the check entirely when
    # `src_names ⊆ dest_names` (so every iteration's left disjunct is `true`).
    src_names = basis_vector_names(axes(v, 1))
    dest_names = basis_vector_names(b)
    pv = parent(v)
    indices = ntuple(identity, Val(length(src_names)))
    unrolled_all(indices) do n
        unrolled_in(src_names[n], dest_names) || iszero(pv[n])
    end || throw(InexactError(:transform, typeof(b), v))
    return result
end
@inline function transform(b::Basis, v::AbstractTensor{2})
    result = reshape(v, (b, axes(v, 2)))
    # Same idea as the Tensor{1} case, with an inner unrolled_all over the
    # row entries: a dropped row only passes if every column entry is zero.
    src_names = basis_vector_names(axes(v, 1))
    dest_names = basis_vector_names(b)
    pv = parent(v)
    rows = ntuple(identity, Val(length(src_names)))
    cols = ntuple(identity, Val(size(pv, 2)))
    unrolled_all(rows) do n
        unrolled_in(src_names[n], dest_names) ||
            unrolled_all(c -> iszero(pv[n, c]), cols)
    end || throw(InexactError(:transform, typeof(b), v))
    return result
end

# outer product
"""
    outer(x, y)
    x ⊗ y

Compute the outer product of `x` and `y`.
"""
function outer end
const ⊗ = outer

@inline outer(x, y) = x * y'

# Cross product of two orthonormal vectors. Reshapes both inputs to the full
# UVW basis (zero-filling missing dims), then applies the standard 3D formula.
# Always returns a `UVWVector`, regardless of which orthonormal sub-bases the
# inputs occupy (UV × W, U × VW, UVW × UVW, etc.).
function cross(
    x::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal}}},
    y::Tensor{1, <:Any, <:Tuple{Basis{Orthonormal}}},
)
    a = reshape(x, (UVWAxis(),))
    b = reshape(y, (UVWAxis(),))
    return UVWVector(
        a.v * b.w - a.w * b.v,
        a.w * b.u - a.u * b.w,
        a.u * b.v - a.v * b.u,
    )
end

# UniformScaling support
function Base.:(+)(A::Tensor{2}, b::UniformScaling)
    Tensor(parent(A) + b, axes(A))
end
function Base.:(-)(A::Tensor{2}, b::UniformScaling)
    Tensor(parent(A) - b, axes(A))
end
function Base.:(+)(a::UniformScaling, B::Tensor{2})
    Tensor(a + parent(B), axes(B))
end
function Base.:(-)(a::UniformScaling, B::Tensor{2})
    Tensor(a - parent(B), axes(B))
end
