using UnrolledUtilities
import StaticArrays: SArray
import LinearAlgebra: AdjointAbsVec, det, dot, norm, norm_sqr

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
check_same_type(::OneScalar, ::T) where {T} = T == OneScalar || scalar_error(T)
check_same_type(::T, ::OneScalar) where {T} = T == OneScalar || scalar_error(T)

combine_bases(b1::Basis, b2::Basis) =
    check_same_type(basis_type(b1), basis_type(b2)) && Basis(
        basis_type(b1),
        unrolled_unique((basis_vector_names(b1)..., basis_vector_names(b2)...)),
    )
overlap_bases(b1::Basis, b2::Basis) =
    check_same_type(basis_type(b1), basis_type(b2)) && Basis(
        basis_type(b1),
        unrolled_filter(in(basis_vector_names(b2)), basis_vector_names(b1)),
    )

# Indices of vectors in src_basis matching the vectors in dest_basis, with
# `nothing` denoting vectors present in dest_basis but missing from src_basis
matching_basis_vector_indices(dest_basis, src_basis) =
    check_same_type(basis_type(dest_basis), basis_type(src_basis)) &&
    unrolled_map(
        Base.Fix2(unrolled_findfirst, basis_vector_names(src_basis)) ∘ ==,
        basis_vector_names(dest_basis),
    )

########################################
## Tensors in Generalized Coordinates ##
########################################

const AbstractBases{N} = NTuple{N, AbstractBasis}
const Bases{N} = NTuple{N, Basis}
const BasisTypes{N} = NTuple{N, BasisType}

# Generic tensor whose components can be expressed in different bases; stores
# its bases as a type parameter to facilitate basis-dependent multiple dispatch
abstract type AbstractTensor{N, T, B <: AbstractBases{N}} <: AbstractArray{T, N} end

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

Base.parent(x::Tensor) = x.components
Base.axes(x::Tensor) = x.bases

Base.zero(x::Tensor) = zero(typeof(x))
Base.one(x::Tensor) = one(typeof(x))

Base.zero(::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(zero(C), B.instance)
Base.one(::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(one(C), B.instance)
Base.convert(::Type{Tensor{N, T, B, C}}, x::AbstractTensor) where {N, T, B, C} =
    Tensor(convert(C, parent(reshape(x, B.instance))), B.instance)

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

# Riemannian metric represented by a transformation between two bases, which can
# be multiplied by a tensor/covector to replace a concrete Basis with AnyBasis
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

src_and_dest_types(row_type, col_type) = (dual_basis_type(col_type), row_type)
cob_arg_types(row_type, col_type, tensor) = (
    src_and_dest_types(row_type, col_type)...,
    src_and_dest_types(unrolled_map(basis_type, axes(tensor))...)...,
)

# Computes the 2-tensor that transforms a src_type basis into a dest_type basis,
# given a similar tensor that turns a g_src_type basis into a g_dest_type basis:
# - Orthonormal -> Covariant:     (∂r/∂ξ)'
# - Orthonormal -> Contravariant: (∂ξ/∂r)
# - Covariant -> Orthonormal:     (∂ξ/∂r)'
# - Contravariant -> Orthonormal: (∂r/∂ξ)
# - Covariant -> Contravariant:   (∂ξ/∂r) * (∂ξ/∂r)'
# - Contravariant -> Covariant:   (∂r/∂ξ)' * (∂r/∂ξ)
function cob_tensor(src_type, dest_type, g_src_type, g_dest_type, tensor) end

for (T1, T2) in ((:Covariant, :Contravariant), (:Contravariant, :Covariant))
    T3 = :Orthonormal
    @eval cob_requires_inv(::$T1, ::$T3, ::$T2, ::$T3) = true
    @eval cob_requires_inv(::$T3, ::$T1, ::$T3, ::$T2) = true
    @eval cob_requires_inv(::$T1, ::$T2, ::$T3, ::$T1) = true
    @eval cob_requires_inv(::$T1, ::$T2, ::$T2, ::$T3) = true
    @eval cob_tensor(::$T1, ::$T3, ::$T3, ::$T2, tensor) = tensor'
    @eval cob_tensor(::$T3, ::$T1, ::$T2, ::$T3, tensor) = tensor'
    @eval cob_tensor(::$T1, ::$T3, ::$T2, ::$T3, tensor) = inv(tensor')
    @eval cob_tensor(::$T3, ::$T1, ::$T3, ::$T2, tensor) = inv(tensor')
    @eval cob_tensor(::$T1, ::$T2, ::$T3, ::$T2, tensor) = tensor * tensor'
    @eval cob_tensor(::$T1, ::$T2, ::$T1, ::$T3, tensor) = tensor' * tensor
    @eval cob_tensor(::$T1, ::$T2, ::$T3, ::$T1, tensor) = inv(tensor * tensor')
    @eval cob_tensor(::$T1, ::$T2, ::$T2, ::$T3, tensor) = inv(tensor' * tensor)
end

cob_requires_inv(_, _, _, _) = false
cob_requires_inv(::T1, ::T2, ::T2, ::T1) where {T1, T2} = true
cob_tensor(::T1, ::T1, _, _, _) where {T1} = 1
cob_tensor(::T1, ::T1, ::T1, ::T1, _) where {T1} = 1 # needed to avoid ambiguity
cob_tensor(::T1, ::T2, ::T1, ::T2, tensor) where {T1, T2} = tensor
cob_tensor(::T1, ::T2, ::T2, ::T1, tensor) where {T1, T2} = inv(tensor)
cob_tensor(::T1, ::T2, ::T3, ::T4, tensor) where {T1, T2, T3, T4} =
    throw(DimensionMismatch("Cannot compute $T1-to-$T2 change of basis tensor \
                             from $T3-to-$T4 metric representation $tensor"))

# Computes the change-of-basis tensor with the given row and column basis types
change_of_basis_tensor(g, row_type, col_type) =
    cob_tensor(cob_arg_types(row_type, col_type, g.tensor)..., g.tensor)

# Determines whether an inverse is needed to compute the change-of-basis tensor
change_of_basis_tensor_requires_inverse(g, row_type, col_type) =
    cob_requires_inv(cob_arg_types(row_type, col_type, g.tensor)...)

# Computes the determinant J = |∂r/∂ξ| (part of the volume element δV = W * J),
# minimizing operation count by computing inv(|∂ξ/∂r)|) instead of |inv(∂ξ/∂r))|
jacobian_determinant(g) =
    change_of_basis_tensor_requires_inverse(g, Contravariant(), Orthonormal()) ?
    det(parent(change_of_basis_tensor(g, Orthonormal(), Covariant()))) :
    inv(det(parent(change_of_basis_tensor(g, Contravariant(), Orthonormal()))))

#############################################
## Tensors with Arbitrary Coordinate Bases ##
#############################################

struct AnyBasis <: AbstractBasis end

dual(::AnyBasis) = AnyBasis()
combine_bases(::AnyBasis, ::AnyBasis) = AnyBasis()
overlap_bases(::AnyBasis, ::AnyBasis) = AnyBasis()
combine_bases(::AnyBasis, b::Basis) = b
overlap_bases(::AnyBasis, b::Basis) = b
combine_bases(b::Basis, ::AnyBasis) = b
overlap_bases(b::Basis, ::AnyBasis) = b

# Tensor with AnyBasis assigned to its axis at index n, which can be reshaped to
# have any BasisType for that axis (and any basis_vector_names for all its axes)
struct TensorWithAnyBasis{n, N, T, B, X <: AbstractTensor{N, T}, G <: Metric} <:
       AbstractTensor{N, T, B}
    x::X
    g::G
end

function TensorWithAnyBasis{n}(x::X, g::G) where {n, X, G}
    bases = Base.setindex(axes(x), AnyBasis(), n)
    return TensorWithAnyBasis{n, ndims(X), eltype(X), typeof(bases), X, G}(x, g)
end

Base.parent((; x)::TensorWithAnyBasis) = x
Base.axes((; x)::TensorWithAnyBasis{n}) where {n} =
    Base.setindex(axes(x), AnyBasis(), n)

Base.zero((; x, g)::TensorWithAnyBasis{n}) where {n} =
    TensorWithAnyBasis{n}(zero(x), g)
Base.one((; x, g)::TensorWithAnyBasis{n}) where {n} =
    TensorWithAnyBasis{n}(one(x), g)

Base.convert(
    ::Type{TensorWithAnyBasis{n, N, T, B, X, G}},
    (; x, g)::TensorWithAnyBasis{n},
) where {n, N, T, B, X, G} = TensorWithAnyBasis{n}(convert(X, x), convert(G, g))

Base.show(io::IO, (; x, g)::TensorWithAnyBasis{n}) where {n} =
    print(io, "TensorWithAnyBasis{", n, "}(", x, ", ", g, ")")

show_inner_tensor(io, x::Tensor, depth) = print(io, "    "^depth, x)
function show_inner_tensor(io, (; x, g)::TensorWithAnyBasis{n}, depth) where {n}
    print(io, "    "^depth, "TensorWithAnyBasis{", n, "}(\n")
    show_inner_tensor(io, x, depth + 1)
    print(io, ",\n", "    "^(depth + 1), g, ",\n", "    "^depth, ")")
end

# Specialize on mime type to bypass the default show(io, mime, ::AbstractArray)
Base.show(io::IO, ::MIME"text/plain", x::TensorWithAnyBasis) =
    show_inner_tensor(io, x, 0)

#################################
## Covector and Vector Aliases ##
#################################

const AbstractCovector =
    AbstractTensor{2, <:Any, <:Tuple{ScalarBasis, AbstractBasis}}
const Covector = Tensor{2, <:Any, <:Tuple{ScalarBasis, Basis}}
const CovectorWithAnyBasis =
    TensorWithAnyBasis{2, 2, <:Any, <:Tuple{ScalarBasis, AnyBasis}}

# AbstractVector and Vector are already defined in base Julia
const VectorWithAnyBasis = TensorWithAnyBasis{1, 1, <:Any, <:Tuple{AnyBasis}}

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
    components_constructor = SArray{Tuple{unrolled_map(length, bases)...}}
    component_indices = unrolled_product(
        unrolled_map(matching_basis_vector_indices, bases, axes(x))...,
    )
    component_values = unrolled_map(component_indices) do indices
        unrolled_any(isnothing, indices) ? zero(eltype(x)) : x[indices...]
    end
    return Tensor(components_constructor(component_values), bases)
end
Base.reshape(x::TensorWithAnyBasis, bases::Bases) =
    reshape(reshape(x, unrolled_map(basis_type, bases)), bases)
Base.reshape(x::AbstractTensor, bases::Basis...) = reshape(x, bases)

# Change the BasisTypes without constraining the basis_vector_names
Base.reshape(x::Tensor, types::BasisTypes) =
    check_ndims(x, length(types)) &&
    unrolled_map(basis_type, axes(x)) == types ? x :
    throw(DimensionMismatch("Metric is needed for change of basis: \
                             $(unrolled_map(basis_type, axes(x))) vs $types"))
function Base.reshape((; x, g)::TensorWithAnyBasis{1}, types::BasisTypes)
    check_ndims(x, length(types))
    col_type = basis_type(dual(axes(x, 1)))
    return reshape(change_of_basis_tensor(g, types[1], col_type) * x, types)
end
function Base.reshape((; x, g)::TensorWithAnyBasis{2}, types::BasisTypes)
    check_ndims(x, length(types))
    row_type = basis_type(dual(axes(x, 2)))
    return reshape(x * change_of_basis_tensor(g, row_type, types[2]), types)
end
Base.reshape(x::AbstractTensor, types::BasisType...) = reshape(x, types)

# Change all bases to a single BasisType
Base.reshape(x::AbstractTensor, type::BasisType) =
    reshape(x, ntuple(Returns(type), Val(ndims(x))))

reshape_for_norm(x::AbstractTensor) = reshape(x, Orthonormal())
reshape_for_norm(x::AbstractCovector) = reshape_for_norm(x')

# Unwrap every TensorWithAnyBasis in x that corresponds to an AnyBasis in bases
drop_any_bases(x) = drop_any_bases(x, axes(x))
drop_any_bases(x::Tensor, bases) = x
drop_any_bases((; x, g)::TensorWithAnyBasis{n}, bases) where {n} =
    bases[n] == AnyBasis() ? drop_any_bases(x, bases) :
    TensorWithAnyBasis{n}(drop_any_bases(x, bases), g)

# Ensure that the result has a TensorWithAnyBasis for every AnyBasis in bases
add_any_bases(result, x) = add_any_bases(result, x, axes(x))
add_any_bases(result, x::Tensor, bases) = result
add_any_bases(result, (; x, g)::TensorWithAnyBasis{n}, bases) where {n} =
    axes(result, n) == AnyBasis() || bases[n] != AnyBasis() ?
    add_any_bases(result, x, bases) :
    TensorWithAnyBasis{n}(add_any_bases(result, x, bases), g)

########################################
## Other Generic Tensor Manipulations ##
########################################

apply_f(f, x::Tensor) = Tensor(f(parent(x)), axes(x))
apply_f(f, (; x, g)::TensorWithAnyBasis{n}) where {n} =
    TensorWithAnyBasis{n}(f(x), g)

function reshape_and_apply_f(f::F, args...) where {F}
    unrolled_foreach(Base.Fix2(check_ndims, ndims(args[1])), args)
    bases = unrolled_map(combine_bases, unrolled_map(axes, args)...)
    if unrolled_in(AnyBasis(), bases)
        (x, etc...) = args
        tensor = reshape_and_apply_f(f, drop_any_bases(x, bases), etc...)
        tensor isa AbstractTensor || return tensor # return scalar values
        return add_any_bases(tensor, x, bases) # add bases to non-scalars
    end
    components = f(unrolled_map(parent ∘ Base.Fix2(reshape, bases), args)...)
    components isa AbstractArray || return components  # return scalar values
    return Tensor(components, bases)               # add bases to non-scalars
end

Base.map(f::F, args::AbstractTensor...) where {F} =
    reshape_and_apply_f(Base.Fix1(map, f), args...)

new_basis_for_product(x, y) =
    axes(x, ndims(x)) == axes(y, 1) == AnyBasis() ?
    new_basis_for_product(parent(x), y) :
    overlap_bases(axes(x, ndims(x)), dual(axes(y, 1)))

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

Base.:*(::AbstractTensor{1}, ::Metric) =
    throw(ArgumentError("Adjoint is needed to multiply a vector by a metric"))
Base.:*(x::AbstractTensor{2}, g::Metric) = TensorWithAnyBasis{2}(x, g)
Base.:*(g::Metric, x::AbstractTensor) = TensorWithAnyBasis{1}(x, g)

# Evaluate products from right to left, and always keep the leftmost metric
Base.:*((; x, g)::TensorWithAnyBasis{1}, g_right::Metric) = g * (x * g_right)
Base.:*((; x, g)::TensorWithAnyBasis{2}, ::Metric) = x * g
Base.:*(g::Metric, (; x)::TensorWithAnyBasis{1}) = g * x

Base.adjoint(x::Covector) = Tensor(parent(x)', (axes(x, 2),))
Base.adjoint(x::Tensor{1}) = Tensor(parent(x)', (ScalarBasis(), axes(x, 1)))
Base.adjoint(x::Tensor{2}) = Tensor(parent(x)', reverse(axes(x)))
Base.adjoint((; x, g)::TensorWithAnyBasis{1}) = x' * g
Base.adjoint((; x, g)::TensorWithAnyBasis{2}) = g * x'

Base.inv(x::Tensor{2}) =
    Tensor(inv(parent(x)), unrolled_map(dual, reverse(axes(x))))
Base.inv((; x, g)::TensorWithAnyBasis{1}) = inv(x) * g
Base.inv((; x, g)::TensorWithAnyBasis{2}) = g * inv(x)

# When x^p has ambiguous bases, x^0 gives the right identity, so x * x^0 === x.
function Base.:^(x::AbstractTensor{2}, p::Integer)
    p == 1 && return x
    p < 0 && return inv(x)^(-p)
    (p == 0 && !unrolled_in(AnyBasis(), axes(x))) &&
        return Tensor(one(parent(x)), (dual(axes(x, 2)), axes(x, 2)))
    bases = (dual(new_basis_for_product(x, x)), new_basis_for_product(x, x))
    return add_any_bases(apply_f(Base.Fix2(^, p), reshape(x, bases)), x)
end

norm(x::AbstractTensor, p::Real = 2) = norm(parent(reshape_for_norm(x)), p)
norm_sqr(x::AbstractTensor) = norm_sqr(parent(reshape_for_norm(x)))

#########################################
## Math Operations on Multiple Tensors ##
#########################################

Base.:+(args::AbstractTensor...) = reshape_and_apply_f(+, args...)
Base.:-(x::AbstractTensor, y::AbstractTensor) = reshape_and_apply_f(-, x, y)
Base.:(==)(x::AbstractTensor, y::AbstractTensor) = reshape_and_apply_f(==, x, y)
Base.isapprox(x::AbstractTensor, y::AbstractTensor; kwargs...) =
    reshape_and_apply_f((x, y) -> isapprox(x, y; kwargs...), x, y)

Base.:*(::AbstractTensor{1}, ::AbstractTensor) =
    throw(ArgumentError("Adjoint is needed to multiply a vector by a tensor"))
function Base.:*(x::AbstractTensor{2}, y::AbstractTensor)
    (x_bases, y_bases) = x_and_y_bases_for_product(x, y)
    unrolled_in(AnyBasis(), x_bases) &&
        return add_any_bases(drop_any_bases(x, x_bases) * y, x, x_bases)
    unrolled_in(AnyBasis(), y_bases) &&
        return add_any_bases(x * drop_any_bases(y, y_bases), y, y_bases)
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
Base.:*(y::Tensor{1}, (; x, g)::CovectorWithAnyBasis) = (y * x) * g
Base.:*((; x, g)::VectorWithAnyBasis, y::AbstractCovector) = g * (x * y)

Base.:/(x::AbstractTensor, y::AbstractTensor{2}) = x * inv(y)
Base.:\(y::AbstractTensor{2}, x::AbstractTensor) = inv(y) * x
