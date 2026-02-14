using UnrolledUtilities
import StaticArrays: SArray
import LinearAlgebra: Adjoint, det, dot, norm, norm_sqr

##############################################
## Basis Vectors in Generalized Coordinates ##
##############################################

abstract type BasisType end

struct Orthonormal <: BasisType end   # Any orthogonal vectors of unit length
struct Covariant <: BasisType end     # Basis vector i is given by eⁱ = ∇ξⁱ
struct Contravariant <: BasisType end # Basis vector i is given by eᵢ = ∂r/∂ξⁱ
# FIXME: Swap Covariant and Contravariant definitions in next breaking release
# (current definition is based on how components transform, not basis vectors)

dual_basis_type(::Orthonormal) = Orthonormal()
dual_basis_type(::Covariant) = Contravariant()
dual_basis_type(::Contravariant) = Covariant()

struct Basis{T <: BasisType, names} end

Basis(::T, names) where {T} =
    unrolled_allunique(names) ? Basis{T, names}() :
    throw(ArgumentError("Basis vector names are not all unique: $names"))

basis_type(::Basis{T}) where {T} = T()
basis_vector_names(::Basis{<:Any, names}) where {names} = names

dual(b) = Basis(dual_basis_type(basis_type(b)), basis_vector_names(b))

Base.length(b::Basis) = length(basis_vector_names(b))

# Extend internal Base.unitrange to support the default show(io, mime, ::Tensor)
Base.unitrange(b::Basis) = Base.OneTo(length(b))

Base.show(io::IO, b::Basis) =
    print(io, typeof(basis_type(b)), join(basis_vector_names(b)), "Basis()")

check_same_basis_type(b1, b2) =
    basis_type(b1) == basis_type(b2) ||
    throw(DimensionMismatch("Metric is required to combine bases of mismatched \
                             types $(basis_type(b1)) and $(basis_type(b2))"))

Base.union(b1::Basis, b2::Basis) =
    check_same_basis_type(b1, b2) && Basis(
        basis_type(b1),
        unrolled_unique((basis_vector_names(b1)..., basis_vector_names(b2)...)),
    )
Base.intersect(b1::Basis, b2::Basis) =
    check_same_basis_type(b1, b2) && Basis(
        basis_type(b1),
        unrolled_filter(in(basis_vector_names(b2)), basis_vector_names(b1)),
    )

# Indices of vectors in src_basis matching the vectors in dest_basis, with
# `nothing` denoting vectors present in dest_basis but missing from src_basis
matching_basis_vector_indices(dest_basis, src_basis) =
    check_same_basis_type(dest_basis, src_basis) &&
    unrolled_map(
        Base.Fix2(unrolled_findfirst, basis_vector_names(src_basis)) ∘ ==,
        basis_vector_names(dest_basis),
    )

########################################
## Tensors in Generalized Coordinates ##
########################################

abstract type AbstractTensor{N, T} <: AbstractArray{T, N} end

const AbstractCovector{T} = Adjoint{T, <:AbstractTensor{1, T}}

Base.zero(x::AbstractTensor) = zero(typeof(x))
Base.one(x::AbstractTensor) = one(typeof(x))

# A tensor represented by its components in a specific set of coordinate bases;
# supports combining different basis_vector_names, but not different basis_types
struct Tensor{N, T, C <: AbstractArray{T, N}, B <: NTuple{N, Basis}} <:
       AbstractTensor{N, T}
    components::C
    bases::B
end

Tensor(components::C, bases::B) where {C, B} =
    size(components) == unrolled_map(length, bases) ?
    Tensor{ndims(C), eltype(C), C, B}(components, bases) :
    throw(DimensionMismatch("Tensor component array size, $(size(components)), \
                             does not match dimensions of tensor coordinate \
                             bases, $(unrolled_map(length, bases))"))

Base.parent(x::Tensor) = x.components
Base.axes(x::Tensor) = x.bases

apply_to_parent(f, x::Tensor) = Tensor(f(parent(x)), axes(x))

Base.zero(::Type{Tensor{N, T, C, B}}) where {N, T, C, B} =
    Tensor(zero(C), B.instance)
Base.one(::Type{Tensor{N, T, C, B}}) where {N, T, C, B} =
    Tensor(one(C), B.instance)
Base.convert(::Type{Tensor{N, T, C, B}}, x::AbstractTensor) where {N, T, C, B} =
    Tensor(convert(C, parent(reshape(x, B.instance))), B.instance)

Base.size(x::Tensor) = unrolled_map(length, axes(x))
Base.getindex(x::Tensor, indices::Integer...) = parent(x)[indices...]
Base.isassigned(x::Tensor, indices::Integer...) =
    isassigned(parent(x), indices...)
Base.setindex!(x::Tensor, component, indices::Integer...) =
    parent(x)[indices...] = component
Base.setindex(x::Tensor, component, indices::Integer...) =
    Tensor(Base.setindex(parent(x), component, indices...), axes(x))

Base.show(io::IO, x::Tensor) =
    print(io, "Tensor(", parent(x), ", ", axes(x), ")")

#############################################
## Metric Terms in Generalized Coordinates ##
#############################################

# A metric represented by a tensor that maps between a specific pair of bases;
# multiplying a tensor by a metric assigns a MetricBasis to one of its axes
struct Metric{T <: Tensor{2}}
    tensor::T
end

Base.zero(g::Metric) = zero(typeof(g))
Base.one(g::Metric) = one(typeof(g))

Base.zero(::Type{Metric{T}}) where {T} = Metric(zero(T))
Base.one(::Type{Metric{T}}) where {T} = Metric(one(T))
Base.convert(::Type{Metric{T}}, g::Metric) where {T} =
    Metric(convert(T, g.tensor))

Base.show(io::IO, g::Metric) = print(io, "Metric(", g.tensor, ")")

# Computes the 2-tensor that converts any src_type basis to any dest_type basis
change_of_basis_tensor(g::Metric, dest_type, src_type) = cob_tensor(
    src_type,
    dest_type,
    basis_type(dual(axes(g.tensor, 2))),
    basis_type(axes(g.tensor, 1)),
    g.tensor,
)

# Orthonormal-to-Covariant     change of basis tensor: (∂r/∂ξ)'
# Orthonormal-to-Contravariant change of basis tensor: (∂ξ/∂r)
# Covariant-to-Orthonormal     change of basis tensor: (∂ξ/∂r)'
# Contravariant-to-Orthonormal change of basis tensor: (∂r/∂ξ)
# Covariant-to-Contravariant   change of basis tensor: (∂ξ/∂r) * (∂ξ/∂r)'
# Contravariant-to-Covariant   change of basis tensor: (∂r/∂ξ)' * (∂r/∂ξ)
for (T1, T2) in ((:Covariant, :Contravariant), (:Contravariant, :Covariant))
    TO = :Orthonormal
    @eval cob_tensor(::$TO, ::$T1, ::$TO, ::$T2, tensor) = inv(tensor')
    @eval cob_tensor(::$T1, ::$TO, ::$T2, ::$TO, tensor) = inv(tensor')
    @eval cob_tensor(::$T1, ::$T2, ::$TO, ::$T2, tensor) = tensor * tensor'
    @eval cob_tensor(::$T1, ::$T2, ::$T1, ::$TO, tensor) = tensor' * tensor
    @eval cob_tensor(::$T1, ::$T2, ::$TO, ::$T1, tensor) = inv(tensor * tensor')
    @eval cob_tensor(::$T1, ::$T2, ::$T2, ::$TO, tensor) = inv(tensor' * tensor)
end
cob_tensor(::T1, ::T1, _, _, _) where {T1} = 1
cob_tensor(::T1, ::T1, ::T1, ::T1, _) where {T1} = 1 # needed to avoid ambiguity
cob_tensor(::T1, ::T2, ::T1, ::T2, tensor) where {T1, T2} = tensor
cob_tensor(::T1, ::T2, ::T2, ::T1, tensor) where {T1, T2} = inv(tensor)
cob_tensor(::T1, ::T2, ::T3, ::T4, tensor) where {T1, T2, T3, T4} =
    throw(DimensionMismatch("Cannot compute $T1-to-$T2 change of basis tensor \
                             from $T3-to-$T4 metric representation $tensor"))

# Computes the metric term J (which determines the volume element δV = W * J),
# avoiding a call to inv when possible by using (∂r/∂ξ)' in place of (∂r/∂ξ)
jacobian_determinant(g::Metric) =
    unrolled_map(basis_type, axes(g.tensor)) == (Covariant(), Orthonormal()) ?
    det(parent(g.tensor)) :
    det(parent(change_of_basis_tensor(g, Orthonormal(), Contravariant())))

#############################################
## Tensors With Arbitrary Coordinate Bases ##
#############################################

struct MetricBasis end # Represents any basis that is compatible with the metric

basis_type(::MetricBasis) =
    throw(ArgumentError("Metric basis type is ambiguous"))
basis_vector_names(::MetricBasis) =
    throw(ArgumentError("Metric basis vector names are ambiguous"))

dual(::MetricBasis) = MetricBasis()

Base.union(b::Basis, ::MetricBasis) = b
Base.union(::MetricBasis, b::Basis) = b
Base.union(::MetricBasis, ::MetricBasis) = MetricBasis()

Base.intersect(b::Basis, ::MetricBasis) = b
Base.intersect(::MetricBasis, b::Basis) = b
Base.intersect(::MetricBasis, ::MetricBasis) = MetricBasis()

# A tensor with a MetricBasis assigned to its axis at dimension index d;
# supports combining both different basis_vector_names and different basis_types
struct TensorWithMetricBasis{d, N, T, X <: AbstractTensor{N, T}, G <: Metric} <:
       AbstractTensor{N, T}
    x::X
    g::G
end

TensorWithMetricBasis{d}(x::X, g::G) where {d, X, G} =
    TensorWithMetricBasis{d, ndims(X), eltype(X), X, G}(x, g)

Base.parent((; x)::TensorWithMetricBasis) = x
Base.axes((; x)::TensorWithMetricBasis{d}) where {d} =
    Base.setindex(axes(x), MetricBasis(), d)

apply_to_parent(f, (; x, g)::TensorWithMetricBasis{d}) where {d} =
    TensorWithMetricBasis{d}(f(x), g)

Base.zero(::Type{TensorWithMetricBasis{d, N, T, X, G}}) where {d, N, T, X, G} =
    TensorWithMetricBasis{d}(zero(X), one(G)) # use one to avoid inverting zero
Base.one(::Type{TensorWithMetricBasis{d, N, T, X, G}}) where {d, N, T, X, G} =
    TensorWithMetricBasis{d}(one(X), one(G))
Base.convert(
    ::Type{TensorWithMetricBasis{d, N, T, X, G}},
    (; x, g)::TensorWithMetricBasis{d},
) where {d, N, T, X, G} =
    TensorWithMetricBasis{d}(convert(X, x), convert(G, g))

# Specialize on mime type to bypass the default show(io, mime, ::AbstractArray)
Base.show(
    io::IO,
    ::MIME"text/plain",
    (; x, g)::TensorWithMetricBasis{d},
) where {d} =
    print(io, "TensorWithMetricBasis{", d, "}(\n    ", x, ",\n    ", g, ",\n)")
Base.show(io::IO, (; x, g)::TensorWithMetricBasis{d}) where {d} =
    print(io, "TensorWithMetricBasis{", d, "}(", x, ", ", g, ")")

##############################################
## Math Operations on Abstract Tensor Types ##
##############################################

Base.:-(x::AbstractTensor) = apply_to_parent(-, x)

Base.:*(n::Number, x::AbstractTensor) = apply_to_parent(Base.Fix1(*, n), x)
Base.:*(x::AbstractTensor, n::Number) = apply_to_parent(Base.Fix2(*, n), x)
Base.:\(n::Number, x::AbstractTensor) = apply_to_parent(Base.Fix1(\, n), x)
Base.:/(x::AbstractTensor, n::Number) = apply_to_parent(Base.Fix2(/, n), x)

Base.:*(g::Metric, x::AbstractTensor) = TensorWithMetricBasis{1}(x, g)
Base.:*(x::AbstractTensor, g::Metric) = TensorWithMetricBasis{2}(x, g)
Base.:*(x::AbstractCovector, g::Metric) = (g * x')'
Base.:*(::Metric, ::Metric) =
    throw(ArgumentError("Metric cannot be multiplied by another Metric"))

Base.:/(x::AbstractTensor, y::AbstractTensor{2}) = x * inv(y)
Base.:\(x::AbstractTensor{2}, y::AbstractTensor) = inv(x) * y

Base.reshape(x::AbstractTensor, bases::Basis...) = reshape(x, bases)

function reshape_for_map(args...)
    bases = unrolled_map(union, unrolled_map(axes, args)...)
    return unrolled_map(Base.Fix2(reshape, bases), args)
end

Base.map(f::F, args::AbstractTensor...) where {F} =
    unrolled_allequal(axes, args) ?
    Tensor(map(f, unrolled_map(parent, args)...), axes(first(args))) :
    map(f, reshape_for_map(args...)...)
Base.:+(args::AbstractTensor...) =
    unrolled_allequal(axes, args) ?
    Tensor(+(unrolled_map(parent, args)...), axes(first(args))) :
    +(reshape_for_map(args...)...)
Base.:-(x::AbstractTensor, y::AbstractTensor) =
    axes(x) == axes(y) ? Tensor(parent(x) - parent(y), axes(x)) :
    -(reshape_for_map(x, y)...)
Base.:(==)(x::AbstractTensor, y::AbstractTensor) =
    axes(x) == axes(y) ? parent(x) == parent(y) : ==(reshape_for_map(x, y)...)

function reshape_for_mul(x, y)
    multiplication_axis = intersect(axes(x, ndims(x)), dual(axes(y, 1)))
    bases_for_x = Base.setindex(axes(x), multiplication_axis, ndims(x))
    bases_for_y = Base.setindex(axes(y), dual(multiplication_axis), 1)
    return (reshape(x, bases_for_x), reshape(y, bases_for_y))
end

Base.:*(x::AbstractTensor{2}, y::AbstractTensor) =
    axes(x, 2) == dual(axes(y, 1)) ?
    Tensor(parent(x) * parent(y), Base.setindex(axes(y), axes(x, 1), 1)) :
    *(reshape_for_mul(x, y)...)
dot(x::AbstractTensor{1}, y::AbstractTensor{1}) =
    axes(x, 1) == dual(axes(y, 1)) ? dot(parent(x), parent(y)) :
    dot(reshape_for_mul(x, y)...)

use_parent_for_norm(x) =
    unrolled_all(!=(MetricBasis()), axes(x)) &&
    unrolled_all(==(Orthonormal()) ∘ basis_type, axes(x))

norm(x::AbstractTensor, p::Real = 2) =
    use_parent_for_norm(x) ? norm(parent(x), p) : norm(reshape_for_norm(x), p)
norm_sqr(x::AbstractTensor) =
    use_parent_for_norm(x) ? norm_sqr(parent(x)) : norm_sqr(reshape_for_norm(x))

##############################################
## Math Operations on Concrete Tensor Types ##
##############################################

parent_tensor(x::Tensor) = x
parent_tensor(x::TensorWithMetricBasis) = parent_tensor(parent(x))

# TODO: Fix remaining commutativity bugs by taking a union over the MetricBasis
# dimension indices from all arguments, instead of just using x
add_metric_bases(arg, ::Tensor) = arg
add_metric_bases(arg, (; x, g)::TensorWithMetricBasis{d}) where {d} =
    TensorWithMetricBasis{d}(add_metric_bases(arg, x), g)

Base.map(f::F, x::TensorWithMetricBasis, args::TensorWithMetricBasis...) where {F} =
    add_metric_bases(map(f, parent_tensor(x), args...), x)
Base.:+(x::TensorWithMetricBasis, args::TensorWithMetricBasis...) =
    add_metric_bases(+(parent_tensor(x), args...), x)
Base.:-(x::TensorWithMetricBasis, y::TensorWithMetricBasis) =
    add_metric_bases(parent_tensor(x) - y, x)
Base.:(==)(x::TensorWithMetricBasis, y::TensorWithMetricBasis) =
    parent_tensor(x) == y

Base.:*((; x, g)::TensorWithMetricBasis{1, 1}, y::AbstractCovector) = g * (x * y)
Base.:*((; x, g)::TensorWithMetricBasis{1, 2}, y::AbstractTensor) = g * (x * y)
Base.:*((; x, g)::TensorWithMetricBasis{1, 2}, g_new::Metric) = g * (x * g_new)
Base.:*((; x, g)::TensorWithMetricBasis{2, 2}, g_new::Metric) = x * (g * g_new)
Base.:*(g_new::Metric, (; x, g)::TensorWithMetricBasis{1, 2}) = (g_new * g) * x

Base.:*(x::Tensor{1}, y::Adjoint{<:Any, <:Tensor{1}}) =
    Tensor(parent(x) * parent(y')', (axes(x, 1), axes(y, 2)))
Base.:*(x::Tensor{2}, y::TensorWithMetricBasis{2, 2}) = (y' * x')'
Base.:*(x::Tensor{1}, y::Adjoint{<:Any, <:TensorWithMetricBasis{1, 1}}) = (y' * x')'

Base.adjoint(x::Tensor{2}) = Tensor(parent(x)', reverse(axes(x)))
Base.adjoint((; x, g)::TensorWithMetricBasis{1, 2}) = x' * g
Base.adjoint((; x, g)::TensorWithMetricBasis{2, 2}) = g * x'

Base.inv(x::Tensor{2}) =
    Tensor(inv(parent(x)), unrolled_map(dual, reverse(axes(x))))
Base.inv((; x, g)::TensorWithMetricBasis{1, 2}) = inv(x) * g
Base.inv((; x, g)::TensorWithMetricBasis{2, 2}) = g * inv(x)

function Base.reshape(x::Tensor, bases::Tuple{Vararg{Basis}})
    ndims(x) == length(bases) ||
        throw(DimensionMismatch("Cannot reshape a $(ndims(x))-tensor into a \
                                 $(length(bases))-tensor"))
    components_constructor = SArray{Tuple{unrolled_map(length, bases)...}}
    component_indices = unrolled_product(
        unrolled_map(matching_basis_vector_indices, bases, axes(x))...,
    )
    component_values = unrolled_map(component_indices) do indices
        unrolled_any(isnothing, indices) ? zero(eltype(x)) : x[indices...]
    end
    return Tensor(components_constructor(component_values), bases)
end

reshape_by_basis_type(g, b1, b2) =
    change_of_basis_tensor(g, basis_type(b1), basis_type(dual(b2)))
Base.reshape((; x, g)::TensorWithMetricBasis{1}, bases::Tuple{Vararg{Basis}}) =
    reshape(reshape_by_basis_type(g, bases[1], dual(axes(x, 1))) * x, bases)
Base.reshape((; x, g)::TensorWithMetricBasis{2}, bases::Tuple{Vararg{Basis}}) =
    reshape(x * reshape_by_basis_type(g, dual(axes(x, 2)), bases[2]), bases)

reshape_for_norm(x::Tensor) =
    throw(ArgumentError("Metric is required to compute norm of tensor with \
                         non-orthonormal bases: $(axes(x))"))

reshape_for_norm((; x, g)::TensorWithMetricBasis{1}) =
    change_of_basis_tensor(g, Orthonormal(), basis_type(axes(x, 1))) * x
reshape_for_norm((; x, g)::TensorWithMetricBasis{2}) =
    x * change_of_basis_tensor(g, basis_type(dual(axes(x, 2))), Orthonormal())
