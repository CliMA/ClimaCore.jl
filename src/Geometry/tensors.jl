####################################################
## Tensor Components in Generalized Coordinates   ##
####################################################

abstract type ComponentsType end

# A `Components{T <: ComponentsType, names}` labels a tensor axis by the
# variance of its components. Variance refers to the way components scale under
# a change of generalized coordinates ξⁱ, whose two reciprocal bases they
# multiply:
#  - the tangent basis `eᵢ = ∂r/∂ξⁱ`, tangent to the coordinate lines: spans V,
#    transforms linearly with the coordinates.
#  - the dual basis `eⁱ = ∇ξⁱ`, with `eⁱ · eⱼ = δⁱⱼ`: spans V*, transforms
#    inversely.

# A tensor is named after the `ComponentsType` of its first axis:
#  - A `CovariantTensor`'s first axis is covariant, and its components `vᵢ`
#    multiply the dual basis; for a vector, `v = vᵢ eⁱ`, v ∈ V*. 
#  - A `ContravariantTensor`'s first axis is contravariant, and its components
#    `vⁱ` multiply the tangent basis; for a vector, `v = vⁱ eᵢ`, v ∈ V.

# Precisely (Stone & Goldbart): given two bases `eⱼ` and `e′ᵢ` over the same
# space, expand one in the other - `eⱼ = aⁱⱼ e′ᵢ` (10.1), for some matrix `a`. A
# vector `v` is invariant (only its components change with the basis), so the
# new components `v′ⁱ` follow from the old `vʲ`: 
#       `v = v′ⁱ e′ᵢ = vʲ eⱼ = vʲ aⁱⱼ e′ᵢ` 
# and comparing coefficients of `e′ᵢ` gives `v′ⁱ = aⁱⱼ vʲ` (10.4). The same
# matrix `a` appears in both laws but in opposite directions, since 10.1 writes
# the old basis via the new, 10.4 writes the new components via the old. The
# components vary inversely or contravariantly to the basis, cancelling its
# change to keep `v = vⁱ eᵢ` fixed. Covariant components are projections onto
# the basis, `vᵢ = v · eᵢ`, so substituting `eᵢ = aʲᵢ e′ⱼ` gives `vᵢ = aʲᵢ (v ·
# e′ⱼ) = aʲᵢ v′ⱼ` (10.6) - the same form as 10.1, so they vary with the basis.

# In the code, these two laws are the two `ComponentsType`s: a
# `Components{Contravariant}` axis stores the `vⁱ` (10.4), a
# `Components{Covariant}` axis stores the `vᵢ` (10.6). A `Tensor`'s first axis
# is one such `Components`: a `ContravariantVector` is a rank-1 `Tensor` over
# `Components{Contravariant}` is a ``, a `CovariantVector` is one over
# `Components{Covariant}`.

# Both forms describe the same physical vector at each point of the domain
# (usually ℝ³) - identical length and direction, differing only in their
# numerical components and the basis those components are multiplied by.

# Changing an axis's component type (`Components{Covariant}` ⟺
# `Components{Contravariant}`) raises or lowers an index with the metric `gᵢⱼ =
# eᵢ · eⱼ` - `vᵢ = gᵢⱼ vʲ` to lower, `vⁱ = gⁱʲ vⱼ` to raise (`gⁱʲ` the inverse)
# - which comes from a `LocalGeometry`; see `project` / `transform` in
# conversions.jl. Reordering, dropping, or zero-filling the component `names`
# within a single component type needs no metric and is what `reshape` does.
#
# Source: M. Stone and P. Goldbart, Mathematics for Physics (Cambridge
# University Press, Cambridge 2004),
# https://cns.gatech.edu/~predrag/courses/PHYS-6124-12/StGoChap10.pdf
struct Covariant <: ComponentsType end     # covariant components vᵢ = v · eᵢ
struct Contravariant <: ComponentsType end # contravariant components vⁱ = v · eⁱ

struct Orthonormal <: ComponentsType end   # components in an orthonormal basis (co = contra)
struct OneScalar <: ComponentsType end     # the trivial scalar axis (a covector's row)

dual_components_type(::Covariant) = Contravariant()
dual_components_type(::Contravariant) = Covariant()
dual_components_type(::Orthonormal) = Orthonormal()
dual_components_type(::OneScalar) = OneScalar()

abstract type AbstractComponents end

"""
    Components{T <: ComponentsType, names}()
    Components(components_type::ComponentsType, names::Tuple)

Type-level description of a single tensor axis. The parameter `T` selects the
component convention ([`Covariant`](@ref), [`Contravariant`](@ref),
[`Orthonormal`](@ref), or [`OneScalar`](@ref)), and `names` is a tuple of
identifiers for the components along the axis (typically dimension indices
like `(1, 3)` for the ξ¹/ξ³ directions, or `(nothing,)` for the scalar row
of a covector; see [`ScalarComponents`](@ref)).

A `Components` is a singleton: all information lives in the type parameters, so
instances are free at runtime and available for multiple dispatch. Named aliases
such as `Covariant13Axis`, `UWAxis`, and `ScalarComponents` are defined for every
supported dimension combination.

# Role in `reshape`

`Components` objects (as opposed to bare `ComponentsType`s) carry the component
names, so they are what lets `reshape` reorder, drop, or zero-fill components
along an axis. Passing `Axes` to `reshape` cannot change the underlying
`ComponentsType` of a concrete `Tensor`; for that, use the component-conversion
helpers in `conversions.jl` (e.g. `project`, `transform`), which apply the
appropriate metric from a `LocalGeometry`.

# Examples
```julia
julia> Covariant13Axis()
Covariant13Components()

julia> length(Covariant13Axis())
2

julia> dual(Covariant13Axis())
Contravariant13Components()

# reshape(tensor, (Components, ...)) reorders and zero-fills by `names`:
julia> v = Covariant12Vector(1.0, 2.0);

julia> reshape(v, (Covariant123Axis(),))         # zero-fill the missing u₃
Tensor([1.0, 2.0, 0.0], (Covariant123Components(),))

julia> reshape(v, (Covariant2Axis(),))           # drop u₁, keep u₂
Tensor([2.0], (Covariant2Components(),))
```
"""
struct Components{T <: ComponentsType, names} <: AbstractComponents end
const ScalarComponents = Components{OneScalar, (nothing,)} # Used in row axes of covectors

Components(::T, names) where {T} =
    unrolled_allunique(names) ? Components{T, names}() :
    throw(ArgumentError("Component names are not all unique: $names"))
Components(::OneScalar, names) =
    names == (nothing,) ? ScalarComponents() :
    throw(ArgumentError("OneScalar basis must contain a single unnamed scalar"))

components_type(::Components{T}) where {T} = T()
component_names(::Components{<:Any, names}) where {names} = names

dual(b::Components) = Components(dual_components_type(components_type(b)), component_names(b))

Base.length(b::Components) = length(component_names(b))

# Extend internal Base.unitrange to support the default show(io, mime, ::Tensor)
Base.unitrange(b::Components) = Base.OneTo(length(b))

Base.show(io::IO, b::Components) =
    print(io, typeof(components_type(b)), join(component_names(b)), "Components()")
Base.show(io::IO, ::ScalarComponents) = print(io, "ScalarComponents()")

no_metric_error(T1, T2) =
    throw(DimensionMismatch("Metric is needed for change of basis: $T1 vs $T2"))
scalar_error(T) =
    throw(DimensionMismatch("Incompatible bases: one scalar vs $T vectors"))

check_same_type(::T1, ::T2) where {T1, T2} = T1 == T2 || no_metric_error(T1, T2)
check_same_type(::OneScalar, ::T) where {T} = scalar_error(T)
check_same_type(::T, ::OneScalar) where {T} = scalar_error(T)
check_same_type(::OneScalar, ::OneScalar) = true

combine_components(b::B, ::B) where {B <: Components} = b
combine_components(b1::Components, b2::Components) =
    check_same_type(components_type(b1), components_type(b2)) &&
    Components(
        components_type(b1),
        unrolled_unique((component_names(b1)..., component_names(b2)...)),
    )
combine_components(b::Components, bases::Components...) =
    unrolled_reduce(combine_components, bases; init = b)
function overlap_components(b1::Components, b2::Components)
    check_same_type(components_type(b1), components_type(b2))
    names2 = component_names(b2)
    overlap = unrolled_filter(n -> unrolled_in(n, names2), component_names(b1))
    return Components(components_type(b1), overlap)
end

# Indices of vectors in src_axis matching the vectors in dest_axis, with
# `nothing` denoting vectors present in dest_axis but missing from src_axis
function matching_component_indices(dest_axis, src_axis)
    check_same_type(components_type(dest_axis), components_type(src_axis))
    return unrolled_map(
        Base.Fix2(unrolled_findfirst, component_names(src_axis)) ∘ ==,
        component_names(dest_axis),
    )
end

########################################
## Tensors in Generalized Coordinates ##
########################################

const AbstractAxes{N} = NTuple{N, AbstractComponents}
const Axes{N} = NTuple{N, Components}
const ComponentsTypes{N} = NTuple{N, ComponentsType}

# Generic tensor whose components can be expressed in different bases; stores
# its bases as a type parameter to facilitate basis-dependent multiple dispatch
abstract type AbstractTensor{N, T, B <: AbstractAxes{N}} <: AbstractArray{T, N} end

"""
    Tensor(components, bases::NTuple{N, Components})
    Tensor(s::UniformScaling, bases::NTuple{2, Components})

`N`-dimensional tensor whose entries in `components` are interpreted with
respect to the given `bases`. Each entry of `bases` is a [`Components`](@ref), and
the shape of `components` must match `length.(bases)`. The bases are
stored as a type parameter so that operations can dispatch on the kind of
basis (covariant, contravariant, orthonormal, or scalar) at compile time.

Use `parent(x)` to get the component array and `axes(x)` to get the bases.
Scalar indexing `x[i, j, ...]` reads directly from the component array;
colon-indexing yields a smaller `Tensor` over the remaining non-colon axes.

# Shapes that `Tensor` takes in practice

- `Tensor{1}` (a vector): `components::SVector`, `bases::Tuple{Components}`.
- `Tensor{2}` covector (a row-vector): the first axis is [`ScalarComponents`](@ref)
  and `components::Adjoint{T, SVector}`. This is what `v'` produces for a
  `Tensor{1}` `v`.
- `Tensor{2}` square tensor: `components::SMatrix`.

The `UniformScaling` constructor is a convenience that converts
`s = λ * I` (where `λ = s.λ` is the scalar stored in Julia's
`LinearAlgebra.UniformScaling`) into an `SMatrix` of the appropriate size -
e.g., `Tensor(2I, (b, b))` builds a diagonal tensor with `2` on the diagonal.

# Role in `reshape`

`reshape(x::Tensor, bases::NTuple{N, Components})` changes basis-vector
*names* along each axis, zero-filling gaps. Changing the `ComponentsType` of a
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
Tensor([1.0, 2.0], (Covariant12Components(),))

julia> parent(v)
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 2.0

julia> axes(v)
(Covariant12Components(),)

julia> v[1], v.u₁                                # indexed and named access
(1.0, 1.0)

julia> reshape(v, (Covariant123Axis(),))         # names-only reshape
Tensor([1.0, 2.0, 0.0], (Covariant123Components(),))
```
"""
struct Tensor{N, T, B <: Axes{N}, C} <: AbstractTensor{N, T, B}
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
function Tensor(s::UniformScaling, bases::NTuple{2, Components})
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

@inline tensor_axes(::Type{T}) where {T <: Tensor} =
    _unwrap(T).parameters[3].instance

Base.zero(x::Tensor) = zero(typeof(x))
Base.one(x::Tensor) = one(typeof(x))

Base.zero(::Type{Tensor{N, T, B, C}}) where {N, T, B, C} =
    Tensor(zero(C), B.instance)
# The Covector representation (Tensor{2} with ScalarComponents row) stores its
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

function axes_at_colons(bases, indices)
    axis_index_pairs = unrolled_map(tuple, bases, indices)
    axis_colon_pairs = unrolled_filter(==(Colon()) ∘ last, axis_index_pairs)
    return unrolled_map(first, axis_colon_pairs)
end

Base.@propagate_inbounds Base.getindex(x::Tensor, indices::TensorIndex...) =
    Tensor(parent(x)[indices...], axes_at_colons(axes(x), indices))
Base.@propagate_inbounds Base.view(x::Tensor, indices::TensorIndex...) =
    Tensor(view(parent(x), indices...), axes_at_colons(axes(x), indices))

#############################################
## Metric Terms in Generalized Coordinates ##
#############################################

"""
    pad_metric_tensor(∂x∂ξ::Tensor{2})

Pads an N×N metric tensor with axes `(Components{Orthonormal, I}, Components{Covariant, I})`
to a full 3×3 tensor with axes `(UVWAxis, Covariant123Axis)`, putting `1`
on diagonal entries for dimensions outside `I` and `0` on cross-coupling
entries. The padded form encodes the "identity metric in directions
orthogonal to `I`" convention as actual matrix entries, so a single matvec
`padded_M * v` covers all source-name configurations without partition
logic. Idempotent for `I == (1, 2, 3)`.
"""
function pad_metric_tensor(∂x∂ξ::Tensor{2})
    src_names = component_names(axes(∂x∂ξ, 1))
    src_names == (1, 2, 3) && return ∂x∂ξ
    full_axes = (UVWAxis(), Covariant123Axis())
    # `reshape` to the full bases zero-fills rows/cols whose name isn't in
    # `src_names`. We then add `1` on diagonal entries at dims not in
    # `src_names` to recover the identity-padding convention.
    padded_zeros = reshape(∂x∂ξ, full_axes)
    iso = _orthogonal_identity(Val(src_names), eltype(∂x∂ξ))
    return Tensor(parent(padded_zeros) + iso, full_axes)
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
    AbstractTensor{2, <:Any, <:Tuple{ScalarComponents, AbstractComponents}}
const Covector = Tensor{2, <:Any, <:Tuple{ScalarComponents, Components}}

#####################################
## Modifying the Axes of a Tensor ##
#####################################

check_ndims(x, N) =
    ndims(x) == N ||
    throw(DimensionMismatch("Cannot reshape $(ndims(x))-tensor into $N-tensor"))

# Change the component_names, and, if possible, change the ComponentsTypes as well
function Base.reshape(x::Tensor, bases::Axes)
    check_ndims(x, length(bases))
    axes(x) == bases && return x
    components_constructor = SArray{Tuple{unrolled_map(length, bases)...}, eltype(x)}
    component_indices = unrolled_product(
        unrolled_map(matching_component_indices, bases, axes(x))...,
    )
    component_values = unrolled_map(component_indices) do indices
        unrolled_any(isnothing, indices) ? zero(eltype(x)) : x[indices...]
    end
    return Tensor(components_constructor(component_values), bases)
end
Base.reshape(x::AbstractTensor, bases::Components...) = reshape(x, bases)

# Change the ComponentsTypes without constraining the component_names
Base.reshape(x::Tensor, types::ComponentsTypes) =
    check_ndims(x, length(types)) &&
    unrolled_map(components_type, axes(x)) == types ? x :
    throw(DimensionMismatch("Metric is needed for change of basis: \
                             $(unrolled_map(components_type, axes(x))) vs $types"))
Base.reshape(x::AbstractTensor, types::ComponentsType...) = reshape(x, types)

# Change all bases to a single ComponentsType
Base.reshape(x::AbstractTensor, type::ComponentsType) =
    reshape(x, ntuple(Returns(type), Val(ndims(x))))

reshape_for_norm(x::AbstractTensor) = reshape(x, Orthonormal())
reshape_for_norm(x::AbstractCovector) = reshape_for_norm(x')

########################################
## Other Generic Tensor Manipulations ##
########################################

apply_f(f, x::Tensor) = Tensor(f(parent(x)), axes(x))

function reshape_and_apply_f(f::F, args...) where {F}
    unrolled_foreach(Base.Fix2(check_ndims, ndims(args[1])), args)
    bases = unrolled_map(combine_components, unrolled_map(axes, args)...)
    components = f(unrolled_map(x -> parent(reshape(x, bases)), args)...)
    components isa AbstractArray || return components  # return scalar values
    return Tensor(components, bases)               # add bases to non-scalars
end


Base.map(f::F, args::AbstractTensor...) where {F} =
    reshape_and_apply_f((xs...) -> map(f, xs...), args...)

# The contracted-axis basis for `x * y`.
new_components_for_product(x, y) = overlap_components(axes(x, ndims(x)), dual(axes(y, 1)))

x_and_y_axes_for_product(x, y) = (
    Base.setindex(axes(x), new_components_for_product(x, y), ndims(x)),
    Base.setindex(axes(y), dual(new_components_for_product(x, y)), 1),
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
Base.adjoint(x::Tensor{1}) = Tensor(parent(x)', (ScalarComponents(), axes(x, 1)))
Base.adjoint(x::Tensor{2}) = Tensor(parent(x)', (axes(x, 2), axes(x, 1)))

Base.inv(x::Tensor{2}) =
    Tensor(inv(parent(x)), (dual(axes(x, 2)), dual(axes(x, 1))))

function Base.:^(x::AbstractTensor{2}, p::Integer)
    p == 1 && return x
    p < 0 && return inv(x)^(-p)
    p == 0 && return Tensor(one(parent(x)), (dual(axes(x, 2)), axes(x, 2)))
    bases = (dual(new_components_for_product(x, x)), new_components_for_product(x, x))
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
    (x_axes, y_axes) = x_and_y_axes_for_product(x, y)
    components = parent(reshape(x, x_axes)) * parent(reshape(y, y_axes))
    return Tensor(components, Base.setindex(axes(y), axes(x, 1), 1))
end

Base.:*(x::AbstractCovector, y::AbstractTensor{1}) = dot(x', y)
function dot(x::AbstractTensor{1}, y::AbstractTensor{1})
    (x_axes, y_axes) = x_and_y_axes_for_product(x, y)
    return dot(parent(reshape(x, x_axes)), parent(reshape(y, y_axes)))
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
const CovariantVector{T, I, S} = Tensor{1, T, Tuple{Components{Covariant, I}}, S}
const ContravariantVector{T, I, S} = Tensor{1, T, Tuple{Components{Contravariant, I}}, S}
const LocalVector{T, I, S} = Tensor{1, T, Tuple{Components{Orthonormal, I}}, S}

# Union types for dispatch
const CovariantTensor = Union{
    Tensor{1, <:Any, <:Tuple{Components{Covariant}}},
    Tensor{2, <:Any, <:Tuple{Components{Covariant}, <:AbstractComponents}},
}
const ContravariantTensor = Union{
    Tensor{1, <:Any, <:Tuple{Components{Contravariant}}},
    Tensor{2, <:Any, <:Tuple{Components{Contravariant}, <:AbstractComponents}},
}
const OrthonormalTensor = Union{
    Tensor{1, <:Any, <:Tuple{Components{Orthonormal}}},
    Tensor{2, <:Any, <:Tuple{Components{Orthonormal}, <:AbstractComponents}},
}

# Concrete axis and vector aliases for all dimension combinations
for I in [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    strI = isempty(I) ? "Null" : join(I)
    N = length(I)
    strUVW = isempty(I) ? "Null" : join(map(i -> [:U, :V, :W][i], I))

    # Axis aliases (Components singletons)
    @eval const $(Symbol(:Covariant, strI, :Axis)) = Components{Covariant, $I}
    @eval const $(Symbol(:Contravariant, strI, :Axis)) = Components{Contravariant, $I}
    @eval const $(Symbol(strUVW, :Axis)) = Components{Orthonormal, $I}

    # A `CovariantNVector` lives in V* and has covariant components `vᵢ`
    # expressed in the contravariant basis `{eⁱ}`. A `ContravariantNVector`
    # lives in V and has contravariant components `vⁱ` expressed in the
    # covariant basis `{eᵢ}`. See the convention block at the top of this
    # file for the full naming explanation. Splatted constructors like
    # `Covariant12Vector(1.0, 2.0)` are handled by the generic
    # `(::Type{T})(args::Number...)` defined below for `T <: Tensor{1}`,
    # which uses `tensor_axes(T)` to extract the basis tuple even when `T`
    # is a UnionAll with free eltype/storage parameters.
    @eval const $(Symbol(:Covariant, strI, :Vector)){T} =
        CovariantVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Contravariant, strI, :Vector)){T} =
        ContravariantVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(strUVW, :Vector)){T} =
        LocalVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Cartesian, strI, :Vector)){T} =
        LocalVector{T, $I, SVector{$N, T}}
    @eval const $(Symbol(:Cartesian, strI, :Axis)) = Components{Orthonormal, $I}
end

# Named property access for vectors (e.g., v.u₁, v.u², v.u)
_symbols(::Covariant) = (:u₁, :u₂, :u₃)
_symbols(::Contravariant) = (:u¹, :u², :u³)
_symbols(::Orthonormal) = (:u, :v, :w)

Base.propertynames(x::Tensor{1}) = _symbols(components_type(x.bases[1]))

# `unrolled_findfirst` walks the compile-time `I` tuple, so the search
# inlines to a flat chain of `name === :u_k` comparisons that fold to a
# direct `parent(x)[idx]` read for the matching name or `zero(T)`.
@inline function Base.getproperty(
    x::Tensor{1, T, <:Tuple{Components{BT, I}}}, name::Symbol,
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
    Tensor(SVector(args...), tensor_axes(T))

"""
    project(basis, v)
    project(basis, v, local_geometry)

Project the first axis of vector/tensor `v` onto `basis`, zero-filling
components not present in the source. When `local_geometry` is provided,
performs a change of basis type (e.g., Covariant → Contravariant) via the metric.
"""
@inline project(b::Components, v::AbstractTensor{1}) = reshape(v, (b,))
@inline project(b::Components, v::AbstractTensor{2}) = reshape(v, (b, axes(v, 2)))
@inline function project(b::Components, v::AbstractTensor{2}, b2::Components)
    reshape(v, (b, b2))
end
@inline project(v::AbstractTensor{2}, b::Components) = reshape(v, (axes(v, 1), b))

"""
    transform(basis, v)

Transform the first axis of vector or 2-tensor `v` to `basis`. Unlike
`project`, throws an `InexactError` if any dropped component is nonzero.
"""
@inline function transform(b::Components, v::AbstractTensor{1})
    result = reshape(v, (b,))
    # Check that no nonzero components were dropped. `unrolled_all` over a
    # static index tuple lets the compiler elide the check entirely when
    # `src_names ⊆ dest_names` (so every iteration's left disjunct is `true`).
    src_names = component_names(axes(v, 1))
    dest_names = component_names(b)
    pv = parent(v)
    indices = ntuple(identity, Val(length(src_names)))
    unrolled_all(indices) do n
        unrolled_in(src_names[n], dest_names) || iszero(pv[n])
    end || throw(InexactError(:transform, typeof(b), v))
    return result
end
@inline function transform(b::Components, v::AbstractTensor{2})
    result = reshape(v, (b, axes(v, 2)))
    # Same idea as the Tensor{1} case, with an inner unrolled_all over the
    # row entries: a dropped row only passes if every column entry is zero.
    src_names = component_names(axes(v, 1))
    dest_names = component_names(b)
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
    x::Tensor{1, <:Any, <:Tuple{Components{Orthonormal}}},
    y::Tensor{1, <:Any, <:Tuple{Components{Orthonormal}}},
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
