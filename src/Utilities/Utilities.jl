module Utilities

using UnrolledUtilities

import ForwardDiff
import InteractiveUtils

include("plushalf.jl")
include("auto_broadcaster.jl")
include("cache.jl")
include("safe_mapreduce.jl")

module Unrolled # TODO: Move all of these functions into UnrolledUtilities.jl

# Alternative to Base.setindex with guaranteed constant propagation
@inline unrolled_setindex(x::Tuple, value, ::Val{i}) where {i} =
    ntuple(n -> n == i ? value : x[n], Val(length(x)))

# Analogue of insert! that follows the same pattern as unrolled_setindex
@inline unrolled_insert(x::Tuple, value, ::Val{i}) where {i} =
    ntuple(n -> n == i ? value : x[n < i ? n : n - 1], Val(length(x) + 1))

# Same as UnrolledUtilities.unrolled_map, but annotated with @propagate_inbounds
@generated unrolled_map_with_inbounds(f, x::NTuple{N, Any}) where {N} = quote
    Base.@_propagate_inbounds_meta
    return Base.Cartesian.@ntuple $N n -> f(x[n])
end

# Remove each function's recursion limit for better type inference on Julia 1.10
if hasfield(Method, :recursion_relation)
    for f in (unrolled_setindex, unrolled_insert, unrolled_map_with_inbounds)
        for m in methods(f)
            m.recursion_relation = Returns(true)
        end
    end
end

end

"""
    cart_ind(n::NTuple, i::Integer)

Returns a `CartesianIndex` from the list
`CartesianIndices(map(x->Base.OneTo(x), n))[i]`
given size `n` and location `i`.
"""
Base.@propagate_inbounds cart_ind(n::NTuple, i::Integer) =
    @inbounds CartesianIndices(map(x -> Base.OneTo(x), n))[i]

"""
    linear_ind(n::NTuple, ci::CartesianIndex)
    linear_ind(n::NTuple, t::NTuple)

Returns a linear index from the list
`LinearIndices(map(x->Base.OneTo(x), n))[ci]`
given size `n` and cartesian index `ci`.

The `linear_ind(n::NTuple, t::NTuple)` wraps `t`
in a `Cartesian` index and calls
`linear_ind(n::NTuple, ci::CartesianIndex)`.
"""
Base.@propagate_inbounds linear_ind(n::NTuple, ci::CartesianIndex) =
    @inbounds LinearIndices(map(x -> Base.OneTo(x), n))[ci]
Base.@propagate_inbounds linear_ind(n::NTuple, loc::NTuple) =
    linear_ind(n, CartesianIndex(loc))

"""
    stable_view(array, indices...)

Like `view`, but with two modifications that avoid expensive operations:
- Every view is a `SubArray`, even when `array` is a GPU array. GPUArrays
  replaces each contiguous view of a `CuArray` with a new `CuArray` derived
  from the same memory buffer, and the derived array's type is not inferrable,
  which makes all host code that builds slice or property views type-unstable.
  The `SubArray`s constructed here have fully inferred types, and they are
  converted to `SubArray`s of `CuDeviceArray`s when passed to kernels.
- A view along the linear indices of a multidimensional `array` (a single
  `Integer` or range of `Integer`s) wraps the `array` in a 1-dimensional
  `ReshapedArray`, instead of using `reshape` like Base's `view` does, which
  allocates a new object whenever it is applied to an `Array`. If the `array`
  is already a `ReshapedArray`, its parent gets wrapped instead, since a
  reshape stores the same values in the same linear order as its parent.

```julia-repl
julia> array = rand(3, 1, 4);

julia> parent(view(array, 4:6))
12-element Vector{Float64}

julia> parent(stable_view(array, 4:6))
12-element reshape(::Array{Float64, 3}, 12) with eltype Float64
```
"""
Base.@propagate_inbounds function stable_view(array::AbstractArray, indices...)
    if indices isa Tuple{Union{Integer, AbstractRange{<:Integer}}} &&
       ndims(array) != 1
        array isa Base.ReshapedArray &&
            return stable_view(parent(array), first(indices))
        flat_array = Base.ReshapedArray(array, (length(array),), ())
        return stable_view(flat_array, first(indices))
    end
    converted = Base.to_indices(array, indices)
    @boundscheck checkbounds(array, converted...)
    reshaped = Base._maybe_reshape_parent(array, Base.index_ndims(converted...))
    return Base.unsafe_view(reshaped, converted...)
end

"""
    unionall_type(T)

Drops all parameters from the type `T`. If the input argument is not a `Type`,
its type is used instead.

# Examples
```julia
julia> unionall_type(typeof([1, 2, 3]))
Array

julia> unionall_type((; a = 1, b = 2))
NamedTuple
```
"""
unionall_type(::Type{T}) where {T} = Base.typename(T).wrapper
unionall_type(x) = unionall_type(typeof(x))

"""
    replace_type_parameter(T, P, P′)

Recursively modifies the parameters of `T`, replacing every subtype of `P` with
`P′`. This is like constructing a value of type `T` and converting subfields of
type `P` to type `P′`, though no constructors are actually called or compiled.
"""
replace_type_parameter(T, P, P′) = replace_type_parameter(T, Val(Tuple{P, P′}))

# Wrap the two constant types in a Val to guarantee recursive inlining
replace_type_parameter(not_a_type, _) = not_a_type
replace_type_parameter(::Type{<:P}, val::Val{Tuple{P, P′}}) where {P, P′} = P′
replace_type_parameter(::Type{T}, val::Val{Tuple{P, P′}}) where {T, P, P′} =
    isempty(T.parameters) ? T :
    unionall_type(T){
        unrolled_map(Base.Fix2(replace_type_parameter, val), Tuple(T.parameters))...,
    }

"""
    fieldtype_vals(T)

Statically inferrable analogue of `Val.(fieldtypes(T))`. Functions of `Type`s
are specialized upon successful constant propagation, but functions of `Val`s
are always specialized, so `fieldtype_vals` can be used in place of `fieldtypes`
to ensure that recursive functions over nested types have inferrable outputs.
"""
@inline fieldtype_vals(::Type{T}) where {T} =
    ntuple(Val ∘ Base.Fix1(fieldtype, T), Val(fieldcount(T)))

# :new may be called with uninitialized fields as of JuliaLang/julia#52169, but
# this leads to segfaults or other compiler errors for immutable DataType fields
@inline can_alloc_uninitialized(::Tuple{Bool, Val{T}}) where {T <: Type} =
    throw(ArgumentError("Cannot allocate unspecified $T"))
@inline can_alloc_uninitialized((mutable, _)::Tuple{Bool, Val{Type{T}}}) where {T} =
    mutable
@inline can_alloc_uninitialized((mutable, _)::Tuple{Bool, Val{T}}) where {T} =
    if T isa Union{Union, UnionAll}
        throw(ArgumentError("Cannot allocate value of ambiguous type $T"))
    else
        mutable_flags = ntuple(Base.Fix1(!isconst, T), Val(fieldcount(T)))
        flags_and_type_vals = zip(mutable_flags, fieldtype_vals(T))
        mutable || unrolled_all(can_alloc_uninitialized, flags_and_type_vals)
    end

"""
    new(T, [fields])

Exposes the `new` pseudo-function that allocates a value of type `T`, which can
otherwise only be explicitly called from inner constructors.

If provided, the second argument is used to initialize fields of the new value
(unlike the lowered pseudo-function, this will not automatically convert to the
`fieldtypes` of `T`). Otherwise, the fields are initialized with arbitrary data,
with special handling of `DataType` fields to avoid errors during compilation.

# Examples
```jldoctest; setup = :(import ClimaCore.Utilities: new), filter = r"\\d+"
julia> new(Int)
4889520192

julia> new(Complex{Int}, (1, 2))
1 + 2im

julia> new(@NamedTuple{a::Type{Int}, b::Int, c::Complex{Int}})
(a = Int64, b = 4889520192, c = 6162822528 + 8036417625im)

julia> new(@NamedTuple{a::DataType, b::Int, c::Complex{Int}}, (Int, 1, 1 + 2im))
(a = Int64, b = 1, c = 1 + 2im)
```
"""
@inline new(::Type{T}) where {T} = maybe_nested_new(Val(T))
@eval @inline new(::Type{T}, fields) where {T} = $(Expr(:splatnew, :T, :fields))

# Wrap each type in a Val to guarantee recursive inlining
@inline maybe_nested_new(::Val{Type{T}}) where {T} = T
@eval @inline maybe_nested_new(val::Val{T}) where {T} =
    can_alloc_uninitialized((false, val)) ? $(Expr(:new, :T)) : nested_new(val)

# A Tuple{Type{T}, ...} turns into a Tuple{DataType, ...} when it is allocated;
# a @NamedTuple{_::Type{T}, ...} also turns into a @NamedTuple{_::DataType, ...}
@inline nested_new(::Val{T}) where {T} =
    new(T, unrolled_map(maybe_nested_new, fieldtype_vals(T)))
@inline nested_new(::Val{T}) where {T <: Tuple} =
    unrolled_map(maybe_nested_new, fieldtype_vals(T))
@inline nested_new(::Val{T}) where {names, T <: NamedTuple{names}} =
    NamedTuple{names}(unrolled_map(maybe_nested_new, fieldtype_vals(T)))

struct InferenceError <: Exception
    f::Any
    args_type::Type{<:Tuple}
end
function Base.showerror(io::IO, (; f, args_type)::InferenceError)
    println(io, "Concrete type of result could not be inferred:\n")
    InteractiveUtils.code_warntype(io, f, args_type)
end

"""
    is_inferred_type(T)

Checks if `T` either satisfies `isconcretetype` or is a `Type{..}` value (or the
more generic `DataType` value).
"""
@inline is_inferred_type(::Type{T}) where {T} =
    T != Union{} && (isconcretetype(T) || T <: Type)

"""
    return_type(f, T)

Equivalent to `Core.Compiler.return_type(f, T)`, but with an additional check to
ensure that the result satisfies [`is_inferred_type`](@ref) whenever `T` does.
Used in place of `Core.Compiler.return_type` to flag deteriorations in type
inference before they can lead to behavioral changes.
"""
@inline return_type(f::F, ::Type{T}) where {F, T} =
    is_inferred_type(T) && !is_inferred_type(Core.Compiler.return_type(f, T)) ?
    throw(InferenceError(f, T)) : Core.Compiler.return_type(f, T)

"""
    unsafe_eltype(itr)

Analogue of `eltype` with support for un-materialized broadcast expressions,
adapted from `Base.Broadcast.combine_eltypes`. Does not perform any safety
checks, and may potentially return non-concrete types (like an empty `Union{}`).
"""
@inline unsafe_eltype(itr) = eltype(itr)
@inline unsafe_eltype((; f, args)::Base.Broadcast.Broadcasted) =
    unrolled_any(has_inferred_error, args) ? Union{} :
    Core.Compiler.return_type(f, Tuple{unrolled_map(unsafe_eltype, args)...})

@inline has_inferred_error(itr) = unsafe_eltype(itr) == Union{}

"""
    safe_eltype(itr)

Analogue of `eltype` with support for un-materialized broadcast expressions,
adapted from `Base.Broadcast.combine_eltypes`. Throws an error when the result
does not satisfy [`is_inferred_type`](@ref), indicating which part of the
expression first encounters a type instability or an error during inference.
"""
@inline safe_eltype(itr) =
    is_inferred_type(unsafe_eltype(itr)) ? unsafe_eltype(itr) : eltype_error(itr)

eltype_error(itr) = throw(InferenceError(eltype, Tuple{typeof(itr)}))
eltype_error(bc::Base.Broadcast.Broadcasted) =
    has_inferred_error(bc) ?
    bc.f(unrolled_map(new ∘ safe_eltype, bc.args)...) : # f throws runtime error
    throw(InferenceError(bc.f, Tuple{unrolled_map(safe_eltype, bc.args)...}))

end # module
