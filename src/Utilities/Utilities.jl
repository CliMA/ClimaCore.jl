module Utilities

using UnrolledUtilities

import ForwardDiff
import InteractiveUtils

# Shared walker for the two macros below: apply set! to every listed function's
# methods that belong to this package (mutations of foreign methods would not
# survive precompilation), including matching Core.kwcall methods and their
# hidden body-function methods, where keyword arguments put the loop bodies.
function set_inference_flag!(set!::S, root::Module, fs) where {S}
    for f in fs
        if f isa Module
            defined_names = filter(Base.Fix1(isdefined, f), names(f; all = true))
            values = map(Base.Fix1(getproperty, f), defined_names)
            functions = filter(Base.Fix2(isa, Function), values)
            set_inference_flag!(set!, root, (functions..., Core.kwcall))
            continue
        end
        f === Core.kwcall && continue # handled with the body functions below
        for method in methods(f)
            Base.moduleroot(method.module) === root || continue
            set!(method)
        end
    end
    for method in methods(Core.kwcall)
        Base.moduleroot(method.module) === root || continue
        F = Base.unwrap_unionall(method.sig).parameters[3]
        Core.kwcall in fs || any(g -> g isa Function && F <: typeof(g), fs) ||
            continue
        set!(method)
        body_function = Base.bodyfunction(method)
        isnothing(body_function) && continue
        foreach(set!, methods(body_function))
    end
    return nothing
end

"""
    @drop_recursion_limits f₁, f₂, ...

Remove the inference recursion limit from every listed function's methods that
this package owns, along with the package's `Core.kwcall` methods that target a
listed function (or all of them, if `Core.kwcall` is listed) and their hidden
body functions, where keyword arguments put the actual loop bodies. Listing a
module applies to every function it defines. Use this for functions that
recurse by design: the default limit widens their argument types, turning
downstream calls into dynamic dispatch with runtime allocations, including
inside GPU kernels. Only owned methods are touched, since a mutated foreign
method would not survive precompilation.
"""
macro drop_recursion_limits(fs...)
    f_exprs = length(fs) == 1 && Meta.isexpr(fs[1], :tuple) ? fs[1].args : fs
    return :(@static if hasfield(Method, :recursion_relation)
        set_inference_flag!(
            m -> m.recursion_relation = Returns(true),
            Base.moduleroot(@__MODULE__),
            ($(map(esc, f_exprs)...),),
        )
    end)
end

"""
    @drop_constprop f₁, f₂, ...

Like [`@drop_recursion_limits`](@ref), but disables constant propagation on the
methods, for functions whose bodies constprop re-infers once per distinct
constant argument (thread counts, index-collection lengths) without ever
reaching a different compiled result.
"""
macro drop_constprop(fs...)
    f_exprs = length(fs) == 1 && Meta.isexpr(fs[1], :tuple) ? fs[1].args : fs
    return :(set_inference_flag!(
        m -> m.constprop = 0x02,
        Base.moduleroot(@__MODULE__),
        ($(map(esc, f_exprs)...),),
    ))
end

"""
    ConvertTo{T}()

A GPU-compatible callable that converts its argument to type `T`, equivalent to
`Base.Fix1(convert, T)` but `isbitstype`. `Base.Fix1` stores a `Type{T}` field,
which is not `isbits`, so it cannot be captured by GPU kernels. `ConvertTo{T}`
is an empty struct and is always `isbits`, making it safe to use in broadcast
expressions that run on the GPU.

# Examples

```julia
julia> isbitstype(typeof(ConvertTo{Float32}()))
true

julia> isbitstype(typeof(Base.Fix1(convert, Float32))) # cannot enter a kernel
false
```
"""
struct ConvertTo{T} end
@inline (::ConvertTo{T})(x) where {T} = convert(T, x)


include("plushalf.jl")
include("static_select.jl")
include("auto_broadcaster.jl")
include("cache.jl")
include("safe_mapreduce.jl")

module Unrolled # TODO: Move all of these functions into UnrolledUtilities.jl

import UnrolledUtilities
import ..Utilities: @drop_recursion_limits
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

# Tuple-only fast path for UnrolledUtilities.unrolled_map,
# whose generic method drags a chain of helpers through inference per distinct
# broadcast or layout type; non-Tuples forward to UnrolledUtilities. This is
# the only shim that earns its keep: removing it costs ~10% of a spectral
# expression's compilation memory (extruded-sphere hyperdiffusion:
# 735/753/1671 MB vs 671/690/1503 MB), while shims for the other unrolled_*
# functions are each worth well under a percent. Not map(f, x) or
# ntuple(n -> f(x[n]), Val(N)): neither survives recursion-lifting or inference
# past 32 elements, which rebreaks deeply nested AutoBroadcaster cases and
# reintroduces GPU kernel allocations (gpu_gc_pool_alloc).
@generated unrolled_tuple_map(f, x::NTuple{N, Any}) where {N} = quote
    Base.@_inline_meta
    return Base.Cartesian.@ntuple $N n -> f(x[n])
end
@generated unrolled_tuple_map(f, x::NTuple{N, Any}, y::NTuple{N, Any}) where {N} = quote
    Base.@_inline_meta
    return Base.Cartesian.@ntuple $N n -> f(x[n], y[n])
end
@inline unrolled_tuple_map(f::F, itrs...) where {F} =
    UnrolledUtilities.unrolled_map(f, itrs...)

# NOTE: unrolled_flatten and unrolled_flatmap are deliberately not defined
# here: expanding them at the call site makes layout_args and
# get_non_point_arg_tuple inline far enough into foreach_slice that constprop
# gives up inside Base.Threads._spawn_set_thrpool, reintroducing a runtime
# dispatch that test/Fields/unit_fusion.jl asserts against with @test_opt.

@drop_recursion_limits unrolled_setindex, unrolled_insert,
unrolled_map_with_inbounds, unrolled_tuple_map

end # module Unrolled

import .Unrolled: unrolled_tuple_map

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
    if indices isa Tuple{Union{Integer, AbstractRange{<:Integer}, Colon}} &&
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

```jldoctest; setup = :(import ClimaCore.Utilities: new), filter = r"-?\\d+"
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

"""
    recursive_bottom_eltype(x)

The scalar type underlying `x`, found by following `eltype` until it stops
changing. For a nested array of arrays this is the type of the numbers at the
bottom, not the type of the outer element.

```julia
julia> recursive_bottom_eltype([[1.0, 2.0]])
Float64
```
"""
recursive_bottom_eltype(a) =
    a == eltype(a) ? a : recursive_bottom_eltype(eltype(a))

end # module
