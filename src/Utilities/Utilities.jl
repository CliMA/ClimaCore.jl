module Utilities

import UnrolledUtilities: unrolled_map

include("plushalf.jl")
include("cache.jl")

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
    unionall_type(::Type{T})

Extract the type of the input, and strip it of any type parameters.

This is useful when one needs the generic constructor for a given type.

Example
=======
```julia
julia> unionall_type(typeof([1, 2, 3]))
Array

julia> struct Foo{A, B}
               a::A
               b::B
       end

julia> unionall_type(typeof(Foo(1,2)))
Foo
```
"""
function unionall_type(::Type{T}) where {T}
    # NOTE: As of version 1.12, there is no simple, user-friendly way to extract
    # the generic type of T, so we need to reach for the internals in Julia.
    # Hopefully, Julia will introduce a simpler, more stable way to do this in a
    # future release.
    return T.name.wrapper
end

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

"""
    new(T, [fields])

Exposes the `new` pseudo-function that allocates a value of type `T` with the
specified fields. Can also be called without a second argument to leave the
allocated value with uninitialized fields.

In contrast to the pseudo-function, this only asserts that all fields match the
`fieldtypes` of `T`, rather than automatically converting them to those types.
"""
@generated new(::Type{T}) where {T} = Expr(:new, :T)
@generated new(::Type{T}, fields) where {T} =
    Expr(:splatnew, :T, :(fields::$(Tuple{fieldtypes(T)...})))

end # module
