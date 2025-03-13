module Utilities

include("plushalf.jl")
include("cache.jl")

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


end # module
