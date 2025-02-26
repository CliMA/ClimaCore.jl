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

end # module
