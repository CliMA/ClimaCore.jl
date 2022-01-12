module Limiters

import ClimaCore: slab, Fields, Topologies, Spaces

export AbstractLimiter

"""
    AbstractLimiter

Supertype for all limiters.

# Interfaces

- [`quasimonotone_limiter!`](@ref)
"""
abstract type AbstractLimiter end

# implementations
include("limiter.jl")

end # end module
