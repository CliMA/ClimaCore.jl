module Limiters

import ..DataLayouts, ..Topologies, ..Spaces, ..Fields
import ..RecursiveApply: rdiv, rmin, rmax
import ClimaCore: slab

export AbstractLimiter

"""
    AbstractLimiter

Supertype for all limiters.

# Interfaces

- [`apply_limiter!`](@ref)
"""
abstract type AbstractLimiter end

# limiter utilities
include("limiter_utils.jl")

# implementations
include("quasimonotone.jl")
include("clip_and_sum.jl")

end # end module
