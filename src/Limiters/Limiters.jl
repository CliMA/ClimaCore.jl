module Limiters

import ..DataLayouts, ..Topologies, ..Spaces, ..Fields
import ..RecursiveApply: rdiv, rmin, rmax
import ..DebugOnly: call_post_op_callback, post_op_callback
import ClimaCore: slab

export AbstractLimiter

"""
    AbstractLimiter

Supertype for all limiters.

# Interfaces

- [`apply_limiter!`](@ref)
"""
abstract type AbstractLimiter end

# implementations
include("quasimonotone.jl")

end # end module
