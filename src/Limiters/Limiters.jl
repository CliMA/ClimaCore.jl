module Limiters

import ..DataLayouts, ..Topologies, ..Spaces, ..Fields
import ..DebugOnly: call_post_op_callback, post_op_callback
import ClimaCore: slab

export AbstractLimiter, QuasiMonotoneLimiter, VerticalMassBorrowingLimiter

"""
    AbstractLimiter

Supertype for all limiters.

# Interfaces

- [`apply_limiter!`](@ref)
"""
abstract type AbstractLimiter end

# implementations
include("quasimonotone.jl")
include("vertical_mass_borrowing_limiter.jl")

end # end module
