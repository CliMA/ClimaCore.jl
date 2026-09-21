module Limiters

import ..DataLayouts, ..Topologies, ..Spaces, ..Fields
using UnrolledUtilities
import ..DebugOnly: call_post_op_callback, post_op_callback
import ClimaCore: slab

export AbstractLimiter,
    QuasiMonotoneLimiter, VerticalMassBorrowingLimiter, PositivityLimiter

"""
    AbstractLimiter

Supertype for all limiters.

Subtypes:

  - [`QuasiMonotoneLimiter`](@ref): horizontal quasi-monotone flux limiter for spectral
    element advection.
  - [`PositivityLimiter`](@ref): Zhang-Shu mean-preserving positivity limiter for
    conserved DG states.
  - [`VerticalMassBorrowingLimiter`](@ref): vertical mass-borrowing limiter that removes
    negative tracer mass.

Subtypes implement [`apply_limiter!`](@ref).
"""
abstract type AbstractLimiter end

# implementations
include("quasimonotone.jl")
include("positivity.jl")
include("vertical_mass_borrowing_limiter.jl")

end # end module
