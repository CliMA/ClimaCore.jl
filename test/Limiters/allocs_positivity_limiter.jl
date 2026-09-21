using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Limiters

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# Applying the PositivityLimiter must not allocate at runtime: the per-slab
# work runs on tuples, and `g`, the floors and the state names are compiled
# into the bound limiter. In the `:allocs` tier because it needs a warm-up
# run; the correctness tests are in positivity_limiter.jl.

ideal_gas_p(ρ, ρe, u1, u2, u3, aux) =
    (oftype(ρ, 1.4) - 1) * (ρe - (u1^2 + u2^2 + u3^2) / (2 * ρ) - ρ * aux)
zs_pressure(U, aux) = ideal_gas_p(U.ρ, U.ρe, U.ρu1, U.ρu2, U.ρu3, aux)

@testset "PositivityLimiter does not allocate" begin
    for FT in (Float32, Float64)
        space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
        state(val) = fill(FT(val), space)
        (ρ, ρe, ρu1, ρu2, ρu3, ρq, aux) =
            state.((1, 10, 1 / 10, -1 / 5, 3 / 10, 1 / 1000, 2))
        # One inadmissible node, so both limiter steps do work.
        parent(ρ)[1, 1, 1, 1, 1] = -FT(1) / 10
        lim = Limiters.PositivityLimiter(
            FT;
            floors = (; ρ = FT(1e-6), ρq = 0),
            g = zs_pressure,
            g_min = FT(1e-2),
            maxiter = 30,
        )
        states = (; ρ, ρe, ρu1, ρu2, ρu3, ρq)
        TU.@test_zero_allocations Limiters.apply_limiter!(states, aux, lim)
    end
end
