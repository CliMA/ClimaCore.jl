using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# In-place broadcasts over a Field must not allocate at runtime. These are
# regression tests for the allocation budget (see the dev-guide's warm-up +
# `@allocated == 0` pattern); they live in the `:allocs` tier because measuring
# runtime allocations requires a warm-up run first.

axpy!(dest, a, x, y) = (@. dest = a * x + y; nothing)
scale!(dest, a, x) = (@. dest = a * x; nothing)
weighted_sum(p, x1, x2, x3, x4, x5) = p.a * (x1 + x2 + x3) - p.b * x4 / x5
weighted_sum!(dest, p, x) =
    (@. dest = weighted_sum((p,), x[1], x[2], x[3], x[4], x[5]); nothing)

@testset "Field broadcasts do not allocate" begin
    for FT in (Float32, Float64)
        space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
        x = ones(space)
        y = ones(space)
        dest = zeros(space)
        a = FT(2)
        TU.@test_zero_allocations axpy!(dest, a, x, y)
        TU.@test_zero_allocations scale!(dest, a, x)
        # A parameter struct and several Fields in one broadcast, like the
        # thermodynamic broadcasts in ClimaAtmos: the broadcast expression has
        # more layout arguments than the (dest, bc) pair that copyto! loops
        # over, so its DataScope must be inferred without widening.
        p = (; a = FT(2), b = FT(3))
        xs = ntuple(_ -> ones(space), 5)
        TU.@test_zero_allocations weighted_sum!(dest, p, xs)
    end
end
