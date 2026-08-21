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

@testset "Field broadcasts do not allocate" begin
    for FT in (Float32, Float64)
        space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
        x = ones(space)
        y = ones(space)
        dest = zeros(space)
        a = FT(2)
        TU.@test_zero_allocations axpy!(dest, a, x, y)
        TU.@test_zero_allocations scale!(dest, a, x)
    end
end
