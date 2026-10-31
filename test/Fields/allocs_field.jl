using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, DataLayouts, Operators, MatrixFields
import ClimaCore.MatrixFields: DiagonalMatrixRow

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

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(;
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)

# Temporaries in a loop over columns are stored in registers (or, on CPUs, in a thread's
# private buffer), which only stay off the heap when every function that writes them or
# reads them is inlined.
column_temporaries!(ᶜdest, ᶜx) =
    DataLayouts.foreach_column(ᶜdest, ᶜx) do ᶜdest, ᶜx
        ᶜtemp = @. 10 * ᶜx
        ᶠtemp = @. ᶠinterp(ᶜtemp) / 2
        ᶜmatrix = @. -(ᶜinterp_matrix()) * DiagonalMatrixRow(ᶠtemp)
        @. ᶜdest = ᶜtemp + ᶜinterp(ᶠtemp)
        @. ᶜdest += ᶜmatrix * ᶠtemp
        return nothing
    end

@testset "Temporaries in loops over columns do not allocate" begin
    for FT in (Float32, Float64)
        space = TU.CenterExtrudedFiniteDifferenceSpace(FT)
        ᶜx = ones(space)
        ᶜdest = zeros(space)
        TU.@test_zero_allocations column_temporaries!(ᶜdest, ᶜx)
    end
end
