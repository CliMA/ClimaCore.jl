using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Spaces, Operators, Geometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# Allocation regression gates for in-place finite-difference stencil
# broadcasts (`:allocs` tier): the column Laplacian (DivergenceF2C ∘
# GradientC2F) and an upwind product must not allocate at runtime. These
# are the FD hot paths of every column physics right-hand side.

laplacian!(dest, f, gradc2f, divf2c) = (@. dest = divf2c(gradc2f(f)); nothing)
upwind!(dest, w, f, upwindc2f, divf2c) =
    (@. dest = divf2c(upwindc2f(w, f)); nothing)

@testset "FD stencil broadcasts do not allocate" begin
    TU.@test_precisions FT begin
        cspace = TU.ColumnCenterFiniteDifferenceSpace(FT)
        fspace = Spaces.FaceFiniteDifferenceSpace(cspace)
        f = sin.(Fields.coordinate_field(cspace).z)
        dest = zeros(cspace)

        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
            top = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        divf2c = Operators.DivergenceF2C()
        TU.@test_zero_allocations laplacian!(dest, f, gradc2f, divf2c)

        w = Geometry.WVector.(ones(fspace))
        upwindc2f = Operators.UpwindBiasedProductC2F(
            bottom = Operators.Extrapolate(0),
            top = Operators.Extrapolate(0),
        )
        TU.@test_zero_allocations upwind!(dest, w, f, upwindc2f, divf2c)
    end
end
