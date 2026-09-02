using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Fields, Spaces, Operators, Geometry

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# Type-inference gates for operator broadcasts (`:inference` tier): the
# materialization of finite-difference and spectral-element operator
# broadcasts must be fully inferred, for both precisions.

apply1(op, f) = op.(f)
apply2(op_outer, op_inner, f) = op_outer.(op_inner.(f))

@testset "Operator broadcast inference" begin
    TU.@test_precisions FT begin
        # Column finite-difference operators
        cspace = TU.ColumnCenterFiniteDifferenceSpace(FT)
        f = sin.(Fields.coordinate_field(cspace).z)
        gradc2f = Operators.GradientC2F(
            bottom = Operators.SetGradient(Geometry.WVector(FT(0))),
            top = Operators.SetGradient(Geometry.WVector(FT(0))),
        )
        divf2c = Operators.DivergenceF2C()
        @test (@inferred apply1(gradc2f, f)) isa Fields.Field
        @test (@inferred apply2(divf2c, gradc2f, f)) isa Fields.Field

        # Spectral-element operators on the sphere
        sspace = TU.SphereSpectralElementSpace(FT)
        coords = Fields.coordinate_field(sspace)
        g = @. sind(coords.long) * cosd(coords.lat)
        grad = Operators.Gradient()
        wdiv = Operators.Divergence{Operators.WeakForm}()
        @test (@inferred apply1(grad, g)) isa Fields.Field
        @test (@inferred apply2(wdiv, grad, g)) isa Fields.Field
    end
end
