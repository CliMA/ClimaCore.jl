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
# GradientC2F), an upwind product, and a mass-flux divergence must not
# allocate at runtime. These are the FD hot paths of every column physics
# right-hand side.

laplacian!(dest, f, gradc2f, divf2c) = (@. dest = divf2c(gradc2f(f)); nothing)
upwind!(dest, w, f, upwindc2f, divf2c) =
    (@. dest = divf2c(upwindc2f(w, f)); nothing)
mass_flux_div!(dest, J, ρ, u³, winterpc2f, divf2c) =
    (@. dest = divf2c(winterpc2f(J, ρ) * u³); nothing)
# The SetValue boundary faces of C2F operators read the closest centers of their stencil
# arguments, which must not be materialized.
setvalue_chain2!(dest, f, gradc2f, divf2c) =
    (@. dest = divf2c(gradc2f(divf2c(gradc2f(f)))); nothing)
setvalue_chain3!(dest, f, gradc2f, divf2c) =
    (@. dest = divf2c(gradc2f(divf2c(gradc2f(divf2c(gradc2f(f)))))); nothing)
setvalue_upwind!(dest, w, f, upwindc2f, interpc2f, interpf2c, divf2c) =
    (@. dest = divf2c(upwindc2f(interpc2f(interpf2c(w)), interpf2c(interpc2f(f)))); nothing)

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

        # A lazy flux with zero-flux conditions, on a space whose number of columns is
        # only known at run time.
        hcspace = TU.CenterExtrudedFiniteDifferenceSpace(FT)
        ρ = 1 .+ Fields.coordinate_field(hcspace).z
        J = Fields.local_geometry_field(hcspace).J
        u³ = Geometry.Contravariant3Vector.(ones(Spaces.face_space(hcspace)))
        winterpc2f = Operators.WeightedInterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        zero_flux = Operators.SetValue(Geometry.Contravariant3Vector(FT(0)))
        flux_divf2c = Operators.DivergenceF2C(bottom = zero_flux, top = zero_flux)
        hdest = zeros(hcspace)
        TU.@test_zero_allocations mass_flux_div!(hdest, J, ρ, u³, winterpc2f, flux_divf2c)

        setvalue_gradc2f = Operators.GradientC2F(
            bottom = Operators.SetValue(FT(1)),
            top = Operators.SetValue(FT(0)),
        )
        setvalue_upwindc2f = Operators.UpwindBiasedProductC2F(
            bottom = Operators.SetValue(FT(1)),
            top = Operators.SetValue(FT(0)),
        )
        extrapolate = Operators.Extrapolate()
        interpc2f = Operators.InterpolateC2F(bottom = extrapolate, top = extrapolate)
        interpf2c = Operators.InterpolateF2C()
        for (dest, f) in ((dest, f), (hdest, ρ))
            ᶠw = Geometry.WVector.(ones(Spaces.face_space(axes(f))))
            TU.@test_zero_allocations setvalue_chain2!(dest, f, setvalue_gradc2f, divf2c)
            TU.@test_zero_allocations setvalue_chain3!(dest, f, setvalue_gradc2f, divf2c)
            TU.@test_zero_allocations setvalue_upwind!(
                dest,
                ᶠw,
                f,
                setvalue_upwindc2f,
                interpc2f,
                interpf2c,
                divf2c,
            )
        end
    end
end
