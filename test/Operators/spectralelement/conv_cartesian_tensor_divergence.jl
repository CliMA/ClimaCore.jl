using Test
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaCore.Geometry: ⊗

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

include("utils_tensor_divergence.jl")

# h-refinement of `Operators.cartesian_tensor_divergence` against analytic
# divergences on the sphere, on both discretizations. The design rate for the
# divergence of a GLL{4} (degree-3) field is 3. Two references: v⊗m with m
# constant in the Cartesian frame, where the momentum components the derivative
# sees are constant; and u⊗u for solid-body rotation, where they vary with
# position and the exact answer carries a radial curvature term of the same
# order as the tangential ones.

@testset "Cartesian tensor divergence: h-convergence, v⊗m [$name]" for (
    name,
    discretization,
) in (
    ("DG sphere", Spaces.DG()),
    ("CG sphere", Spaces.CG()),
)
    FT = Float64
    R = FT(6.371e6)
    central = Operators.CentralNumericalFlux(identity)
    helems = (3, 6, 12)
    errs = zeros(FT, length(helems))
    for (i, helem) in enumerate(helems)
        space = tensor_div_sphere_space(FT; radius = R, helem, discretization)
        @test Spaces.is_continuous(space) == (discretization isa Spaces.CG)
        coords = Fields.coordinate_field(space)

        mcart = Geometry.Cartesian123Vector(FT(0.3), FT(-0.7), FT(0.5))
        mloc = local_cartesian_field(space, mcart)

        # v = (0, cosφ): ∇·v = -2 sinφ / R, so ∇·(v⊗m) = (-2 sinφ / R) m.
        v = @. Geometry.UVVector(FT(0), cosd(coords.lat))
        T = @. v ⊗ mloc
        completion = tensor_div_completion(space; numflux = central)
        dT = Operators.cartesian_tensor_divergence(T, completion)

        div_v_exact = @. -2 * sind(coords.lat) / R
        dT_exact = @. div_v_exact * mloc
        errs[i] = maximum(abs, parent(@. dT - dT_exact))
    end
    Δh = [FT(1) / helem for helem in helems]
    rates = TU.convergence_rate(errs, Δh)
    @info "Cartesian tensor divergence convergence, v⊗m ($name)" errs rates
    @test all(rates .>= 2.5)
    @test errs[end] < errs[1] / 50
end

@testset "Cartesian tensor divergence: h-convergence, u⊗u [$name]" for (
    name,
    discretization,
) in (
    ("DG sphere", Spaces.DG()),
    ("CG sphere", Spaces.CG()),
)
    FT = Float64
    R = FT(6.371e6)
    U = FT(20)
    central = Operators.CentralNumericalFlux(identity)
    helems = (3, 6, 12)
    errs = zeros(FT, length(helems))
    for (i, helem) in enumerate(helems)
        space = tensor_div_sphere_space(FT; radius = R, helem, discretization)
        coords = Fields.coordinate_field(space)
        u = solid_body_velocity(coords, U)
        T = @. u ⊗ u
        completion = tensor_div_completion(space; numflux = central)
        dT = Operators.cartesian_tensor_divergence(T, completion)
        dT_exact = solid_body_flux_divergence(coords, U, R)
        errs[i] = maximum(abs, parent(@. dT - dT_exact))
    end
    Δh = [FT(1) / helem for helem in helems]
    rates = TU.convergence_rate(errs, Δh)
    @info "Cartesian tensor divergence convergence, u⊗u ($name)" errs rates
    @test all(rates .>= 2.5)
    # 49.5x (DG) and 51.7x (CG) measured over the 4x refinement.
    @test errs[end] < errs[1] / 40
end
