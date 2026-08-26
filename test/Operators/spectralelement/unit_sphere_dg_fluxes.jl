using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

import ClimaCore  # for `pkgdir` below
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

include("utils_dg.jl")  # dg_sphere_space, dg_jump_penalty

# DG interface-flux consistency on the cubed-sphere spectral element space:
# the jump penalty must vanish pointwise on continuous (single-valued) fields,
# and applying it must not allocate. Global conservation and the weak/strong
# (SBP) divergence identity for the central flux are covered — for perturbed,
# non-constant states, on both the sphere and a plane — in
# unit_dg_stability.jl.

# The `for FT` closure below captures `FT::DataType`, so every local derived
# from it is `Any`-typed and an inline `@allocated` there measures dynamic
# dispatch, not the operator. This barrier specializes on the concrete
# argument types, so the measurement runs in a typed frame.
@noinline function measured_flux_allocs(fn::F, r, q) where {F}
    Operators.add_numerical_flux_internal!(fn, r, q)
    return @allocated Operators.add_numerical_flux_internal!(fn, r, q)
end

@testset "Cubed-Sphere DG Interface Fluxes" begin
    for FT in (Float32, Float64)
        ClimaComms.allowscalar(ClimaComms.device()) do
            space = dg_sphere_space(FT)
            @test !Spaces.is_continuous(space)
            coords = Fields.coordinate_field(space)
            lgeom = Fields.local_geometry_field(space)

            smooth_scalar(coords) =
                @. sind(coords.long) * cosd(coords.lat)^2

            @testset "Penalty flux vanishes for continuous fields [$FT]" begin
                q = smooth_scalar(coords)
                r = similar(q)
                fill!(parent(r), 0)
                Operators.add_numerical_flux_internal!(dg_jump_penalty, r, q)
                rn = @. r / lgeom.WJ
                tol = FT == Float32 ? 1e-4 : 1e-10
                @test maximum(abs, parent(rn)) < tol * maximum(abs, parent(q))
            end

            @testset "Zero allocation sentinel [$FT]" begin
                q = smooth_scalar(coords)
                r = similar(q)
                fill!(parent(r), 0)
                allocs = measured_flux_allocs(dg_jump_penalty, r, q)
                if !(ClimaComms.device() isa ClimaComms.CUDADevice) &&
                   TU.allocation_checks_meaningful()
                    @test allocs == 0
                end
            end
        end
    end
end
