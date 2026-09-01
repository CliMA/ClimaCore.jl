using Test
using LinearAlgebra
using Random
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

include("utils_dg.jl")  # dg_sphere_space, sw_flux, sw_wavespeed, sw_roeflux

# Two-point DG interface numerical-flux tests using ClimaCore's exported operators:
# - `Operators.CentralNumericalFlux`
# - `Operators.RusanovNumericalFlux`
# - Custom Roe numerical flux `sw_roeflux` (from utils_dg.jl)
#
# Properties verified:
#   1. Consistency: dissipation vanishes on single-valued (continuous) fields
#   2. Conservation: single-valued interface fluxes sum globally to zero for conserved scalars (ρ, ρθ)
#   3. Dual precision: Float32 and Float64
# (The zero-allocation sentinel for `add_numerical_flux_interior!` lives in
# unit_sphere_dg_fluxes.jl.)

@testset "Two-Point DG Numerical Fluxes on the Sphere" begin
    for FT in (Float32, Float64)
        ClimaComms.allowscalar(ClimaComms.device()) do
            space = dg_sphere_space(FT)
            @test !Spaces.is_continuous(space)
            coords = Fields.coordinate_field(space)

            params = (; g = FT(9.81))

            # Direct use of library exported numerical flux wrappers
            central = Operators.CentralNumericalFlux(sw_flux)
            rusanov = Operators.RusanovNumericalFlux(sw_flux, sw_wavespeed)
            roe = sw_roeflux

            shallow_water_state(ρ, uv, θ) =
                map((ρi, uvi, θi) -> (; ρ = ρi, ρu = ρi * uvi, ρθ = ρi * θi), ρ, uv, θ)

            base_test_state() = shallow_water_state(
                (@. FT(1) + FT(0.1) * sind(coords.long) * cosd(coords.lat)^2),
                (@. Geometry.UVVector(
                    cosd(coords.long),
                    -sind(coords.long) * sind(coords.lat),
                )),
                (@. FT(300) + FT(10) * cosd(coords.lat)),
            )

            @testset "Consistency (dissipation vanishes for continuous fields) [$FT]" begin
                y = base_test_state()
                rc = similar(y)
                fill!(parent(rc), 0)
                Operators.add_numerical_flux_interior!(central, rc, y, params)
                scale = maximum(abs, parent(rc))

                for numflux in (rusanov, roe)
                    r = similar(y)
                    fill!(parent(r), 0)
                    Operators.add_numerical_flux_interior!(numflux, r, y, params)
                    tol = 100 * eps(FT)
                    @test maximum(abs, parent(r) .- parent(rc)) < tol * scale
                end
            end

            @testset "Conservation (single-valued interface flux, node sum zero) [$FT]" begin
                y = base_test_state()
                Random.seed!(1234)
                y_parent_cpu = Array(parent(y))
                y_parent_cpu .+= FT(0.05) .* (rand(FT, size(y_parent_cpu)) .- FT(0.5))
                copyto!(parent(y), y_parent_cpu)

                for numflux in (central, rusanov, roe)
                    r = similar(y)
                    fill!(parent(r), 0)
                    Operators.add_numerical_flux_interior!(numflux, r, y, params)
                    tol = 100 * eps(FT)
                    for comp in (r.ρ, r.ρθ)
                        scale = sum(abs, parent(comp))
                        @test abs(sum(parent(comp))) < tol * scale
                    end
                end
            end
        end
    end
end
