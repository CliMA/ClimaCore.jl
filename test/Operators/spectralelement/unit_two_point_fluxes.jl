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
import ClimaCore.Geometry: ⊗

include("utils_dg.jl")  # dg_sphere_space

# Two-point DG interface numerical-flux tests using ClimaCore's exported operators:
# - `Operators.CentralNumericalFlux`
# - `Operators.RusanovNumericalFlux`
# - Custom Roe numerical flux (as functor)
#
# Properties verified:
#   1. Consistency: dissipation vanishes on single-valued (continuous) fields
#   2. Conservation: single-valued interface fluxes sum globally to zero for conserved scalars (ρ, ρθ)
#   3. Dual precision: Float32 and Float64
# (The zero-allocation sentinel for `add_numerical_flux_internal!` lives in
# unit_sphere_dg_fluxes.jl.)

function sw_flux(state, p)
    ρ, ρu, ρθ = state.ρ, state.ρu, state.ρθ
    u = ρu / ρ
    FT = eltype(ρ)
    I_tensor =
        (Geometry.UVVector(FT(1), FT(0)) ⊗ Geometry.UVVector(FT(1), FT(0))) +
        (Geometry.UVVector(FT(0), FT(1)) ⊗ Geometry.UVVector(FT(0), FT(1)))
    return (
        ρ = ρu,
        ρu = (ρu ⊗ u) + (p.g * ρ^2 / 2) * I_tensor,
        ρθ = ρθ * u,
    )
end

function sw_wavespeed(state, p)
    return sqrt(p.g)
end

function roe_average(ρ⁻, ρ⁺, v⁻, v⁺)
    return (sqrt(ρ⁻) * v⁻ + sqrt(ρ⁺) * v⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))
end

function sw_roeflux(normal, (y⁻, params⁻), (y⁺, params⁺))
    λ = sqrt(params⁻.g)
    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * normal

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * normal

    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)
    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    ρ̄ = sqrt(ρ⁻ * ρ⁺)
    ū = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ̄ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c̄ = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)
    ūₙ = ū' * normal

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * normal

    c⁻² = 1 / c̄^2
    w1 = abs(ūₙ - c̄) * (Δp - ρ̄ * c̄ * Δuₙ) * 0.5 * c⁻²
    w2 = abs(ūₙ + c̄) * (Δp + ρ̄ * c̄ * Δuₙ) * 0.5 * c⁻²
    w3 = abs(ūₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(ūₙ) * ρ̄
    w5 = abs(ūₙ) * (Δρθ - θ̄ * Δp * c⁻²)

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (
            w1 * (ū - c̄ * normal) + w2 * (ū + c̄ * normal) + w3 * ū +
            w4 * (Δu - Δuₙ * normal)
        ) * 0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ̄ + w5) * 0.5
    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)

    F⁻ = sw_flux(y⁻, params⁻)
    F⁺ = sw_flux(y⁺, params⁺)
    return (
        ρ = ((F⁻.ρ + F⁺.ρ) / 2)' * normal + Δf.ρ,
        ρu = ((F⁻.ρu + F⁺.ρu) / 2)' * normal + Δf.ρu,
        ρθ = ((F⁻.ρθ + F⁺.ρθ) / 2)' * normal + Δf.ρθ,
    )
end

@testset "Two-Point DG Numerical Fluxes on the Sphere" begin
    for FT in (Float32, Float64)
        space = dg_sphere_space(FT)
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
            Operators.add_numerical_flux_internal!(central, rc, y, params)
            scale = maximum(abs, parent(rc))

            for numflux in (rusanov, roe)
                r = similar(y)
                fill!(parent(r), 0)
                Operators.add_numerical_flux_internal!(numflux, r, y, params)
                tol = 100 * eps(FT)
                @test maximum(abs, parent(r) .- parent(rc)) < tol * scale
            end
        end

        @testset "Conservation (single-valued interface flux, node sum zero) [$FT]" begin
            y = base_test_state()
            Random.seed!(1234)
            parent(y) .+= FT(0.05) .* (rand(FT, size(parent(y))) .- FT(0.5))

            for numflux in (central, rusanov, roe)
                r = similar(y)
                fill!(parent(r), 0)
                Operators.add_numerical_flux_internal!(numflux, r, y, params)
                tol = 100 * eps(FT)
                for comp in (r.ρ, r.ρθ)
                    scale = sum(abs, parent(comp))
                    @test abs(sum(parent(comp))) < tol * scale
                end
            end
        end
    end
end
