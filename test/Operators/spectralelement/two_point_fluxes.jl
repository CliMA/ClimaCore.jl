using Test
using ClimaComms
using LinearAlgebra
using Random
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

# Interface (two-point) numerical-flux tests for 
# `RusanovNumericalFlux` and `RoeNumericalFlux`, exercised on a 2D cubed-sphere
# spectral-element space with the shallow-water state (ρ, ρu, ρθ). The flux,
# wavespeed and Roe-average match examples/bickleyjet/bickleyjet_dg.jl.
#
# TODO : Refactor/ unify API
#
# Two properties are checked:
#   1. Consistency — on a single-valued (continuous) interface state the added
#      dissipation vanishes, so the flux reduces exactly to the central flux.
#   2. Conservation — the flux is single-valued across each interface: it adds
#      equal-and-opposite contributions to the two adjacent elements, so each
#      scalar tendency is globally conserved (its total node sum is zero) even
#      when the state is discontinuous (dissipation active).

# TODO Extend to FT = (Float32, Float64)
const FT = Float64

function sphere_space(; radius = FT(6.371e6), helem = 4, Nq = 4)
    context = ClimaComms.context()
    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    return Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{Nq}())
end

space = sphere_space()
coords = Fields.coordinate_field(space)

const params = (; g = FT(9.81))

# Shallow-water physical flux, max wavespeed and Roe average
function sw_flux(state, p)
    u = state.ρu / state.ρ
    return (
        ρ = state.ρu,
        ρu = (state.ρu ⊗ u) + (p.g * state.ρ^2 / 2) * I,
        ρθ = state.ρθ * u,
    )
end
sw_wavespeed(state, p) = sqrt(p.g)
roe_average(ρ⁻, ρ⁺, v⁻, v⁺) =
    (sqrt(ρ⁻) * v⁻ + sqrt(ρ⁺) * v⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

central = Operators.CentralNumericalFlux(sw_flux)
rusanov = Operators.RusanovNumericalFlux(sw_flux, sw_wavespeed)
roe = Operators.RoeNumericalFlux(sw_flux, roe_average)

# Assemble a shallow-water state field from ρ, velocity and θ fields.
shallow_water_state(ρ, uv, θ) =
    map((ρi, uvi, θi) -> (; ρ = ρi, ρu = ρi * uvi, ρθ = ρi * θi), ρ, uv, θ)

# Smooth, single-valued base state (solid-body-like flow).
base_test_state() = shallow_water_state(
    (@. FT(1) + FT(0.1) * sind(coords.long) * cosd(coords.lat)^2),
    (@. Geometry.UVVector(
        cosd(coords.long),
        -sind(coords.long) * sind(coords.lat),
    )),
    (@. FT(300) + FT(10) * cosd(coords.lat)),
)

@testset "Rusanov/Roe consistency: dissipation vanishes for continuous fields" begin
    y = base_test_state()

    rc = similar(y)
    fill!(parent(rc), 0)
    Operators.add_numerical_flux_interior!(central, rc, y, params)
    scale = maximum(abs, parent(rc))

    for numflux in (rusanov, roe)
        r = similar(y)
        fill!(parent(r), 0)
        Operators.add_numerical_flux_interior!(numflux, r, y, params)
        # numerical flux = central + dissipation(jump); the jump is (near) zero
        # for a single-valued state, so the dissipation must vanish.
        @test maximum(abs, parent(r) .- parent(rc)) < 1e-10 * scale
    end
end

@testset "Rusanov/Roe conservation: single-valued interface flux (node sum zero)" begin
    y = base_test_state()
    # Inject per-node discontinuities so the dissipation is genuinely active;
    # kept small enough that ρ stays strictly positive (Roe needs √ρ).
    Random.seed!(1234)
    parent(y) .+= FT(0.05) .* (rand(FT, size(parent(y))) .- FT(0.5))

    for numflux in (rusanov, roe)
        r = similar(y)
        fill!(parent(r), 0)
        Operators.add_numerical_flux_interior!(numflux, r, y, params)
        # ρ and ρθ are advected scalars: each interface adds ∓sWJ·flux to its two
        # nodes, and these equal-and-opposite contributions cancel in the global
        # sum, so each scalar is globally conserved (total node sum is zero). (ρu
        # lives in rotating local frames and is not summed this way.)
        for comp in (r.ρ, r.ρθ)
            scale = sum(abs, parent(comp))
            @test abs(sum(parent(comp))) < 1e-12 * scale
        end
    end
end
