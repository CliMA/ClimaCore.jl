# Shared helpers for the DG interface-flux tests (unit_two_point_fluxes.jl,
# unit_sphere_dg_fluxes.jl, unit_dg_stability.jl). Definitions only — no
# top-level @test (see test/README.md on `utils_` files).
import ClimaComms
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import ClimaCore.Geometry: ⊗

# The standard cubed-sphere spectral-element space used by the DG tests.
# `discretization = Spaces.DG()` marks the grid as DG (skips DSS; see
# `test/Spaces/unit_discontinuous_spaces.jl`). The distributed tests pass an
# MPI `context` to partition the sphere across ranks.
function dg_sphere_space(
    ::Type{FT};
    radius = FT(6.371e6),
    helem = 4,
    Nq = 4,
    context = ClimaComms.SingletonCommsContext(),
) where {FT}
    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    return Spaces.SpectralElementSpace2D(
        htopology,
        Quadratures.GLL{Nq}();
        discretization = Spaces.DG(),
    )
end

# Central numerical flux of the physical flux q*u through the face normal,
# for a state `(; q, uv)`.
dg_central_flux(normal, (y⁻,), (y⁺,)) =
    ((y⁻.q * y⁻.uv + y⁺.q * y⁺.uv) / 2)' * normal

# Antisymmetric single-valued jump penalty on a scalar.
dg_jump_penalty(normal, (q⁻,), (q⁺,)) = (q⁻ - q⁺) / 2

# Central numerical flux of the vector field itself through the face normal.
dg_central_flux_uv(normal, (uv⁻,), (uv⁺,)) = ((uv⁻ + uv⁺) / 2)' * normal

# DG divergence of a vector field: weak divergence in the element interior
# plus the central numerical flux through element interfaces, normalized by
# the mass weights — the element-local form used by the shallow-water DG smoke
# test. The interface flux completes the weak volume term at element
# boundaries; DG applies no DSS.
function dg_divergence(uv)
    space = axes(uv)
    lgeom = Fields.local_geometry_field(space)
    hwdiv = Operators.Divergence{Operators.WeakForm}()
    F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), uv)
    r = @. hwdiv(F) * (-(lgeom.WJ))
    Operators.add_numerical_flux_interior!(dg_central_flux_uv, r, uv)
    return @. -r / lgeom.WJ
end

# ---------------------------------------------------------------------------
# Shallow-water physical flux + Roe interface flux (test-local; not Operators)
# ---------------------------------------------------------------------------

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

sw_wavespeed(state, p) = sqrt(p.g)

roe_average(ρ⁻, ρ⁺, v⁻, v⁺) =
    (sqrt(ρ⁻) * v⁻ + sqrt(ρ⁺) * v⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

"""
    sw_roeflux(normal, (y⁻, params⁻), (y⁺, params⁺))

Shallow-water Roe numerical flux for the conserved state `(; ρ, ρu, ρθ)`.
Central flux of `sw_flux` plus Roe dissipation (wave strengths `w1`–`w5`).
Used by `unit_two_point_fluxes.jl` as a custom functor.
"""
function sw_roeflux(normal, (y⁻, params⁻), (y⁺, params⁺))
    λ = sqrt(params⁻.g)
    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺

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
