# Shared helpers for the DG interface-flux tests (unit_two_point_fluxes.jl,
# unit_sphere_dg_fluxes.jl, unit_dg_stability.jl). Definitions only — no
# top-level @test (see test/README.md on `utils_` files).
import ClimaComms
import ClimaCore:
    Domains, Meshes, Topologies, Spaces, Quadratures, Geometry

# The standard cubed-sphere spectral-element space used by the DG tests.
function dg_sphere_space(
    ::Type{FT};
    radius = FT(6.371e6),
    helem = 4,
    Nq = 4,
) where {FT}
    context = ClimaComms.SingletonCommsContext()
    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    return Spaces.SpectralElementSpace2D(htopology, Quadratures.GLL{Nq}())
end

# Central numerical flux of the physical flux q*u through the face normal,
# for a state `(; q, uv)`.
dg_central_flux(normal, (y⁻,), (y⁺,)) =
    ((y⁻.q * y⁻.uv + y⁺.q * y⁺.uv) / 2)' * normal

# Antisymmetric single-valued jump penalty on a scalar.
dg_jump_penalty(normal, (q⁻,), (q⁺,)) = (q⁻ - q⁺) / 2
