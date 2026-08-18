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
    hwdiv = Operators.WeakDivergence()
    F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), uv)
    r = @. hwdiv(F) * (-(lgeom.WJ))
    Operators.add_numerical_flux_internal!(dg_central_flux_uv, r, uv)
    return @. -r / lgeom.WJ
end
