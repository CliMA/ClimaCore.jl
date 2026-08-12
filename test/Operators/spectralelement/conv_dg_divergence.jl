using Test
using LinearAlgebra
using IntervalSets
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU

# h-refinement convergence of the DG divergence: weak divergence plus central
# interface flux (the element-local DG form used by the shallow-water DG smoke
# test), applied to a smooth vector field on a doubly-periodic plane, must
# converge to the analytic divergence at the design order of the GLL{Nq}
# element (≈ Nq - 1 or better). This is the DG counterpart of the CG
# conv_sphere_* tests; the structural (SBP/conservation) properties are covered
# in unit_dg_stability.jl.

function periodic_plane(::Type{FT}, nelem; L = FT(2π), Nq = 4) where {FT}
    context = ClimaComms.SingletonCommsContext()
    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(zero(L)) .. Geometry.XPoint{FT}(L),
        Geometry.YPoint{FT}(zero(L)) .. Geometry.YPoint{FT}(L);
        x1periodic = true,
        x2periodic = true,
    )
    mesh = Meshes.RectilinearMesh(domain, nelem, nelem)
    topology = Topologies.Topology2D(context, mesh)
    return Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{Nq}())
end

central_flux(normal, (uv⁻,), (uv⁺,)) = ((uv⁻ + uv⁺) / 2)' * normal

# DG divergence of F: weak divergence in the element interior plus the central
# numerical flux through element interfaces, normalized by the mass weights.
function dg_divergence(uv)
    space = axes(uv)
    lgeom = Fields.local_geometry_field(space)
    hwdiv = Operators.WeakDivergence()
    F = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), uv)
    r = @. hwdiv(F) * (-(lgeom.WJ))
    Operators.add_numerical_flux_internal!(central_flux, r, uv)
    return @. -r / lgeom.WJ
end

@testset "DG divergence h-refinement convergence" begin
    FT = Float64
    nelems = (4, 8, 16)
    errs = zeros(FT, length(nelems))
    for (i, nelem) in enumerate(nelems)
        space = periodic_plane(FT, nelem)
        coords = Fields.coordinate_field(space)
        uv = @. Geometry.UVVector(sin(coords.x), sin(coords.y))
        div_exact = @. cos(coords.x) + cos(coords.y)
        div_dg = dg_divergence(uv)
        errs[i] = maximum(abs, parent(div_dg) .- parent(div_exact))
    end
    Δh = [FT(1) / nelem for nelem in nelems]
    rates = TU.convergence_rate(errs, Δh)
    @info "DG divergence convergence" errs rates
    # GLL{4} elements: design order ≈ 3; require it within a safety margin and
    # sanity-bound it from above (superconvergence can push it toward Nq).
    @test all(rates .>= 2.5)
    @test all(rates .<= 5.5)
    @test errs[end] < errs[1] / 50
end
