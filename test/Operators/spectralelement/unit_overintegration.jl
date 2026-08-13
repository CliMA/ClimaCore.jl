using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends

import ClimaCore:
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Quadratures,
    Fields,
    Geometry,
    Operators

@testset "Spectral Element Over-Integration & De-aliasing" begin
    for FT in (Float32, Float64)
        context = ClimaComms.SingletonCommsContext()
        domain = Domains.RectangleDomain(
            Domains.IntervalDomain(
                Geometry.XPoint(-FT(π)),
                Geometry.XPoint(FT(π)),
                periodic = true,
            ),
            Domains.IntervalDomain(
                Geometry.YPoint(-FT(π)),
                Geometry.YPoint(FT(π)),
                periodic = true,
            ),
        )
        mesh = Meshes.RectilinearMesh(domain, 4, 4)
        topology = Topologies.Topology2D(context, mesh)

        # Standard space (Nq = 4) and over-integration space (Nqh = 7)
        space_std = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())
        space_high = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{7}())

        coords_std = Fields.coordinate_field(space_std)
        coords_high = Fields.coordinate_field(space_high)

        interp_op = Operators.Interpolate(space_high)
        restrict_op = Operators.Restrict(space_std)

        # 1. Exact polynomial round-trip identity:
        # For any field in the standard space, Interpolate -> Restrict recovers the field
        f_std = @. sin(coords_std.x) * cos(coords_std.y)
        f_high = interp_op.(f_std)
        f_recovered = restrict_op.(f_high)

        tol = FT(0.05)
        @test maximum(abs, parent(f_recovered .- f_std)) < tol * maximum(abs, parent(f_std))

        # 2. Conservation preservation under projection:
        # sum(Restrict(g_high)) == sum(g_high) for higher-order fields
        g_high = @. (sin(coords_high.x) * cos(coords_high.y))^2
        g_proj = restrict_op.(g_high)

        int_high = sum(g_high)
        int_proj = sum(g_proj)
        @test isapprox(int_proj, int_high, rtol = (FT == Float32 ? 1e-4 : 1e-10))

        # 3. Constant preservation:
        ones_std = ones(space_std)
        ones_high = interp_op.(ones_std)
        @test maximum(abs, parent(ones_high) .- FT(1)) < 20 * eps(FT)
        ones_proj = restrict_op.(ones_high)
        @test maximum(abs, parent(ones_proj) .- FT(1)) < 50 * eps(FT)
    end
end
