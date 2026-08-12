# h-refinement of the inter-element jump of the strong-form divergence.
#
# One element's divergence of a smooth field is consistent up to its boundary,
# so the discontinuity across coincident boundary nodes — what `weighted_dss!`
# averages away — must vanish at the design rate O(h^(Nq-1)), and so must the
# divergence of a toroidal VSH, which is pure error. These rates are the
# threshold-free form of the oscillation checks in
# `unit_sphere_vsh_divergence.jl`: element-boundary oscillations that are
# anomalously strong break them even while a global error norm still
# converges.

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore
import ClimaCore: Domains, Meshes, Operators, Quadratures, Spaces, Topologies

include("utils_vsh_divergence.jl")
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities: convergence_rate

const FT = Float64

@testset "strong divergence inter-element jump: h-refinement" begin
    (l, m) = (4, 3)
    radius = FT(2)
    Nq = 4
    Nes = [3, 9, 27] # odd, so no node sits at a pole
    sdiv = Operators.Divergence()

    jump_S = zeros(FT, length(Nes))
    max_T = zeros(FT, length(Nes))
    Δh = zeros(FT, length(Nes))
    for (i, Ne) in enumerate(Nes)
        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, Ne)
        topology = Topologies.Topology2D(ClimaComms.context(), mesh)
        space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{Nq}())
        positions = unit_sphere_positions(space)
        scale =
            l * (l + 1) / radius^2 *
            maximum(abs, parent(ylm_field(l, m, space)))

        S = vsh_field(spheroidal_uv, l, m, space, radius)
        T = vsh_field(toroidal_uv, l, m, space, radius)
        jump_S[i] = max_interelement_jump(sdiv.(S), positions) / scale
        max_T[i] = maximum(abs, parent(sdiv.(T))) / scale
        Δh[i] = 1 / Ne
    end

    # Measured rates: ≈ 2.82 for the jump, ≈ 2.8-2.9 for the toroidal error.
    for rate in convergence_rate(jump_S, Δh)
        @test rate ≈ Nq - 1 atol = 0.5
    end
    for rate in convergence_rate(max_T, Δh)
        @test rate ≈ Nq - 1 atol = 0.5
    end
    # Absolute pins at the finest resolution (measured 3.0e-4 and 7.6e-5).
    @test jump_S[end] ≤ 1e-3
    @test max_T[end] ≤ 3e-4
end
