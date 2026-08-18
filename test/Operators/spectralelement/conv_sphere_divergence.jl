using Test
using ClimaComms
using StaticArrays, IntervalSets
using ClimaCore
import ClimaCore:
    Fields,
    Domains,
    Meshes,
    Topologies,
    Spaces,
    Operators,
    Geometry,
    Quadratures
using StaticArrays, IntervalSets, LinearAlgebra

FT = Float64

radius = FT(1)
domain = Domains.SphereDomain(radius)

sdiv = Operators.Divergence()
wdiv = Operators.WeakDivergence()

@testset "divergence" begin
    Ne = 7
    Nq = 6

    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )

    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    u_local = @. Geometry.UVVector(
        cosd(coords.long),
        -sind(coords.long) * sind(coords.lat),
    )

    u = Geometry.transform.(Ref(Geometry.Contravariant12Axis()), u_local)

    div_us = sdiv.(u)
    Spaces.weighted_dss!(div_us)

    div_uw = wdiv.(u)
    Spaces.weighted_dss!(div_uw)

    div_exact = @. -2 * sind(coords.long) * cosd(coords.lat) / radius

    @test div_us ≈ div_exact rtol = 1e-2
    @test div_uw ≈ div_exact rtol = 1e-2
end

@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities: convergence_rate

@testset "convergence tests for the divergence operator on the sphere" begin
    Nes = [3, 9, 27]
    Nqs = [4, 6]

    for (Iq, Nq) in enumerate(Nqs)
        err_sdiv = Array{FT}(undef, length(Nes))
        err_wdiv = Array{FT}(undef, length(Nes))
        Δh = Array{FT}(undef, length(Nes))

        for (Ie, Ne) in enumerate(Nes)
            mesh = Meshes.EquiangularCubedSphere(domain, Ne)
            grid_topology = Topologies.Topology2D(
                ClimaComms.SingletonCommsContext(
                    ClimaComms.CPUSingleThreaded(),
                ),
                mesh,
            )

            quad = Quadratures.GLL{Nq}()
            space = Spaces.SpectralElementSpace2D(grid_topology, quad)
            coords = Fields.coordinate_field(space)

            u_local = @. Geometry.UVVector(
                cosd(coords.long),
                -sind(coords.long) * sind(coords.lat),
            )

            u =
                Geometry.transform.(
                    Ref(Geometry.Contravariant12Axis()),
                    u_local,
                )

            div_us = sdiv.(u)
            Spaces.weighted_dss!(div_us)

            div_uw = wdiv.(u)
            Spaces.weighted_dss!(div_uw)

            div_exact = @. -2 * sind(coords.long) * cosd(coords.lat) / radius

            err_sdiv[Ie] = norm(div_us .- div_exact)
            err_wdiv[Ie] = norm(div_uw .- div_exact)
            Δh[Ie] = 1 / Ne

        end

        convergence_rate_sdiv = convergence_rate(err_sdiv, Δh)
        convergence_rate_wdiv = convergence_rate(err_wdiv, Δh)

        for Ie in range(1, length = length(Nes) - 1)
            @test convergence_rate_sdiv[Ie] ≈ (Nq - 1) atol = 0.1
            @test convergence_rate_wdiv[Ie] ≈ (Nq - 1) atol = 0.1
        end
    end
end
