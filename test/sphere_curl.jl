using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

FT = Float64

radius = FT(1)
domain = Domains.SphereDomain(radius)

scurl = Operators.Curl()
wcurl = Operators.WeakCurl()

@testset "curl" begin
    Ne = 7
    Nq = 6

    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(mesh)

    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    u_local = @. Geometry.UVVector(
        sind(coords.long) * sind(coords.lat),
        cosd(coords.long),
    )

    u = Geometry.transform.(Ref(Geometry.Covariant12Axis()), u_local)

    curl_us = scurl.(u)
    Spaces.weighted_dss!(curl_us)

    curl_uw = wcurl.(u)
    Spaces.weighted_dss!(curl_uw)

    curl_exact = @. Geometry.Contravariant3Vector(
        -2 * sind(coords.long) * cosd(coords.lat) / radius,
    )


    @test curl_us ≈ curl_exact rtol = 1e-2
    @test curl_uw ≈ curl_exact rtol = 1e-2

end

"""
    convergence_rate(err, Δh)
Estimate convergence rate given vectors `err` and `Δh`
    err = C Δh^p+ H.O.T
    err_k ≈ C Δh_k^p
    err_k/err_m ≈ Δh_k^p/Δh_m^p
    log(err_k/err_m) ≈ log((Δh_k/Δh_m)^p)
    log(err_k/err_m) ≈ p*log(Δh_k/Δh_m)
    log(err_k/err_m)/log(Δh_k/Δh_m) ≈ p
"""
convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "convergence tests for the curl operator on the sphere" begin
    Nes = [3, 9, 27]
    Nqs = [4, 6]

    for (Iq, Nq) in enumerate(Nqs)
        err_scurl = Array{FT}(undef, length(Nes))
        err_wcurl = Array{FT}(undef, length(Nes))
        Δh = Array{FT}(undef, length(Nes))

        for (Ie, Ne) in enumerate(Nes)
            mesh = Meshes.EquiangularCubedSphere(domain, Ne)
            grid_topology = Topologies.Topology2D(mesh)

            quad = Spaces.Quadratures.GLL{Nq}()
            space = Spaces.SpectralElementSpace2D(grid_topology, quad)
            coords = Fields.coordinate_field(space)

            u_local = @. Geometry.UVVector(
                sind(coords.long) * sind(coords.lat),
                cosd(coords.long),
            )

            u = Geometry.transform.(Ref(Geometry.Covariant12Axis()), u_local)

            curl_us = scurl.(u)
            Spaces.weighted_dss!(curl_us)

            curl_uw = wcurl.(u)
            Spaces.weighted_dss!(curl_uw)

            curl_exact = @. Geometry.Contravariant3Vector(
                -2 * sind(coords.long) * cosd(coords.lat) / radius,
            )

            err_scurl[Ie] = norm(curl_us .- curl_exact)
            err_wcurl[Ie] = norm(curl_uw .- curl_exact)
            Δh[Ie] = 1 / Ne

        end

        convergence_rate_scurl = convergence_rate(err_scurl, Δh)
        convergence_rate_wcurl = convergence_rate(err_scurl, Δh)
        for Ie in range(1, length = length(Nes) - 1)
            @test convergence_rate_scurl[Ie] ≈ (Nq - 1) atol = 0.1
            @test convergence_rate_wcurl[Ie] ≈ (Nq - 1) atol = 0.1
        end
    end
end
