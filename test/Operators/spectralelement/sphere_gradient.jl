using Test
using ClimaComms
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJHF
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

grad = Operators.Gradient()
wgrad = Operators.WeakGradient()

@testset "gradient" begin
    Ne = 5
    Nq = 6

    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )

    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    η = @. sind(coords.long) * cosd(coords.lat)
    ∇η = grad.(η)
    ∇η_w = wgrad.(η)
    ∇η_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇η)
    ∇ηw_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇η_w)
    Spaces.weighted_dss!(∇η_local)
    Spaces.weighted_dss!(∇ηw_local)

    ∇η_exact = @. Geometry.UVVector(
        cosd(coords.long) / radius,
        -sind(coords.long) * sind(coords.lat) / radius,
    )

    @test ∇η_local ≈ ∇η_exact rtol = 1e-2
    @test ∇ηw_local ≈ ∇η_exact rtol = 1e-2

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

@testset "convergence tests for the gradient operator on the sphere" begin
    Nes = [3, 9, 27]
    Nqs = [4, 6]

    for (Iq, Nq) in enumerate(Nqs)
        err_sgrad = Array{FT}(undef, length(Nes))
        err_wgrad = Array{FT}(undef, length(Nes))
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

            η = @. sind(coords.long) * cosd(coords.lat)
            ∇η = grad.(η)
            ∇η_w = wgrad.(η)
            ∇η_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇η)
            ∇ηw_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇η_w)
            Spaces.weighted_dss!(∇η_local)
            Spaces.weighted_dss!(∇ηw_local)

            ∇η_exact = @. Geometry.UVVector(
                cosd(coords.long) / radius,
                -sind(coords.long) * sind(coords.lat) / radius,
            )

            err_sgrad[Ie] = norm(∇η_local .- ∇η_exact)
            err_wgrad[Ie] = norm(∇ηw_local .- ∇η_exact)
            Δh[Ie] = 1 / Ne

        end

        convergence_rate_sgrad = convergence_rate(err_sgrad, Δh)
        convergence_rate_wgrad = convergence_rate(err_wgrad, Δh)

        for Ie in range(1, length = length(Nes) - 1)
            @test convergence_rate_sgrad[Ie] ≈ (Nq - 1) atol = 0.1
            @test convergence_rate_wgrad[Ie] ≈ (Nq - 1) atol = 0.1
        end

    end

end
