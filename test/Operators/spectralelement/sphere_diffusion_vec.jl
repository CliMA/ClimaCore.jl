using Test
using ClimaComms
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry, Quadratures
using StaticArrays, IntervalSets, LinearAlgebra

include("sphere_sphericalharmonics.jl")

# diffusion on vector: input covariant vecotor
function ∇²(u)
    scurl = Operators.Curl()
    sdiv = Operators.Divergence()
    wcurl = Operators.WeakCurl()
    wgrad = Operators.WeakGradient()

    χ = Spaces.weighted_dss!(
        @. wgrad(sdiv(u)) - Geometry.Covariant12Vector(
            wcurl(Geometry.Covariant3Vector(scurl(u))),
        )
    )
    return χ
end

@testset "diffusion on vectors" begin
    FT = Float64

    l = Int(7)
    m = Int(4)

    Ne = 7
    Nq = 6

    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )

    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    # compute vector spherical harmonics (VSH) as the gradient of scalar spherical harmonics
    # eigenfunction properties of VSH: ∇² VSH(l,m) = -l(l+1) VSH(l,m)
    VSH_local = map(coords) do coord
        Geometry.UVVector(VSH(l, m, coord.lat, coord.long)...)
    end
    VSH_cov = Geometry.transform.(Ref(Geometry.Covariant12Axis()), VSH_local)

    # diffusion
    ∇²VSH_cov = ∇²(VSH_cov)
    ∇²VSH_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇²VSH_cov)

    # exact solution
    ∇²VSH_exact = @. -l * (l + 1) * VSH_local

    @test ∇²VSH_local ≈ ∇²VSH_exact rtol = 2e-2
end

convergence_rate(err, Δh) =
    [log(err[i] / err[i - 1]) / log(Δh[i] / Δh[i - 1]) for i in 2:length(Δh)]

@testset "convergence tests for vector diffusions on the sphere" begin
    FT = Float64

    Nes = [4, 8, 16]
    Nqs = [3, 4, 5, 6]

    l = Int(7)
    m = Int(4)

    radius = FT(1)
    domain = Domains.SphereDomain(radius)
    for (Iq, Nq) in enumerate(Nqs)
        err_diffusion = Array{FT}(undef, length(Nes))
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

            VSH_local = map(coords) do coord
                Geometry.UVVector(VSH(l, m, coord.lat, coord.long)...)
            end
            VSH_cov =
                Geometry.transform.(Ref(Geometry.Covariant12Axis()), VSH_local)
            ∇²VSH_cov = ∇²(VSH_cov)
            ∇²VSH_local = Geometry.transform.(Ref(Geometry.UVAxis()), ∇²VSH_cov)
            ∇²VSH_exact = @. -l * (l + 1) * VSH_local

            err_diffusion[Ie] = norm(∇²VSH_local .- ∇²VSH_exact)
            Δh[Ie] = 1 / Ne
        end

        convergence_rate_∇² = convergence_rate(err_diffusion, Δh)
        for Ie in range(1, length = length(Nes) - 1)
            @test convergence_rate_∇²[Ie] > (Nq - 2)
        end
    end
end
