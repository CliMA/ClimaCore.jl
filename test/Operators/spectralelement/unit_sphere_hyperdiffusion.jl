using Test
using ClimaComms
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
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

include("sphere_sphericalharmonics.jl")

@testset "Scalar biharmonic equation in 2D sphere" begin
    # ∇⁴ u(θ,φ) = f(θ,φ)
    # Please note that (θ,φ) is colatitude and azimuth as in the spherical coordinate system, that
    # correspond to π/2-latitude and longitude on Earth
    # True solution (eigenfunction): u(θ,φ) = Yₗᵐ(θ,φ)
    # => ∇⁴ u(θ,φ) = f(θ,φ) = l^2*(l+1)^2/(radius)^4 * u(θ,φ)

    FT = Float64

    radius = FT(1)
    domain = Domains.SphereDomain(radius)

    Ne = 16
    mesh = Meshes.EquiangularCubedSphere(domain, Ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )

    Nq = 6
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    coords = Fields.coordinate_field(space)

    l = 7
    m = 4

    # Define eigensolution
    u = @. Ylm(l, m, coords.lat, coords.long)

    function ∇⁴(u)
        grad = Operators.Gradient()
        wdiv = Operators.WeakDivergence()
        diff = @. wdiv(grad(u))
        Spaces.weighted_dss!(diff)

        hyperdiff = @. wdiv(grad(diff))
        Spaces.weighted_dss!(hyperdiff)
        return hyperdiff
    end

    # Call the diffusion operator
    hyperdiff = ∇⁴(u)

    # compute the exact solution
    exact_solution = @. l^2 * (l + 1)^2 / (radius)^4 * u

    @test hyperdiff ≈ exact_solution rtol = 2e-2

end
