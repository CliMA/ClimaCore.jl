using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

using OrdinaryDiffEq
using SphericalHarmonics

@testset "Scalar biharmonic equation in 2D sphere" begin
    # - ∇⁴ u(θ,φ) = f(θ,φ)
    # Please note that (θ,φ) is colatitude and azimuth as in the spherical coordinate system, that
    # correspond to π/2-latitude and longitude on Earth
    # True solution (eigenfunction): u(θ,φ) = Yₗᵐ(θ,φ)
    # => - ∇⁴ u(θ,φ) = f(θ,φ) = -l^2*(l+1)^2/(radius)^4 * u(θ,φ)

    FT = Float64

    radius = FT(6.37122e6)
    domain = Domains.SphereDomain(radius)

    Ne = 16
    mesh = Meshes.Mesh2D(domain, Meshes.EquiangularSphereWarp(), Ne)
    grid_topology = Topologies.Grid2DTopology(mesh)

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # Define eigensolution
    u = map(Fields.coordinate_field(space)) do coord
        l = 7
        m = 4
        real(
            computeYlm(deg2rad(90 - coord.lat), deg2rad(coord.long), lmax = l)[(l, m)],
        )
    end


    function ∇⁴u(space)
        coords = Fields.coordinate_field(space)
        ∇⁴u = map(coords) do coord
            l = 7
            m = 4
            -l^2 * (l + 1)^2 / (radius)^4 * real(
                computeYlm(
                    deg2rad(90 - coord.lat),
                    deg2rad(coord.long),
                    lmax = l,
                )[(l, m)],
            )
        end

        return ∇⁴u
    end

    function hyperdiffusion(u)
        grad = Operators.Gradient()
        wdiv = Operators.WeakDivergence()
        diff = @. -wdiv(grad(u))
        Spaces.weighted_dss!(diff)

        hyperdiff = @. wdiv(grad(diff))
        Spaces.weighted_dss!(hyperdiff)
        return hyperdiff
    end

    # Call the diffusion operator
    hyperdiff = hyperdiffusion(u)

    # compute the exact solution
    exact_solution = ∇⁴u(space)

    @test hyperdiff ≈ exact_solution rtol = 2e-2

end
