using Test
using StaticArrays, IntervalSets
using ClimaComms
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

@testset "Scalar Poisson problem - ∇⋅∇ = f on the cubed-sphere" begin
    # Poisson equation on a sphere
    # - ∇⋅(∇ u(ϕ, λ)) = f(ϕ, λ)

    # With:
    # ϕ = asin(z / R) # latitude (R radius of sphere)
    # λ = atan2(y, x) # longitude ∈ [-π, π]
    # True solution (eigenfunction): u(ϕ, λ) = sin(λ) * cos(ϕ)
    # => - ∇⋅(∇ u(ϕ, λ)) = f(ϕ, λ) = (2/R²) * u(ϕ, λ)

    FT = Float64

    R = FT(3) # radius
    ne = 8
    Nq = 4
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
        mesh,
    )
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # Define eigensolution
    u = map(Fields.coordinate_field(space)) do coord
        sind(coord.long) * cosd(coord.lat)
    end

    function laplacian_u(space, R)
        coords = Fields.coordinate_field(space)
        laplacian_u = map(coords) do coord
            (2 / R^2) * sind(coord.long) * cosd(coord.lat)
        end

        return laplacian_u
    end

    function diffusion(u)
        grad = Operators.Gradient()
        wdiv = Operators.WeakDivergence()
        diff = @. -wdiv(grad(u))
        Spaces.weighted_dss!(diff)
        return diff
    end

    # Call the diffusion operator
    diff = diffusion(u)

    exact_solution = laplacian_u(space, R)

    @test norm(diff .- exact_solution) ≤ 1e-3
end
