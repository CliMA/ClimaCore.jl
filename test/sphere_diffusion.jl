using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
import ClimaCore.Meshes: EquiangularSphereWarp, Mesh2D
using StaticArrays, IntervalSets, LinearAlgebra

using OrdinaryDiffEq

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
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(R)
    mesh = Mesh2D(domain, EquiangularSphereWarp(), ne)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # Define eigensolution
    u = map(Fields.coordinate_field(space)) do coord
        sin(coord.long) * cos(coord.lat)
    end

    function laplacian_u(space, R)
        coords = Fields.coordinate_field(space)
        laplacian_u = map(coords) do coord
            (2 / R^2) * sin(coord.long) * cos(coord.lat)
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

    @show norm(diff .- exact_solution)
    # @test norm(diff .- exact_solution) ≤ 5e-3
end
