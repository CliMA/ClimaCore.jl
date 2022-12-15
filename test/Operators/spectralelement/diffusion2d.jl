using Test
using ClimaComms
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

using OrdinaryDiffEq

@testset "Scalar Poisson problem - ∇⋅∇ = f in 2D" begin
    # Poisson equation
    # - ∇⋅(∇ u(x,y)) = f(x,y)

    # True solution (eigenfunction): u(x,y) = sin(c₁ + k₁ x) * sin(c₂ + k₂ y)
    # => - ∇⋅(∇ u(x,y)) = f(x,y) = (k₁^2 + k₂^2) * u(x,y)

    FT = Float64

    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-π) .. Geometry.XPoint{FT}(π),
        Geometry.YPoint{FT}(-π) .. Geometry.YPoint{FT}(π),
        x1periodic = true,
        x2periodic = true,
    )

    mesh = Meshes.RectilinearMesh(domain, 10, 10)
    grid_topology = Topologies.Topology2D(
        ClimaComms.SingletonCommsContext(),
        mesh,
    )

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # Define eigensolution
    u = map(Fields.coordinate_field(space)) do coord
        c₁ = 0.0
        k₁ = 1.0
        c₂ = 2.0
        k₂ = 3.0
        sin(c₁ + k₁ * coord.x) * sin(c₂ + k₂ * coord.y)
    end

    function laplacian_u(space)
        coords = Fields.coordinate_field(space)
        laplacian_u = map(coords) do coord
            c₁ = 0.0
            k₁ = 1.0
            c₂ = 2.0
            k₂ = 3.0
            (k₁^2 + k₂^2) * sin(c₁ + k₁ * coord.x) * sin(c₂ + k₂ * coord.y)
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

    exact_solution = laplacian_u(space)

    @test norm(diff .- exact_solution) ≤ 5e-3
end
