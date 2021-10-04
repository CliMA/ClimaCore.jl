using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

using OrdinaryDiffEq

@testset "2D field Poisson problem - ∇⋅∇ = f" begin
    # Poisson equation
    # - ∇⋅(∇ u(x,y)) = f(x,y)

    # True solution (eigenfunction): u(x,y) = sin(c₁ + k₁ x) * sin(c₂ + k₂ y)
    # => - ∇⋅(∇ u(x,y)) = f(x,y) = (k₁^2 + k₂^2) * u(x,y)

    FT = Float64

    domain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(-π)..Geometry.XPoint{FT}(π),
        Geometry.YPoint{FT}(-π)..Geometry.YPoint{FT}(π),
        x1periodic = true,
        x2periodic = true,
    )

    mesh = Meshes.EquispacedRectangleMesh(domain, 3, 3)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    x = Fields.coordinate_field(space).x
    y = Fields.coordinate_field(space).y
    c₁ = 0.0
    k₁ = 1.0
    c₂ = 2.0
    k₂ = 3.0

    # Define eigensolution
    u(coord, c₁, c₂, k₁, k₂) = sin(c₁ + k₁ * coord.x) * sin(c₂ + k₂ * coord.y)
    true_sol = u.(Fields.coordinate_field(space), c₁, c₂, k₁, k₂)

    function diffusion(f)
        diff = zeros(eltype(f), space)
        ∇f = Operators.slab_gradient(f)
        diff .= Operators.slab_weak_divergence(∇f)

        Spaces.horizontal_dss!(diff)
        return diff
    end

    diff = diffusion(true_sol)

    @show (diff .- (k₁^2 + k₂^2) .* true_sol)
    @test norm(diff .- (k₁^2 + k₂^2) .* true_sol) ≤ 1e-2
end
