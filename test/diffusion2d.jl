using Test
using StaticArrays, IntervalSets
import ClimaCore.DataLayouts: IJFH
import ClimaCore:
    Fields, Domains, Meshes, Topologies, Spaces, Operators, Geometry
using StaticArrays, IntervalSets, LinearAlgebra

using OrdinaryDiffEq

@testset "2D field dx/dt = ∇⋅∇  ODE solve" begin
    FT = Float64

    domain = Domains.RectangleDomain(
        FT(-π)..FT(π),
        FT(-π)..FT(π),
        x1periodic = true,
        x2periodic = true,
    )

    mesh = Meshes.EquispacedRectangleMesh(domain, 5, 5)
    grid_topology = Topologies.GridTopology(mesh)

    Nq = 6
    quad = Spaces.Quadratures.GLL{Nq}()
    points, weights = Spaces.Quadratures.quadrature_points(Float64, quad)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    # 2D field
    # dx/dt = ∇⋅∇ x

    # Standard heat equation:
    # ∂_t f(x1,x2) =  ∇⋅( α ∇ f(x1,x2) ) + g(x1,x2), α > 0

    # Advection Equation
    # ∂_t f + c ∂_x f  = 0
    # the solution translates to the right at speed c,
    # so if you you have a periodic domain of size [0, 1]
    # at time t, the solution is f(x - c * t, y)

    f(x, t) = sin(x.x1) * exp(-t)
    y0 = f.(Fields.coordinate_field(space), 0.0)

    function rhs!(dydt, y, _, t)

        ∇y = Operators.slab_gradient(y)
        dydt .= .-Operators.slab_weak_divergence(∇y)

        Spaces.horizontal_dss!(dydt)
        Spaces.variational_solve!(dydt)
    end

    # Solve the ODE operator
    prob = ODEProblem(rhs!, y0, (0.0, 1.0))
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

    # Reconstruct the result Field at the last timestep
    y1 = sol(1.0)

    @show y1 .- f.(Fields.coordinate_field(space), 1.0)

    @test y1 ≈ f.(Fields.coordinate_field(space), 1.0) rtol = 5e-5
end
