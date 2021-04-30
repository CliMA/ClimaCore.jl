using Test
using StaticArrays
import ClimateMachineCore.DataLayouts: IJFH
import ClimateMachineCore: Fields, Domains, Topologies, Meshes
import ClimateMachineCore.Operators
import ClimateMachineCore.Geometry
using LinearAlgebra

using DifferentialEquations

@testset "2D field dx/dt = ∇⋅∇  ODE solve" begin
    FT = Float64

    domain = Domains.RectangleDomain(
        x1min = FT(-π),
        x1max = FT(π),
        x2min = FT(-π),
        x2max = FT(π),
        x1periodic = true,
        x2periodic = true,
    )
    discretization = Domains.EquispacedRectangleDiscretization(domain, 5, 5)
    grid_topology = Topologies.GridTopology(discretization)

    Nq = 6
    quad = Meshes.Quadratures.GLL{Nq}()
    points, weights = Meshes.Quadratures.quadrature_points(Float64, quad)
    mesh = Meshes.Mesh2D(grid_topology, quad)

    # 2D field
    # dx/dt = ∇⋅∇ x

    # Standard heat equation:
    # ∂_t f(x1,x2) =  ∇⋅( α ∇ f(x1,x2) ) + g(x1,x2), α > 0

    # Advection Equation
    # ∂_t f + c ∂_x f  = 0
    # the solution translates to the right at speed c,
    # so if you you have a periodic domain of size [0, 1]
    # at time t, the solution is f(x - c * t, y)


    f0(x) = sin(x.x1)
    y0 = f0.(Fields.coordinate_field(mesh))

    function reconstruct(rawdata, field)
        D = typeof(Fields.field_values(field))
        Fields.Field(D(rawdata), Fields.mesh(field))
    end

    function rhs!(rawdydt, rawdata, field, t)
        # reconstuct Field objects
        y = reconstruct(rawdata, field)
        dydt = reconstruct(rawdydt, field)

        ∇y = Operators.slab_gradient(y)
        dydt .= Operators.slab_weak_divergence(∇y)

        # apply DSS
        dydt_data = Fields.field_values(dydt)
        WJ = copy(mesh.local_geometry.WJ) # quadrature weights * jacobian
        Operators.horizontal_dss!(WJ, mesh)
        dydt_data .*= mesh.local_geometry.WJ
        Operators.horizontal_dss!(dydt_data, mesh)
        dydt_data ./= WJ

        return rawdydt
    end

    # 1. make DifferentialEquations work on Fields: i think we need to extend RecursiveArrayTools
    #    - this doesn't seem like it will work directly: ideally we want a way to unwrap and wrap as required
    #
    # 2. Define isapprox on Fields
    # 3. weighted DSS

    # Solve the ODE operator
    prob = ODEProblem(rhs!, parent(Fields.field_values(y0)), (0.0, 1.0), y0)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

    # Reconstruct the result Field at the last timestep
    y1 = reconstruct(sol.u[end], y0)

    @test parent(Fields.field_values(y0)) .* exp(-1) ≈
          parent(Fields.field_values(y1)) rtol = 1e-6
end
