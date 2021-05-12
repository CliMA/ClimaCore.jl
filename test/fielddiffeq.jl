using Test
using OrdinaryDiffEq
using ClimateMachineCore
import ClimateMachineCore: Domains, Topologies, Meshes, Fields
using LinearAlgebra


domain = Domains.RectangleDomain(
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 3, 4
Nq = 4
discretization = Domains.EquispacedRectangleDiscretization(domain, n1, n2)
grid_topology = Topologies.GridTopology(discretization)
quad = Meshes.Quadratures.GLL{Nq}()
mesh = Meshes.Mesh2D(grid_topology, quad)

# y = [cos(x1+x2+t), sin(x1+x2+t)]
# dy/dt = [-y[2], y[1]]

f(x, t) = (c = cos(x.x1 + x.x2 + t), s = sin(x.x1 + x.x2 + t))
function dfdt(y, _, t)
    broadcast(y -> (c = -y.s, s = y.c), y)
end
function dfdt!(dydt, y, _, t)
    broadcast!(y -> (c = -y.s, s = y.c), dydt, y)
end

y0 = f.(Fields.coordinate_field(mesh), 0.0)
y1 = f.(Fields.coordinate_field(mesh), 1.0)

# Solve the ODE operator
prob = ODEProblem(dfdt!, y0, (0.0, 1.0))
sol = solve(prob, Tsit5(), reltol = 1e-6)

@test norm(sol(1.0) .- y1) <= 1e-6 * norm(y1)
