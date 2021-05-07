using DifferentialEquations
using ClimateMachineCore
import ClimateMachineCore: Domains, Topologies, Meshes, Fields


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

# y = [cos(x*t), sin(x*t)]
# dy/dt = [-sin(x*t), cos(x*t)] = [-y[2], y[1]]

f(x,t) = (c=cos(x.x1*t), s=sin(x.x1*t))

function dfdt(y, _, t)
  broadcast(y -> (c=-y.s, s=y.c), y)
end
function dfdt!(dydt, y, _, t)
    broadcast!(y -> (c=-y.s, s=y.c), dydt, y)
end

y0 = f.(Fields.coordinate_field(mesh), 0.0)

@noinline function OrdinaryDiffEq.calculate_residuals!(out::Fields.Field, u₀::Fields.Field, u₁::Fields.Field, α, ρ, internalnorm,t)
    out .= (u₁ .- u₀) ./ (α + max(internalnorm(u₀,t), internalnorm(u₁,t)) * ρ)
end
@noinline function OrdinaryDiffEq.calculate_residuals!(out::Fields.Field, ũ::Fields.Field, u₀::Fields.Field, u₁::Fields.Field, α, ρ, internalnorm,t)
  out .= ũ ./ (α + max(internalnorm(u₀,t), internalnorm(u₁,t)) * ρ)
end

# Solve the ODE operator
prob = ODEProblem(dfdt!, y0, (0.0, 1.0))
sol = solve(prob, Tsit5())


