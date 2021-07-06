using Test
using OrdinaryDiffEq
using ClimaCore
import ClimaCore: Domains, Topologies, Meshes, Fields, Spaces
using LinearAlgebra, IntervalSets



domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 3, 4
Nq = 4
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

# y = [cos(x1+x2+t), sin(x1+x2+t)]
# dy/dt = [-y[2], y[1]]

f(x, t) = (c = cos(x.x1 + x.x2 + t), s = sin(x.x1 + x.x2 + t))
function dfdt(y, _, t)
    broadcast(y -> (c = -y.s, s = y.c), y)
end
function dfdt!(dydt, y, _, t)
    broadcast!(y -> (c = -y.s, s = y.c), dydt, y)
end

y0 = f.(Fields.coordinate_field(space), 0.0)
y1 = f.(Fields.coordinate_field(space), 1.0)

# Solve the ODE operator
prob = ODEProblem(dfdt!, y0, (0.0, 1.0))
sol = solve(prob, Tsit5(), reltol = 1e-6)

@test norm(sol(1.0) .- y1) <= 1e-6 * norm(y1)


# implicit solvers
# require use of FieldVector



domain = Domains.IntervalDomain(0.0, Float64(pi); x3boundary = (:left, :right))

mesh = Meshes.IntervalMesh(domain; nelems = 16)

center_space = Spaces.CenterFiniteDifferenceSpace(mesh)

# y(z, t) = z*exp(α*t)
# dydt(z, t) = α*y(t)
y = Fields.FieldVector(z = Fields.coordinate_field(center_space))

function f!(dydt, y, α, t)
    dydt.z .= α .* y.z
end

function f_jac!(J, y, α, t)
    copyto!(J, α * LinearAlgebra.I)
end


prob = ODEProblem(ODEFunction(f!), copy(y), (0.0, 1.0), 0.1)
sol = solve(prob, SSPRK22(), dt = 0.01)


sol = solve(prob, ImplicitEuler(), reltol = 1e-6)
y1 = similar(y)
y1 .= y .* exp(0.1)

@test norm(sol[end] .- y1) <= 1e-4 * norm(y1)
