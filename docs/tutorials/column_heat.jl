# # Tutorial: Solve a column PDE
#
# This tutorial solves the heat equation on a vertical column with the
# finite-difference operators and advances it in time with ClimaTimeSteppers.
# It shows how a boundary value is imposed through an operator's boundary
# condition and how a ClimaCore field serves as the state of a time stepper.
#
# The equation is
# ```math
# \frac{\partial y}{\partial t} = \alpha\, \nabla \cdot \nabla y,
# ```
# with `y(0) = 1` at the bottom (Dirichlet) and `∂y/∂z(10) = 0` at the top
# (Neumann), starting from `y = 0`.

using ClimaComms
ClimaComms.@import_required_backends
using IntervalSets
import ClimaCore: Domains, Meshes, Spaces, Fields, Geometry, Operators
import ClimaTimeSteppers as CTS
using CairoMakie
CairoMakie.activate!(type = "png")

# ## 1. The column

domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0) .. Geometry.ZPoint(10.0),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain, nelems = 32)
device = ClimaComms.device()
center_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
z = Fields.coordinate_field(center_space).z
y0 = zeros(center_space)

# ## 2. The tendency
#
# The state lives on cell centers. There is no node at the bottom boundary, so
# the Dirichlet condition enters through the gradient at the first face:
# `Operators.gradient_c2f_dirichlet` builds the `SetGradient` boundary stencil
# that a prescribed boundary value implies, `∂y/∂ξ³[½] = 2 (y[1] − 1)` for a
# value of 1. The Neumann condition at the top is a `SetGradient` with a zero
# covariant component. The face-to-center divergence then needs no boundary
# conditions of its own.

function heat_tendency!(dydt, y, α, t)
    ∇y = Operators.gradient_c2f_dirichlet(
        y;
        bottom = 1.0,
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    divf2c = Operators.DivergenceF2C()
    @. dydt = α * divf2c(∇y)
    return dydt
end

# ## 3. Time integration
#
# A ClimaCore field is a valid state for ClimaTimeSteppers: the tendency is
# the explicit part of a `ClimaODEFunction`, and the third-order
# strong-stability-preserving Runge–Kutta scheme advances it.

α = 0.1
prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = heat_tendency!),
    y0,
    (0.0, 5.0),
    α,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = 0.1,
    saveat = collect(0.0:1.0:5.0),
)

# ## 4. The result
#
# Heat enters from the bottom boundary and diffuses upward; the top boundary
# passes no flux.

column_values(f) = vec(parent(f))
fig = Figure(size = (450, 450))
ax = Axis(fig[1, 1], xlabel = "y", ylabel = "z", title = "Heat equation, α = $α")
for (t, y) in zip(sol.t, sol.u)
    lines!(ax, column_values(y), column_values(z), label = "t = $t")
end
axislegend(ax, position = :rt)
fig

# The bottom value of the solution approaches 1 as the boundary condition
# demands; the extrapolated value at the first face is exactly 1 by
# construction of the Dirichlet stencil.

yend = column_values(sol.u[end])
(bottom_center = yend[1], top_center = yend[end])

# A vertical diffusion this stiff is usually solved implicitly. The
# [cubed-sphere tutorial](extruded_sphere.md) does that with a banded Jacobian
# from `MatrixFields`.
