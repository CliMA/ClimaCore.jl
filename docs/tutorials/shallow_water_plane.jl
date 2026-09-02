# # Tutorial: Shallow water on a plane
#
# This tutorial solves the shallow-water equations on a doubly periodic plane
# with continuous-Galerkin spectral elements: a Bickley jet that rolls up
# through barotropic instability. It shows a full spectral-element tendency
# with weak-form operators, direct stiffness summation, and fourth-order
# hyperdiffusion, and it uses a `NamedTuple`-valued field as the model state.
#
# The equations, in vector-invariant form with the velocity in covariant
# components `u_i`, are
# ```math
# \begin{align*}
# \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho u) &= 0, \\
# \frac{\partial u_i}{\partial t} + \bigl((\nabla \times u) \times u\bigr)_i &= -\nabla_i \left(g \rho + \tfrac12 \|u\|^2\right), \\
# \frac{\partial \rho\theta}{\partial t} + \nabla \cdot (\rho\theta\, u) &= 0,
# \end{align*}
# ```
# where `ρ` is the layer depth, `g` the gravitational acceleration, and `θ` a
# passive tracer. The momentum equation is the vector-invariant form of the
# advection term, `(∇ × u) × u + ∇(½‖u‖²)` in place of `(u ⋅ ∇) u`. The code
# adds fourth-order hyperdiffusion, `−D₄ ∇⁴` acting on `u` and `ρθ`, to remove
# grid-scale energy.

using ClimaComms
ClimaComms.@import_required_backends
using LinearAlgebra
using IntervalSets
import ClimaCore:
    Domains, Meshes, Topologies, Quadratures, Spaces, Fields, Geometry, Operators, Remapping
import ClimaTimeSteppers as CTS
using CairoMakie
import ClimaCore.Visualize: fieldheatmap
CairoMakie.activate!(type = "png")

# ## 1. The plane

domain = Domains.RectangleDomain(
    Geometry.XPoint(-2π) .. Geometry.XPoint(2π),
    Geometry.YPoint(-2π) .. Geometry.YPoint(2π),
    x1periodic = true,
    x2periodic = true,
)
mesh = Meshes.RectilinearMesh(domain, 16, 16)
context = ClimaComms.SingletonCommsContext(ClimaComms.device())
topology = Topologies.Topology2D(context, mesh)
space = Spaces.SpectralElementSpace2D(topology, Quadratures.GLL{4}())

# ## 2. The initial state
#
# A zonal jet `c sech²(y)` with a vortical perturbation from the streamfunction
# `Ψ′ = exp(−(y + l/10)²/2l²) cos(kx) cos(ky)`. The velocity is set in the
# orthonormal basis and converted to covariant components with the local
# geometry, which is the form the momentum equation is written in.

parameters = (;
    ϵ = 0.1,   # perturbation amplitude
    l = 0.5,   # Gaussian width
    k = 0.5,   # perturbation wavenumber
    ρ₀ = 1.0,  # reference depth
    c = 1.0,   # jet speed
    g = 10.0,  # gravity
    D₄ = 1e-4, # hyperdiffusion coefficient
)

function init_state(local_geometry, p)
    (; x, y) = local_geometry.coordinates
    U₁ = p.c / cosh(y)^2
    ϕ = exp(-(y + p.l / 10)^2 / (2 * p.l^2))
    u₁′ = ϕ * (y + p.l / 10) / p.l^2 * cos(p.k * x) * cos(p.k * y)
    u₁′ += p.k * ϕ * cos(p.k * x) * sin(p.k * y)
    u₂′ = -p.k * ϕ * sin(p.k * x) * cos(p.k * y)
    u = Geometry.Covariant12Vector(
        Geometry.UVVector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′),
        local_geometry,
    )
    return (; ρ = p.ρ₀, u = u, ρθ = p.ρ₀ * sin(p.k * y))
end

y0 = init_state.(Fields.local_geometry_field(space), Ref(parameters))
fieldheatmap(y0.ρθ)

# ## 3. The tendency
#
# Flux divergences use the weak form, gradients the strong form, so that the
# discrete mass and energy budgets telescope ([Spectral elements: CG and
# DG](../explanation/discretizations.md)). The fourth-order hyperdiffusion is
# two Laplacian passes with a DSS of the intermediate field between them; the
# whole tendency is completed by a final DSS. `@.` fuses each group of operators
# into one pass over the elements.

function shallow_water_tendency!(dydt, y, p, t)
    (; D₄, g) = p
    sdiv = Operators.Divergence()
    wdiv = Operators.Divergence{Operators.WeakForm}()
    grad = Operators.Gradient()
    wgrad = Operators.Gradient{Operators.WeakForm}()
    curl = Operators.Curl()
    wcurl = Operators.Curl{Operators.WeakForm}()

    ## first Laplacian pass of the hyperdiffusion
    @. dydt.u =
        wgrad(sdiv(y.u)) -
        Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))
    @. dydt.ρθ = wdiv(grad(y.ρθ))
    Spaces.weighted_dss!(dydt)

    ## second pass, then the dynamics
    @. dydt.u =
        -D₄ * (
            wgrad(sdiv(dydt.u)) - Geometry.Covariant12Vector(
                wcurl(Geometry.Covariant3Vector(curl(dydt.u))),
            )
        )
    @. dydt.ρθ = -D₄ * wdiv(grad(dydt.ρθ))
    @. begin
        dydt.ρ = -wdiv(y.ρ * y.u)
        dydt.u += -grad(g * y.ρ + norm(y.u)^2 / 2) + y.u × curl(y.u)
        dydt.ρθ += -wdiv(y.ρθ * y.u)
    end
    Spaces.weighted_dss!(dydt)
    return dydt
end

# ## 4. Time integration

prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = shallow_water_tendency!),
    y0,
    (0.0, 20.0),
    parameters,
)
sol = CTS.solve(
    prob,
    CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()),
    dt = 0.05,
    saveat = collect(0.0:5.0:20.0),
)

# ## 5. The roll-up
#
# The tracer marks the jet; the perturbation grows and wraps the tracer into
# vortices. Contour lines show the roll-up more sharply than a filled color
# scale. The spectral-element field is interpolated onto a regular grid with
# `Remapping.interpolate`, and the grid is drawn with `contour`. A
# 100 × 100 grid keeps the interpolation light for the documentation build.

xs = range(-2π, 2π, length = 100)
ys = range(-2π, 2π, length = 100)
target_hcoords = [Geometry.XYPoint(x, y) for x in xs, y in ys]
levels = range(-0.9, 0.9, length = 9)

fig = Figure(size = (900, 600))
for (i, (t, y)) in enumerate(zip(sol.t, sol.u))
    ax = Axis(fig[(i - 1) ÷ 3 + 1, (i - 1) % 3 + 1], title = "t = $t", aspect = 1)
    ρθ = Remapping.interpolate(y.ρθ; target_hcoords)
    contour!(ax, xs, ys, ρθ; levels, colormap = :balance, colorrange = (-1, 1))
    hidedecorations!(ax)
end
fig

# Mass is conserved to round-off by the weak divergence completed by DSS, and
# total energy drifts only through the hyperdiffusion. The integrator advances
# `y0` in place, so the comparison uses the snapshot saved at `t = 0`.

energy(y, p) = sum(@. y.ρ * norm(y.u)^2 / 2 + p.g * y.ρ^2 / 2)
y_start = sol.u[1]
(
    mass_drift = abs(sum(sol.u[end].ρ) - sum(y_start.ρ)) / sum(y_start.ρ),
    energy_drift = (energy(sol.u[end], parameters) - energy(y_start, parameters)) /
                   energy(y_start, parameters),
)

# The same case on a discontinuous-Galerkin space, with one tendency for both,
# is [Tutorial: CG and DG with one tendency](cg_dg_switch.md).
