# Apply boundary conditions

Boundary conditions in ClimaCore attach to operators, not to fields. A
finite-difference operator takes one condition per boundary name of its
domain; a DG interface flux takes a horizontal boundary condition or a
one-sided flux. This page lists the forms and when each applies.

## Vertical boundaries (finite-difference operators)

The vertical domain names its two boundaries, `(:bottom, :top)` by convention,
and every `C2F` operator whose stencil reaches a boundary face takes a keyword
of that name.

| Condition                     | Meaning at the boundary face                                                       | Accepted by                                                                                                                                       |
|:----------------------------- |:---------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SetValue(x₀)`                | The operator's *argument* takes the value `x₀`                                     | `InterpolateC2F`, `DivergenceF2C` (a flux), and, through `DirichletOperator`, `GradientC2F`, `DivergenceC2F`, `CurlC2F`, `UpwindBiasedProductC2F` |
| `SetGradient(v₀)`             | The gradient equals the covariant vector `v₀`                                      | `GradientC2F`                                                                                                                                     |
| `SetDivergence(d₀)`           | The divergence equals `d₀`                                                         | `DivergenceC2F`                                                                                                                                   |
| `SetCurl(c₀)`                 | The curl equals the contravariant vector `c₀`                                      | `CurlC2F`                                                                                                                                         |
| `Extrapolate()` / `Outflow()` | Use the nearest interior value (`Outflow(; order)` for higher-order extrapolation) | `InterpolateC2F`, the upwind operators                                                                                                            |

Steps:

 1. Name the boundaries when constructing the vertical domain, `boundary_names = (:bottom, :top)`.

 2. Give a condition to the outermost operator whose stencil touches the
    boundary, and only to it; inner operators in a fused expression stay
    inside the boundary faces. In `divf2c(gradc2f(θ))`, the divergence takes
    `SetValue` fluxes and the gradient takes none:

    ```julia
    gradc2f = Operators.GradientC2F()
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(0.0)),
        top = Operators.SetValue(Geometry.WVector(0.0)),
    )
    @. dydt = divf2c(κ * gradc2f(θ))
    ```

 3. For a Dirichlet value of the argument itself, use the helper that builds
    the equivalent gradient stencil:
    `Operators.gradient_c2f_dirichlet(θ; bottom = θ₀, top = Operators.SetGradient(...))`
    ([Solve a column PDE](../tutorials/column_heat.md)).

A `SetGradient` prescribes the covariant component `∂x/∂ξ³`, the derivative
along the third coordinate line. Over terrain, the coordinate surface is tilted,
so a zero covariant component is a zero normal derivative only where the
surface is flat; the `GradientC2F` docstring gives the value that
prescribes a zero normal derivative on a slope.

## Horizontal boundaries

Spectral-element domains are periodic or walled per direction
(`periodic = true` on an `IntervalDomain`, or `boundary_names`). On a CG
space, the operators impose horizontal conditions through the state, for
example by projecting the velocity onto the wall. On a DG space, the interface
flux at a boundary face needs the exterior state:

  - `Operators.PeriodicBC()`: the domain wraps; no boundary faces exist.
  - `Operators.ReflectingWallBC()`: the exterior state mirrors the normal
    momentum, `Operators.ghost_state(bc, normal, argvals⁻)`, so the wall passes
    no normal flux.
  - A one-sided flux function `boundary_numflux(normal, argvals⁻)` passed to
    `tendency_completion`, for any other closure; [`examples/plane/bickleyjet.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/plane/bickleyjet.jl)
    builds a reflecting wall this way for its `noslip` case.

With the boundary flux omitted, DG boundary faces contribute a zero flux,
which is a closed-boundary closure.

## Sponges and damping

Absorbing layers near the model top are tendencies, not boundary conditions;
ClimaAtmos's [sponge page](https://clima.github.io/ClimaAtmos.jl/stable/sponge/)
describes its viscous and Rayleigh sponges, which are written with the
operators above.
