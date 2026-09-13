# Choose CG or DG

The horizontal spectral-element grid is continuous Galerkin (CG) or
discontinuous Galerkin (DG). The choice is made once, when the grid is
constructed, and a model written with `Operators.tendency_completion` runs on
either. This page is the decision guide; [Spectral elements: CG and
DG](../explanation/discretizations.md) has the theory.

## Set the discretization

Pass `discretization = Grids.DG()` (or `Grids.CG()`, the default) to any
constructor that builds a horizontal spectral-element grid:

```julia
space = Spaces.SpectralElementSpace2D(topology, quad; discretization = Grids.DG())
space = CubedSphereSpace(; radius, h_elem, n_quad_points, discretization = Grids.DG())
```

Read it back with `Spaces.discretization(space)`; `Spaces.is_continuous(space)`
is the Boolean form. An omitted keyword follows the quadrature: Gauss–Lobatto–
Legendre nodes give `CG()`, Gauss–Legendre nodes give `DG()`.

## Write the tendency once

Build the completion from the tendency field at setup and pass the interface
flux unconditionally; it is used on DG and ignored on CG:

```julia
numflux = Operators.RusanovNumericalFlux(physical_flux, wavespeed)
completion = Operators.tendency_completion(dydt; numflux)

function rhs!(dydt, y, (params, completion), t)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    @. dydt = -wdiv(physical_flux(y, Ref(params)))
    Operators.complete_tendency!(completion, dydt, y, params)
    return dydt
end
```

The [CG and DG tutorial](../tutorials/cg_dg_switch.md) runs this on both
spaces.

## Decide

| Question                               | CG                                                                                                                 | DG                                                                                                                                                                            |
|:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------ |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How is grid-scale energy removed?      | Explicit fourth-order hyperdiffusion, tuned per resolution                                                         | The interface-flux penalty (Rusanov, Roe); no hyperdiffusion needed                                                                                                           |
| What does inter-element coupling cost? | One DSS per completed tendency; the only horizontal communication                                                  | One numerical-flux evaluation per face per completed tendency, plus a halo exchange                                                                                           |
| Is the scheme conservative?            | Yes, to round-off, through the inner-product-preserving DSS                                                        | Yes, to round-off, through antisymmetric interface fluxes                                                                                                                     |
| Which operators exist today?           | All of them: strong and weak `Gradient`, `Divergence`, `Curl`; scalar and vector Laplacians; limiters; hypsography | The same element-local operators and the scalar Laplacian (with interior-penalty face terms), plus the flux-differencing divergence and face lifting; no vector Laplacian yet |
| Can the tendency be a `FieldVector`?   | Yes, completed by one batched DSS                                                                                  | No; the state must be one field with a composite element type                                                                                                                 |
| Horizontal boundary conditions?        | Periodic, or imposed by the model on the state; the operators take none                                            | `PeriodicBC`, `ReflectingWallBC`, or a one-sided `boundary_numflux`                                                                                                           |

## Check the coupling

On a periodic domain, the integral of a conservative tendency vanishes on both
grids; a nonzero value means the completion was skipped or the interface flux
lacks antisymmetry.

```julia
abs(sum(dydt.ρ)) < 1e-12 * sum(abs, dydt.ρ)
```
