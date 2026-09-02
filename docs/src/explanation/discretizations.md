# Spectral elements: continuous and discontinuous Galerkin

ClimaCore discretizes the horizontal directions with spectral elements. The
domain is partitioned into quadrilateral elements, and within each element a
field is a polynomial represented by its values at quadrature nodes. Two
variants differ only in how neighboring elements are coupled: the continuous
Galerkin (CG) method used by the CliMA atmosphere [Yatunin2026](@cite) and the
discontinuous Galerkin (DG) method of [Souza2023](@cite). This page develops
the shared element-local machinery, then each coupling, then what ClimaCore
implements of each.

The two methods are analogues of two classical families. CG behaves like a
high-order finite-difference method: one value per physical node, global
continuity, and derivatives that are exact for polynomials up to the element
degree. DG behaves like a high-order finite-volume method: element-local
values, jumps at interfaces, and coupling through fluxes with an upwind bias
that supplies the dissipation the scheme needs.

## The nodal basis

Each element is the image of the reference square `[-1, 1]²` under a
coordinate map `x = r(ξ¹, ξ²)`. On the reference square, `Nq`
Gauss–Lobatto–Legendre (GLL) points `ζ_n` in each direction carry the field,
and a field is the tensor-product Lagrange interpolant through them
[Karniadakis05a, Deville02a](@cite):

```math
\psi(\xi^1, \xi^2) = \sum_{n_1, n_2 = 1}^{N_q} L_{n_1}(\xi^1)\, L_{n_2}(\xi^2)\, \psi_{n_1 n_2},
```

with `L_n` the Lagrange polynomial that is one at `ζ_n` and zero at the other
nodes. Polynomials of degree `Nq - 1` are represented exactly. The GLL nodes
include the endpoints `±1`, so nodes on element boundaries are shared between
neighbors; Gauss–Legendre nodes (`Quadratures.GL`) lie strictly inside the
element, and a grid on them is discontinuous.

Integrals are approximated by the same nodes and their weights `w_n`,

```math
\int_{\Omega_e} \psi\, dV \approx \sum_{n_1, n_2} w_{n_1} w_{n_2}\, J_{n_1 n_2}\, \psi_{n_1 n_2},
```

where `J` is the Jacobian determinant of the coordinate map at the node. The
product `WJ = w_{n_1} w_{n_2} J` is the discrete volume of a node and appears
in every conservation statement below; it is stored in the grid's local
geometry.

Derivatives within an element are derivatives of the interpolant,

```math
\left. \frac{\partial \psi}{\partial \xi^1} \right|_{\zeta_{m_1}, \zeta_{m_2}}
  = \sum_{n_1} D_{m_1 n_1}\, \psi_{n_1 m_2},
```

with `D` the differentiation matrix of the Lagrange basis, followed by the
metric terms of the map to obtain physical-space derivatives ([Hybrid grids
and generalized coordinates](geometry.md)). Because a derivative can raise the
degree beyond what the nodes represent, products of fields are represented
only approximately; this aliasing is the reason the two methods below differ
in stability.

## Strong and weak forms

Every horizontal derivative comes in two forms. The **strong** form
differentiates the interpolant directly, as above. The **weak** form is
defined by the discrete integration by parts: the weak derivative `∂̃_i` of
`ψ` is the unique nodal field satisfying

```math
\langle \phi, \widetilde{\partial}_i \psi \rangle + \langle \psi, \partial_i \phi \rangle = 0
\quad \text{for every nodal field } \phi,
```

where `⟨·,·⟩` is the quadrature inner product with weights `WJ`. The weak
derivative is the negative adjoint of the strong one. Both have the same cost
and accuracy and satisfy the vector identities `∇ × ∇ = 0` and
`∇ ⋅ (∇ ×) = 0` [Taylor2010](@cite).

Which form to use is decided by conservation. A term conserves its integral
when the two operators acting in it are adjoint to each other, so that summing
over the domain telescopes into boundary terms. A flux divergence therefore
uses the weak divergence, `Divergence{WeakForm}`, so that its integral over
the domain reduces to boundary fluxes (a discrete divergence theorem). A
gradient in a momentum equation uses the strong gradient, `Gradient()`, so
that it pairs with the weak divergence of the mass flux and kinetic energy is
conserved [Taylor20j](@cite). A Laplacian is a weak divergence of a strong
gradient, `Divergence{WeakForm}(Gradient(ψ))`, which is the composition
`Operators.scalar_laplacian` provides. Which atmospheric term uses which form
is tabulated in ClimaAtmos's
[discretization page](https://clima.github.io/ClimaAtmos.jl/stable/discretization/).

In ClimaCore, the form is a type parameter: `Operators.Divergence()` is the
strong form and `Operators.Divergence{Operators.WeakForm}()` the weak form,
and likewise for `Gradient` and `Curl`. Within an element, the weak form of a
flux divergence is

```math
(\widetilde{\nabla} \cdot F)_{m}
  = -\frac{1}{WJ_m} \sum_{n} D_{n m}\, w_n\, (J F^{i})_{n} \quad \text{(summed over directions } i),
```

which is why weak-form results are incomplete at element-boundary nodes: the
sum runs over the element's own nodes only, and the neighbor's contribution to
a shared node is missing until the completion step.

## Continuous Galerkin: direct stiffness summation

In the CG method, a field is single-valued at every physical node, including
those on element boundaries, and the function space is continuous. The
completion step is direct stiffness summation (DSS), `Spaces.weighted_dss!`:
each element-boundary node value is replaced by the `WJ`-weighted average of
its copies in the adjacent elements [Deville02a, Taylor2010](@cite). DSS is a
projection `𝒫` onto the continuous space that preserves the inner product,
`⟨φ, 𝒫ψ⟩ = ⟨𝒫φ, ψ⟩`, so the summation-by-parts identity of the weak operators
survives it, and the CG model conserves mass, energy, and tracers to round-off
without fixers [Yatunin2026](@cite). DSS is the only operation that
communicates horizontally between elements.

CG relies on explicit dissipation. Grid-scale energy produced by aliasing is
removed by fourth-order hyperdiffusion, `∇⁴ψ`, built from two
applications of `Operators.scalar_laplacian` (or `vector_laplacian`) with a
DSS of the intermediate field between them [Lauritzen18a, Ullrich18a](@cite).
The coefficients and their scaling with resolution are model choices;
ClimaAtmos documents its
[hyperdiffusion](https://clima.github.io/ClimaAtmos.jl/stable/hyperdiffusion/).

## Discontinuous Galerkin: flux differencing and interface fluxes

In the DG method, a node on an element boundary carries one value per element,
and the two values may differ. The completion step adds the surface term of
the weak form. For a conservation law `∂ₜy + ∇⋅F(y) = 0`, the weak form of
the divergence in an element is

```math
\langle \phi, \widetilde{\nabla} \cdot F \rangle_e
  = -\langle \nabla \phi, F \rangle_e + \oint_{\partial \Omega_e} \phi\, F^*(y^-, y^+; \widehat n)\, dS ,
```

where `y⁻` is the state inside the element, `y⁺` the state in the neighbor,
`n̂` the outward normal, and `F*` a *numerical flux*, a single-valued function
of the two face states that replaces the two-valued physical flux
[Hesthaven07a](@cite). The element-local weak operator above supplies the
first term; the completion adds the second, `F*` weighted by the face
Jacobian, at every boundary node. (In strong form the same surface term
reads `F* − F(y⁻)⋅n̂`, the difference between the numerical and the element's
own flux.) The numerical flux is antisymmetric in its two sides, so the sum
over all faces telescopes and the scheme is globally conservative; its
dissipative part acts on the jump `y⁺ − y⁻` and vanishes where the solution
is continuous. [DSS and numerical fluxes](interelement.md) describes the
provided fluxes.

Souza et al. [Souza2023](@cite) add a second ingredient for the volume term.
Aliasing lets the discrete kinetic energy equation acquire a spurious source
even though the continuous one has none, and the fix is to write the volume
derivative in *flux-differencing* (split) form: the differentiation matrix
acts on a symmetric two-point flux `F♯(y_m, y_n)` between pairs of nodes,

```math
(\nabla \cdot F)_m \approx 2 \sum_n D_{m n}\, F^\sharp(y_m, y_n) \cdot \widehat n_{m n},
```

rather than on the pointwise flux `F(y_n)`. Different choices of `F♯` are
algebraically equivalent rewritings of the same equation with different
discrete properties [Gassner16a, Chan18a](@cite). With the Kennedy–Gruber
two-point flux [Kennedy08a](@cite), which uses arithmetic averages
`{ρ}{u}` of the two nodes' variables, the discrete kinetic energy equation
mimics the continuous one (the scheme is kinetic-energy preserving), and with
a matching treatment of the gravity term it preserves kinetic plus potential
energy [Souza2023](@cite). The paper's Held–Suarez and boundary-layer
simulations, run with ClimateMachine, the predecessor of ClimaCore
[Sridhar22a](@cite), use no filter, hyperdiffusion, or sponge; the interface
fluxes (a Rusanov or Roe penalty on a central flux) supply all dissipation.

## What ClimaCore implements

| Capability                              | CG (`Grids.CG()`)                                                        | DG (`Grids.DG()`)                                                                |
|:--------------------------------------- |:------------------------------------------------------------------------ |:-------------------------------------------------------------------------------- |
| Element-local strong and weak operators | `Gradient`, `Divergence`, `Curl`, each with `{WeakForm}`                 | Same operators, same code                                                        |
| Completion of a weak-form tendency      | `Spaces.weighted_dss!`                                                   | `Operators.add_numerical_flux_interior!` and `add_numerical_flux_boundary!`      |
| Model-level switch                      | `Operators.tendency_completion` → `DSSCompletion`                        | `Operators.tendency_completion` → `NumericalFluxCompletion`                      |
| Provided interface fluxes               | –                                                                        | `CentralNumericalFlux`, `RusanovNumericalFlux`; user functions such as Roe       |
| Flux-differencing volume term           | –                                                                        | `Operators.add_flux_differencing_divergence!` with a user two-point flux         |
| Non-conservative (gradient, curl) terms | Strong form plus DSS                                                     | `Operators.add_lifting_flux_interior!`                                           |
| Scalar Laplacian and ∇⁴ hyperdiffusion  | `scalar_laplacian` (weak div of strong grad)                             | `scalar_laplacian` with interior-penalty face terms (`sipg_laplacian_tendency!`) |
| Vector Laplacian                        | `vector_laplacian` (grad-div minus curl-curl)                            | Not implemented; `vector_laplacian` raises an error on a DG space                |
| `FieldVector` tendency completion       | One batched DSS                                                          | The state is one field with a composite element type                             |
| Horizontal boundary conditions          | Periodic, or imposed by the model on the state (the operators take none) | `PeriodicBC`, `ReflectingWallBC`, or a one-sided `boundary_numflux`              |
| Serialization                           | Discretization stored by `InputOutput`; files without it read back as CG |                                                                                  |

The discretization is a type parameter of `SpectralElementGrid1D` and
`SpectralElementGrid2D`, set with the `discretization` keyword of the grid
and space constructors and read back with `Spaces.discretization`. An omitted
keyword follows the quadrature: `CG()` when the nodes are shared across
element boundaries (GLL), `DG()` otherwise. Because the discretization is a
type parameter, code that dispatches on it resolves at compile time, and the
CG code path is unchanged by the presence of DG.

## Choosing between them

CG is the discretization of the production atmosphere and land models: it
conserves without fixers, DSS is cheap, and the tuned hyperdiffusion is well
understood. DG runs with its interface dissipation as the only closure, so it
needs no hyperdiffusion, is stable for under-resolved flows,
and keeps elements independent, which is attractive for limiters and for local
time stepping; it adds a numerical flux evaluation per face and the
flux-differencing volume term, and in ClimaCore the `FieldVector` completion
and the vector Laplacian are available on CG only. A model written with `tendency_completion` can be run both ways
and compared; the [CG and DG tutorial](../tutorials/cg_dg_switch.md) does
this for the shallow-water Bickley jet.
