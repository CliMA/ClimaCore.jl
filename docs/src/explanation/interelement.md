# DSS and numerical fluxes

A spectral-element derivative is computed one element at a time. On the
boundary nodes of an element, the result is incomplete: it has seen only that
element's side of the field. Two mechanisms complete it. On a continuous (CG)
grid, it is direct stiffness summation (DSS); on a discontinuous (DG) grid, it
is a numerical flux. This page describes both, and the `tendency_completion`
interface that lets a model use either through one call.

## Direct stiffness summation

On a CG grid, a node on an element boundary has one physical location and
several stored copies, one per element that touches it. A field is continuous
when all copies agree. After an element-local operation, the copies disagree,
and DSS makes them agree again: each copy is replaced by the volume-weighted
average of all copies at that location,

```math
P(\psi)(x) = \frac{\sum_{m \in C(x)} \psi_m\, \delta V_m}{\sum_{m \in C(x)} \delta V_m},
\qquad \delta V_m = w_{i} w_{j} J_m,
```

where `C(x)` is the set of copies at `x`, `w` are the quadrature weights, and
`J` is the metric Jacobian at that node [Deville02a, Taylor2010](@cite).
Horizontal vector components are averaged after conversion to the local
orthonormal (`UVW`) frame and converted back afterwards, because covariant
and contravariant components in neighboring elements refer to different
bases; vertical (`Covariant3`, `Contravariant3`) components are averaged as
they are, since the vertical basis vector is continuous across a horizontal
element boundary.

`Spaces.weighted_dss!(field, buffer)` performs the operation. On a distributed
topology, it is the only horizontal communication in the model: interior copies
are averaged locally, boundary copies are exchanged with the owning
processes through a `DSSBuffer` created once with `Spaces.create_dss_buffer`.
The three-phase form `weighted_dss_start!` / `weighted_dss_internal!` /
`weighted_dss_ghost!` overlaps the exchange with local work. Several fields
share one exchange when passed together, and a `FieldVector` is completed in
one batched call.

DSS preserves the discrete inner product: for any two fields,
`⟨φ, 𝒫ψ⟩ = ⟨𝒫φ, ψ⟩`. That is what carries the summation-by-parts identity of
the weak operators from element level to the whole domain, and it is why the
CG atmosphere conserves mass, energy, and tracer mass to round-off without
fixers [Yatunin2026](@cite). The atmosphere applies DSS after each stage of
the time integrator and after each implicit solve, so that the state stays
continuous across element boundaries to round-off.

## Interface numerical fluxes

On a DG grid, the copies of a boundary node are independent degrees of freedom
and stay so. Coupling enters as the surface term of the weak form: for a
conservation law `∂ₜy + ∇⋅F(y) = 0`, each element face contributes the
numerical flux `F*(y⁻, y⁺; n̂)`, a single-valued flux computed from the state
`y⁻` inside the element, the state `y⁺` in the neighbor, and the outward unit
normal `n̂` [Hesthaven07a](@cite) ([Spectral elements: CG and
DG](discretizations.md) derives the term).
`Operators.add_numerical_flux_interior!(numflux, dydt, y, args...)` adds it,
weighted by the face Jacobian, to the mass-weighted residual `WJ ∂ₜy` on
every interior face: subtracted on the minus side and added on the plus
side.
The flux function is called as `numflux(normal, argvals⁻, argvals⁺)` and must
be antisymmetric, `numflux(n̂, a⁻, a⁺) == -numflux(-n̂, a⁺, a⁻)`, so that
the two elements sharing a face exchange equal and opposite amounts; that
antisymmetry is what makes the scheme globally conservative.

Two fluxes are provided for a physical flux function `F`:

  - `Operators.CentralNumericalFlux(F)`: `(F(y⁻) + F(y⁺))/2 ⋅ n̂`. No
    dissipation, so a transport problem run with it is unstable.
  - `Operators.RusanovNumericalFlux(F, λ)`: the central flux plus
    `(λ/2)(y⁻ − y⁺)` with `λ` the larger of the two sides' maximal signal
    speeds. The penalty acts on the jump across the face and is the dissipation
    that stabilizes the scheme; it vanishes where the solution is continuous.
    A Roe flux [Roe81a](@cite) is a less dissipative choice of the same form
    and can be supplied as a user function (the DG Bickley jet example does).

Non-conservative terms (a gradient or a curl) are completed by *lifting*
rather than by a flux: both elements at a face receive their own correction,
`Operators.add_lifting_flux_interior!`. Domain boundaries are handled by
`Operators.add_numerical_flux_boundary!` with a one-sided flux or a
`HorizontalBoundaryCondition` such as `ReflectingWallBC`, which constructs the
exterior ghost state.

On a distributed topology, the face terms need the neighbor's boundary
values. `Operators.start_dg_ghost_exchange` starts one halo exchange for a
state that several face operators then share, so each halo message is sent
once per tendency evaluation rather than once per operator.

## The same tendency on both grids

The volume part of a weak-form tendency is identical on CG and DG grids: both
use the same nodal basis and the same element-local weak operators. Only the
completion differs. `Operators.tendency_completion(dydt; numflux, boundary_numflux)`
inspects the discretization of the space that `dydt` lives on and returns

  - a `DSSCompletion`, holding a DSS buffer, on `Grids.CG()`, or
  - a `NumericalFluxCompletion`, holding the flux functions, on `Grids.DG()`.

`Operators.complete_tendency!(completion, dydt, y, args...)` then applies DSS,
or weights by `WJ`, adds the interior and boundary fluxes, and unweights. A
model's tendency function is written once:

```julia
function rhs!(dydt, y, (params, completion), t)
    wdiv = Operators.Divergence{Operators.WeakForm}()
    rparams = Ref(params)
    @. dydt = -wdiv(physical_flux(y, rparams))
    Operators.complete_tendency!(completion, dydt, y, params)
    return dydt
end
```

The `numflux` keyword is required on a DG space and ignored on a CG space, so
a model passes it unconditionally. The completion is selected by dispatch on
the grid's type parameter, so the choice costs nothing at run time. A
`FieldVector` tendency is supported on CG spaces, where the completion is one
batched DSS; on DG spaces the interface flux needs the whole state at a face
node, so the tendency must be a single field with a composite element type.
[Tutorial: CG and DG with one tendency](../tutorials/cg_dg_switch.md) runs the
shallow-water Bickley jet both ways.

## Element ordering and communication

Elements are numbered along a space-filling curve
(`Topologies.spacefillingcurve`), a Hilbert-type curve on each cubed-sphere
panel [Cerveny24a](@cite) and on rectilinear meshes, so that elements that
are neighbors in space are close in memory and a contiguous range of the
curve is a compact patch for one MPI process. The
[Topologies](../reference/topologies.md) reference shows the curve on the
cubed sphere.
