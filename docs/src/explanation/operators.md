# Operators and broadcasting

A ClimaCore operator is an object that participates in a broadcast
expression, in the place a function would take. This page explains that
design, the rules an operator expression follows, and the consequences for
performance and for aliasing.

## Operators are broadcastable stencils

Julia's dot syntax fuses a chain of elementwise operations into one loop:
`@. c = a * b + sin(a)` visits each element once. ClimaCore extends the
broadcast machinery so that spatial operators take part in the same fusion.
In

```julia
wdiv = Operators.Divergence{Operators.WeakForm}()
grad = Operators.Gradient()
@. dydt = -wdiv(ρ * grad(θ))
```

`grad(θ)` and `wdiv(...)` are broadcast calls of operator objects. No
intermediate field is allocated for the gradient or the product: the whole
right-hand side compiles to one kernel that, for each element, reads `θ` and
`ρ` at the element's nodes, applies the differentiation matrix, multiplies,
applies the weak divergence, and writes `dydt`. Finite-difference operators
fuse the same way, column by column: `@. dydt = div(κ * grad(θ))` with
`GradientC2F` and `DivergenceF2C` is one pass over each column.

Operators are matrix-free. The action of an operator on a field is defined
directly; no global matrix is assembled. When a matrix is needed, for an
implicit vertical solve, `MatrixFields.operator_matrix` returns the banded
matrix of a finite-difference operator as a field of matrix rows, and
`MatrixFields` provides the algebra on those.

## Where an operator's result lives

Every operator has an input space and an output space, and a broadcast is
checked for consistency: a spectral-element operator returns a field on the
same space as its argument; a `C2F` operator on a center field returns a face
field, and an `F2C` operator the reverse. An expression that mixes center and
face fields needs an interpolation between them, and the broadcast checks
this at construction. The integrating reductions `sum`, `mean`, and `norm`
weight each node by the quadrature weight times the Jacobian, so `sum(ρ)` is
the mass in the domain, not the sum of nodal values (`norm` divides by the
domain volume by default); `maximum`, `minimum`, and `extrema` are plain
reductions over the nodal values.

The value type of an operator's result follows from its input: a `Gradient`
of a scalar is a covariant vector, a `Divergence` of any vector type is a
scalar, and the conversion of the input to the components the stencil needs
happens inside the operator ([Hybrid grids and generalized
coordinates](geometry.md)). Fields of `NamedTuple`s are supported throughout,
so a whole model state `(; ρ, ρu, ρθ)` passes through one broadcast.

## Completing spectral-element results

Element-local spectral operators are incomplete at element-boundary nodes
until a completion step runs ([DSS and numerical fluxes](interelement.md)).
That step is a separate pass after the broadcast: a weak-form derivative is
written to a field, and `Spaces.weighted_dss!` or `Operators.complete_tendency!` is called
on the result afterwards. Because DSS is a separate pass, the pattern for a
hyperdiffusion is prepare, complete, apply:

```julia
∇²χ = similar(χ)
Operators.scalar_laplacian!(∇²χ, χ)             # prepare the intermediate
Spaces.weighted_dss!(∇²χ)                       # complete the intermediate
@. dydt -= ν * Operators.scalar_laplacian(∇²χ)  # second pass fuses into dydt
```

On a CG space, `scalar_laplacian` returns a lazy expression that fuses into
its consumer; on a DG space, it returns a materialized field that already
contains the interior-penalty face terms, and `weighted_dss!` is a no-op, so
the same three lines are correct on both.

## Aliasing

A spectral operator reads all nodes of an element before writing any, so an
in-place broadcast whose output aliases an operator's input, such as
`@. χ = wdiv(grad(χ))`, is undefined. The mutating atoms guard against this:
`Operators.scalar_laplacian!(out, χ)` copies `χ` to scratch when `out === χ`.
Finite-difference stencils have the same property along a column. When an
expression must update a field from its own derivative, write into a
separate field or use the `!` forms.

## Why fusion matters on a GPU

On a CPU, fusion saves memory traffic. On a GPU, it also saves kernel launches:
each unfused operation is one launch of a few microseconds, and a tendency
with dozens of terms written unfused is launch-bound at moderate resolution.
ClimaCore's broadcast kernels assign one GPU thread per nodal point;
finite-difference kernels give a column to a group of threads along its
levels, and spectral-element kernels give each element slab a group of
threads, one per node, that stages the slab in shared memory for the
differentiation matrix. The atmosphere's tendency is written so that
consecutive terms fuse into few kernels [Yatunin2026](@cite). The limit of
fusion is register pressure: a kernel that fuses too many operators spills
registers or exhausts shared memory and slows down, so where to cut a long
expression is a measured choice. The shared
[GPU performance guide](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/performance/gpu_performance.md)
gives the rules for writing kernel-compatible code, and
[Performance and portability](performance.md) the measurements.
