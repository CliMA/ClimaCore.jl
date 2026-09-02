# Limit tracers

Spectral-element transport overshoots: a tracer bounded between 0 and 1
develops values outside that range where the flow is under-resolved.
`ClimaCore.Limiters` provides a horizontal quasi-monotone limiter that
restores element-local bounds while conserving mass, and a vertical mass
borrowing limiter that removes negative values. Both are applied to the
state after a time step or stage.

## Quasi-monotone limiter (horizontal)

The limiter of [GubaOpt2014](@cite) solves, in each element, the constrained
least-squares problem of finding the field closest to the transported one that
lies within bounds set by the neighboring elements' values, keeping the
element's tracer mass.

1. Create the limiter once for a tracer density field (a scalar or a
   `NamedTuple` of tracers), and keep it in the model's cache:

   ```julia
   limiter = Limiters.QuasiMonotoneLimiter(ρq)
   ```

2. Before the transport step, compute the bounds from the state the step
   starts from. In a multi-stage scheme, this is the stage's starting state:

   ```julia
   Limiters.compute_bounds!(limiter, y.ρq, y.ρ)
   ```

3. After the step (or stage), apply the limiter to the transported tracer
   density, with the transported density as the weight:

   ```julia
   Limiters.apply_limiter!(y_new.ρq, y_new.ρ, limiter)
   ```

On a distributed topology, `compute_bounds!` exchanges neighbor bounds
across process boundaries. [`examples/plane/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/plane/limiters_advection.jl),
[`examples/sphere/limiters_advection.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/sphere/limiters_advection.jl), and
[`examples/hybrid/sphere/deformation_flow.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/sphere/deformation_flow.jl) run the standard advection tests
with and without the limiter; the [Example gallery](../explanation/examples.md)
states their equations.

## Vertical mass borrowing limiter

`Limiters.VerticalMassBorrowingLimiter(q_min)` fills a negative tracer mass at
one level by borrowing from the level below, and continues downward (and, if
the bottom goes negative, back up) until every level satisfies the given
minimum, conserving the column's tracer mass [zhang2018impact](@cite):

```julia
limiter = Limiters.VerticalMassBorrowingLimiter((0.0, 0.0))  # one minimum per tracer
Limiters.apply_limiter!(q, ρ, limiter)
```

## Limited vertical reconstructions

Vertical advection can be made monotone at the operator level rather than by
a post-step limiter: `Operators.LinVanLeerC2F` (the van Leer limiter with the
local-extrema constraint of [Lin1994](@cite)), `Operators.TVDLimitedFluxC2F`
with a slope limiter (`RZeroLimiter`, `MinModLimiter`, `KorenLimiter`,
`SuperbeeLimiter`, …), and the flux-corrected transport operators
`FCTBorisBook` and `FCTZalesak` [BorisBook1973, zalesak1979fully](@cite) return
limited face fluxes given the face velocity and the center field. Their
docstrings on the [Operators](../reference/operators.md) reference page give
the stencils.
