# Limiters

```@meta
CurrentModule = ClimaCore
```

The limiters supertype is

```@docs
Limiters.AbstractLimiter
```

`QuasiMonotoneLimiter` acts on the horizontal spectral-element structure of a
field; `VerticalMassBorrowingLimiter` acts along each column;
`PositivityLimiter` acts on the conserved state of a DG discretization,
element by element. All are applied to the state after a step or stage
([Limit tracers](../howto/limiters.md)).

## Interfaces

```@docs
Limiters.QuasiMonotoneLimiter
Limiters.VerticalMassBorrowingLimiter
Limiters.PositivityLimiter
Limiters.compute_bounds!
Limiters.apply_limiter!
Limiters.apply_positivity_limiter!
Limiters.print_convergence_stats
```
