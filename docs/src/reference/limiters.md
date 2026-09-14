# Limiters

```@meta
CurrentModule = ClimaCore
```

The limiters supertype is

```@docs
Limiters.AbstractLimiter
```

`QuasiMonotoneLimiter` acts on the horizontal spectral-element structure of a
field; `VerticalMassBorrowingLimiter` acts along each column. Both are applied
to the state after a step or stage ([Limit tracers](../howto/limiters.md)).

## Interfaces

```@docs
Limiters.QuasiMonotoneLimiter
Limiters.VerticalMassBorrowingLimiter
Limiters.compute_bounds!
Limiters.apply_limiter!
Limiters.print_convergence_stats
```
