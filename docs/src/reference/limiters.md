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
```

## Internals

```@docs
Limiters.compute_element_bounds!
Limiters.compute_neighbor_bounds_local!
Limiters.compute_neighbor_bounds_ghost!
Limiters.apply_limit_slab!
Limiters.column_massborrow!
```
