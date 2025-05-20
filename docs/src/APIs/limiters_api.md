# Limiters

```@meta
CurrentModule = ClimaCore
```

The limiters supertype is

```@docs
Limiters.AbstractLimiter
```

This class of flux-limiters is applied only in the horizontal direction (on spectral advection operators).

## Interfaces

```@docs
Limiters.QuasiMonotoneLimiter
Limiters.compute_bounds!
Limiters.apply_limiter!
```

## Internals

```@docs
Limiters.compute_element_bounds!
Limiters.compute_neighbor_bounds_local!
Limiters.compute_neighbor_bounds_ghost!
Limiters.apply_limit_slab!
```
