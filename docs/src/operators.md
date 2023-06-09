# Operators

```@meta
CurrentModule = ClimaCore.Operators
```

_Operators_ can compute spatial derivative operations.

 - for performance reasons, we need to be able to "fuse" multiple operators and
 function applications
 - Julia provides a tool for this: **broadcasting**, with a very flexible API

Can think of operators are "pseudo-functions": can't be called directly, but
act similar to functions in the context of broadcasting. They are matrix-free,
in the sense that we define the _action_ of the operator directly on a field,
without explicitly assembling the matrix representing the discretized operator.

## Spectral element operators

### Differential Operators
```@docs
Gradient
Divergence
WeakDivergence
WeakGradient
Curl
WeakCurl
```

### Interpolation Operators
```@docs
Interpolate
Restrict
```

### DSS
```@docs
Spaces.weighted_dss!
Spaces.create_ghost_buffer
Spaces.weighted_dss_start!
Spaces.weighted_dss_internal!
Spaces.weighted_dss_ghost!
```

## Finite difference operators

Finite difference operators are similar with some subtle differences:
- they can change staggering (center to face, or vice versa)
- they can span multiple elements
  - no DSS is required
  - boundary handling may be required

We use the following convention:
 - centers are indexed by integers `1, 2, ..., n`
 - faces are indexed by half integers `half, 1+half, ..., n+half`

```@docs
FiniteDifferenceOperator
```

### Interpolation operators

```@docs
InterpolateC2F
InterpolateF2C
WeightedInterpolateC2F
WeightedInterpolateF2C
UpwindBiasedProductC2F
Upwind3rdOrderBiasedProductC2F
FCTBorisBook
FCTZalesak
LeftBiasedC2F
RightBiasedC2F
LeftBiasedF2C
RightBiasedF2C
```

### Derivative operators

```@docs
GradientF2C
GradientC2F
AdvectionF2F
AdvectionC2C
DivergenceF2C
DivergenceC2F
CurlC2F
```

### Other

```@docs
SetBoundaryOperator
FirstOrderOneSided
ThirdOrderOneSided
```

## Finite difference boundary conditions

```@docs
BoundaryCondition
SetValue
SetGradient
SetDivergence
Extrapolate
```

## Internal APIs

```@docs
return_eltype
return_space
stencil_interior_width
stencil_interior
boundary_width
stencil_left_boundary
stencil_right_boundary
```
