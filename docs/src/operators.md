# Operators

```@meta
CurrentModule = ClimaCore.Operators
```

## Spectral element operators

### Differential Operators
```@docs
Divergence
WeakDivergence
Gradient
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
```

## Finite difference operators

### Interpolation operators

```@docs
InterpolateC2F
InterpolateF2C
WeightedInterpolateC2F
WeightedInterpolateF2C
UpwindBiasedProductC2F
LeftBiasedC2F
RightBiasedC2F
LeftBiasedF2C
RightBiasedF2C
```

### Derivative operators

```@docs
AdvectionF2F
AdvectionC2C
DivergenceF2C
DivergenceC2F
GradientF2C
GradientC2F
CurlC2F
```

### Other

```@docs
SetBoundaryOperator
```

## Finite difference boundary conditions

```@docs
SetValue
SetGradient
SetDivergence
Extrapolate
```
