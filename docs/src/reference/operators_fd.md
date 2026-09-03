# Operators: finite difference

```@meta
CurrentModule = ClimaCore.Operators
```

Stencil operators along a column. They map between the two staggerings
(`C2F` from centers to faces, `F2C` from faces to centers), reach across cell
boundaries without DSS, and take boundary conditions by the names of the
domain's boundaries ([Staggered vertical discretization](../explanation/vertical.md),
[Apply boundary conditions](../howto/boundary_conditions.md)). Centers are
indexed by integers `1, …, n`; faces are addressed with `Utilities.PlusHalf`
values, `half, 1 + half, …, n + half` (`half = PlusHalf(0)`), integers tagged as
face positions, which the stencil docstrings write as `½, …, n + ½`.

```@docs
FiniteDifferenceOperator
```

## Interpolation operators

```@docs
InterpolateC2F
InterpolateF2C
WeightedInterpolateC2F
WeightedInterpolateF2C
AdvectionOperator
UpwindBiasedProductC2F
Upwind3rdOrderBiasedProductC2F
FCTBorisBook
FCTZalesak
LinVanLeerC2F
TVDLimitedFluxC2F
BottomBiasedC2F
TopBiasedC2F
BottomBiasedF2C
TopBiasedF2C
AbstractTVDSlopeLimiter
```

## Derivative operators

```@docs
GradientF2C
GradientC2F
DivergenceF2C
DivergenceC2F
CurlC2F
```

## Boundary operators

```@docs
SetBoundaryOperator
```

## Dirichlet (`SetValue`) replacement helpers

```@docs
DirichletOperator
gradient_c2f_dirichlet
divergence_c2f_dirichlet
curl_c2f_dirichlet
upwind_biased_product_c2f_dirichlet
```

## Boundary conditions

```@docs
AbstractBoundaryCondition
VerticalBoundaryCondition
SetCurl
SetValue
SetGradient
SetDivergence
Extrapolate
Outflow
```

[`Outflow`](@ref) is a physically named convenience constructor for
[`Extrapolate`](@ref) (an outflow extrapolation whose order-0 case is the
zero-normal-gradient closure), accepted wherever `Extrapolate` is.

## Integrals

```@docs
column_integral_definite!
column_integral_indefinite!
column_reduce!
column_accumulate!
```
