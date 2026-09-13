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

A boundary condition is attached to an operator by boundary name, e.g.
`GradientC2F(; bottom = SetGradient(v₀), top = SetGradient(v₁))`. Which
conditions an operator accepts is listed in its own docstring.

A boundary left without a condition is not an error, but what it does depends
on the operator:

| Operator                                                                                                                       | Boundary left without a condition                           |
|:------------------------------------------------------------------------------------------------------------------------------ |:----------------------------------------------------------- |
| `InterpolateC2F`, `GradientC2F`, `DivergenceC2F`, `CurlC2F`                                                                    | that boundary face is `NaN`                                 |
| `InterpolateF2C`, `GradientF2C`, `DivergenceF2C`                                                                               | not needed; every center value is well defined              |
| `UpwindBiasedProductC2F`, `Upwind3rdOrderBiasedProductC2F`, `LinVanLeerC2F`, `FCTBorisBook`, `FCTZalesak`, `TVDLimitedFluxC2F` | defaults to `Extrapolate()`, the only condition they accept |

A center-to-face stencil needs a center value on either side of the face and
the boundary faces have only one, so there is nothing to fall back on; filling
them with `NaN` means a forgotten boundary condition shows up in the output
rather than silently producing a plausible number.

Such a boundary only needs a condition if the enclosing broadcast actually
reads that face: in `divf2c.(gradc2f.(x))`, `DivergenceF2C`'s own boundary
handling means the boundary faces of the inner `GradientC2F` are never read.

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
