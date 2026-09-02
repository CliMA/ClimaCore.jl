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
SplitDivergence
Curl
```

### Strong and weak forms

`Divergence`, `Gradient`, and `Curl` each have a strong and a weak variant,
selected by the [`FormType`](@ref) type parameter: `Divergence()` is the strong
form (`Divergence{StrongForm}`) and `Divergence{WeakForm}()` is the weak form
(and likewise for `Gradient` and `Curl`). Both forms are documented in the
operator docstrings above.

```@docs
FormType
StrongForm
WeakForm
```

### Interpolation Operators

```@docs
Interpolate
Restrict
```

## Finite difference operators

Finite difference operators are similar with some subtle differences:

  - they can change staggering (center to face, or vice versa)
  - they can span multiple elements
      + no DSS is required
      + boundary handling may be required

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

### Derivative operators

```@docs
GradientF2C
GradientC2F
DivergenceF2C
DivergenceC2F
CurlC2F
```

### Other

```@docs
SetBoundaryOperator
```

### Dirichlet (`SetValue`) replacement helpers

```@docs
DirichletOperator
gradient_c2f_dirichlet
divergence_c2f_dirichlet
curl_c2f_dirichlet
upwind_biased_product_c2f_dirichlet
```

## Finite difference boundary conditions

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

## Discontinuous Galerkin operators

Face and volume operators for discontinuous-Galerkin (DG) discretizations on
spectral-element spaces constructed with `discretization = Grids.DG()`. The
face operators act on a mass-weighted residual (`WJ * ∂Y/∂t`) and complete
the weak-form (or flux-differencing) volume terms at element interfaces.

The Laplacian below couples elements two ways: [`SIPGLaplacianFlux`](@ref) at
the faces, and a jump correction folded into the volume divergence. Including
both ensures order of accuracy is maintained with a symmetric operator.

```@docs
add_numerical_flux_interior!
add_numerical_flux_boundary!
add_lifting_flux_interior!
lifting_correction
add_flux_differencing_divergence!
add_sipg_laplacian_flux_interior!
sipg_laplacian_tendency
sipg_laplacian_tendency!
sipg_penalty_parameter
sipg_penalty_parameter!
start_dg_ghost_exchange
DGGhostExchange
```

### Model-level CG↔DG switching

A model's tendency assembly can be written once for both discretizations: the
element-local weak-form tendency is completed across element interfaces by a
completion object built from the space — DSS on continuous spaces, interface
numerical fluxes on discontinuous ones.

```@docs
tendency_completion
complete_tendency!
AbstractTendencyCompletion
DSSCompletion
NumericalFluxCompletion
```

[`cartesian_tensor_divergence`](@ref) uses this same switch to compute the
horizontal divergence of a rank-2 flux tensor (e.g. the momentum flux `ρu⊗u`)
on either discretization. The weak `Divergence` drops the Christoffel terms
`Γⁱ_jk Tʲᵏ` on the tensor's momentum axis on a curved space; the helper first
rotates that momentum axis into the global Cartesian basis (where the
Christoffel symbols vanish), then applies `Divergence` and completes the
interfaces with the supplied completion, so the plain operator becomes exact on
both CG and DG. Casting the conservation law's momentum components in the
Cartesian basis to eliminate the connection terms follows [Vinokur1974](@cite).

Like the spectral `Divergence` it is built from, it differentiates along the
horizontal directions only, so on an extruded space the vertical flux
divergence is a separate term. Keep that term in the Cartesian frame too — the
connection terms it drops are removed by the same rotation — and rotate it back
with `Geometry.LocalVector`. The docstring below works the pair through.

```@docs
cartesian_tensor_divergence
cartesian_tensor_divergence!
```

### Numerical fluxes and face lifts

```@docs
AbstractNumericalFlux
CentralNumericalFlux
RusanovNumericalFlux
SIPGLaplacianFlux
central_gradient_lift
central_curl3_lift
jump_penalty_lift
```

### DG boundary conditions

```@docs
HorizontalBoundaryCondition
ghost_state
PeriodicBC
ReflectingWallBC
```

## Integrals

```@docs
column_integral_definite!
column_integral_indefinite!
column_reduce!
column_accumulate!
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
left_interior_idx
right_interior_idx
```
