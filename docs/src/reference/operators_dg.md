# Operators: discontinuous Galerkin

```@meta
CurrentModule = ClimaCore.Operators
```

Face and volume operators for discontinuous-Galerkin (DG) discretizations on
spectral-element spaces constructed with `discretization = Grids.DG()` ([Choose CG or DG](../howto/choose_cg_dg.md)). The
face operators act on a mass-weighted residual (`WJ * ∂Y/∂t`) and complete
the weak-form (or flux-differencing) volume terms at element interfaces.

The Laplacian couples elements in two ways: through [`SIPGLaplacianFlux`](@ref) at
the faces and through a jump correction folded into the volume divergence. Together,
the two give a symmetric operator that keeps the order of accuracy.

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

## Model-level CG↔DG switching

A model's tendency assembly works on both discretizations: the
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

## Tensor divergence

The weak-form horizontal divergence of a rank-2 flux tensor (the momentum flux
`ρu⊗u`, for instance) runs on both CG and DG spaces through the same completion.
Rotating the momentum axis into the global Cartesian basis, where the Christoffel
symbols vanish, makes the connection term the weak `Divergence` omits exact
[Vinokur1974](@cite); the interface coupling is DSS on a CG space and the
numerical flux on a DG one.

```@docs
cartesian_tensor_divergence
cartesian_tensor_divergence!
```

## Numerical fluxes and face lifts

```@docs
AbstractNumericalFlux
CentralNumericalFlux
RusanovNumericalFlux
SIPGLaplacianFlux
central_gradient_lift
central_curl3_lift
jump_penalty_lift
```

## Boundary conditions

```@docs
HorizontalBoundaryCondition
ghost_state
PeriodicBC
ReflectingWallBC
```
