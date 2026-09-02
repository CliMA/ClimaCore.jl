# Spectral Element Discretizations: Continuous vs Discontinuous Galerkin

ClimaCore uses Spectral Element Methods (SEM) to discretize spatial domains. Our architecture unifies these methods such that element-local operations are mathematically and computationally identical, allowing us to support both Continuous Galerkin (CG) and Discontinuous Galerkin (DG) discretizations via a single configuration switch.

This page details the theoretical underpinnings drawn from [Souza2023](@cite) and [Yatunin2026](@cite).

## Shared Mathematical Foundation: Galerkin Projection on Elements

Both CG and DG in this framework share the same fundamental intra-element representations and volume integration schemes. The domain is partitioned into quadrilateral (or hexahedral) elements, and state variables are represented using tensor products of 1D Lagrange polynomial basis functions $L_n(\xi)$ through $N_q = N_p + 1$ Gauss-Lobatto-Legendre (GLL) quadrature points $\hat{\zeta}_n$:

```math
\psi(\xi^1, \xi^2, \xi^3) \approx \sum_{n_1, n_2, n_3} L_{n_1}(\xi^1) L_{n_2}(\xi^2) L_{n_3}(\xi^3) \hat{\psi}(n_1, n_2, n_3)
```

Because both methods share this nodal polynomial basis within elements, all element-interior (volume) operations—evaluating gradients, divergences, and weak-form volume tendencies—are computationally identical. The divergence in their mathematical formulation appears exclusively at element boundaries.

## Continuous Galerkin (CG) & Direct Stiffness Summation (DSS)

The Continuous Galerkin (CG) spectral element method enforces $C^0$ continuity across element boundaries. 
As described by Yatunin et al. [Yatunin2026](@cite), to enforce continuity and preserve conservation properties, the CG formulation applies the Direct Stiffness Summation (DSS) operator, denoted as $\mathcal{P}$.

The DSS operator replaces multi-valued boundary nodes with a single continuous value via volume-weighted averaging over collocated neighborhoods across adjacent elements:

```math
\mathcal{P}(\mathcal{I}\psi)[\boldsymbol{r}(\hat{\zeta}_{n_1}, \hat{\zeta}_{n_2})] = \frac{\sum_{m \in \mathcal{C}} \hat{\psi}_m \delta V_m}{\sum_{m \in \mathcal{C}} \delta V_m}
```

where $\mathcal{C}$ denotes the set of collocated nodes in adjacent elements, and $\delta V$ is the discrete volume element constructed from GLL weights and metric Jacobians.

### Conservation and SBP Property
By preserving the discrete inner product:
```math
\langle \mathcal{I}\phi, \mathcal{P}(\mathcal{I}\psi) \rangle = \langle \mathcal{I}\psi, \mathcal{P}(\mathcal{I}\phi) \rangle
```
the CG method ensures that differential operators satisfy discrete analogues of integration by parts (Summation-by-Parts or SBP properties), guaranteeing exact global conservation of mass, momentum, and energy without artificial fixers.

## Discontinuous Galerkin (DG) & Flux-Differencing

The Discontinuous Galerkin (DG) method does not enforce continuity across element boundaries, allowing for multi-valued solutions at interfaces. 
Instead of the DSS operator, DG uses numerical interface fluxes to communicate state information between decoupled elements.

Following the Flux-Differencing Discontinuous Galerkin (FDDG) method from Souza et al. [Souza2023](@cite), the discretization is split into volume fluxes and interface fluxes.

### Volume Fluxes and Kinetic Energy Preservation (KEP)
Within elements, FDDG uses two-point volume numerical fluxes to achieve non-linear stability and satisfy discrete entropy inequalities without relying on explicit filters or sponge layers. For example, using the Kennedy-Gruber KEP flux:
```math
\mathcal{F}_{\rho} = \{\rho\}\{u\}, \quad \mathcal{F}_{\rho u} = \{\rho\}\{u\} \otimes \{u\} + \{p\}\mathbb{I}
```
where $\{\cdot\}$ represents arithmetic averaging between interpolation nodes.

### Interface Fluxes and Upwinding
At element interfaces, DG relies on numerical fluxes to couple elements across boundaries:
```math
\mathcal{F}_{\text{interface}} = \mathcal{F}_{\text{central}} + \mathcal{F}_{\text{penalty}}
```
The penalty term $\mathcal{F}_{\text{penalty}}$ (e.g., Rusanov or Roe flux) applies dissipation proportional to the jump $\llbracket \psi \rrbracket = (\psi^+ - \psi^-)/2$ across the interface, adding the necessary numerical dissipation akin to upwinding to maintain stability.

## The CG/DG Switch Architecture

By separating the element-local volume tendency computation from the boundary coupling step, ClimaCore allows users to write the physical right-hand side (RHS) equations once. The codebase implements an `AbstractTendencyCompletion` mechanism where the grid type (`Grids.CG()` vs `Grids.DG()`) acts as a switch:

1. **Volume Tendency:** The model computes the shared weak-form volume tendency within each element independently.
2. **Boundary Completion:** The model dynamically dispatches the completion step based on the grid configuration flag:
   - If configured as `Grids.CG()`, the code applies `DSSCompletion`, invoking the DSS operator $\mathcal{P}$ to enforce boundary continuity.
   - If configured as `Grids.DG()`, the code applies `NumericalFluxCompletion`, evaluating the interface numerical flux using the jumps between elements.

This abstraction ensures that flipping a single configuration switch (`Grids.CG()` vs `Grids.DG()`) completely alters the interface coupling mechanism at runtime, seamlessly allowing the same model code to run both discrete formulations.
