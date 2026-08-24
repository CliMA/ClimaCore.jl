<div align="center">
  <img src="docs/src/assets/logo.svg" alt="ClimaCore.jl Logo" width="128" height="128">
</div>

# ClimaCore.jl

The dynamical core (_dycore_) of the CliMA Earth System Model: composable, GPU-capable tools for discretizing and solving partial differential equations on the sphere and in Cartesian domains.

|||
|------------------:|:------------------------------------------------------------|
| **Documentation** | [![stable][docs-stable-img]][docs-stable-url] [![dev][docs-dev-img]][docs-dev-url] |
| **Version**       | [![version][version-img]][version-url]                      |
| **License**       | [![license][license-img]][license-url]                      |
| **Tests**         | [![gha ci][gha-ci-img]][gha-ci-url] [![buildkite][bk-ci-img]][bk-ci-url] |
| **Code Coverage** | [![codecov][codecov-img]][codecov-url]                      |
| **Downloads**     | [![Downloads][dlt-img]][dlt-url]                            |
| **DOI**           | [![zenodo][zenodo-img]][zenodo-url]                         |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://CliMA.github.io/ClimaCore.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaCore.jl/dev/

[version-img]: https://juliahub.com/docs/General/ClimaCore/stable/version.svg
[version-url]: https://juliahub.com/ui/Packages/General/ClimaCore

[license-img]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: https://github.com/CliMA/ClimaCore.jl/blob/main/LICENSE

[gha-ci-img]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml/badge.svg?branch=main
[gha-ci-url]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml?query=branch%3Amain

[bk-ci-img]: https://badge.buildkite.com/2b63d3c49347804f61bd8e99c8b85e05871253b92612cd1af4.svg?branch=main
[bk-ci-url]: https://buildkite.com/clima/climacore-ci/builds?branch=main

[codecov-img]: https://codecov.io/gh/CliMA/ClimaCore.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaCore.jl

[dlt-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaCore&query=total_requests&label=Downloads
[dlt-url]: https://juliapkgstats.com/pkg/ClimaCore

[zenodo-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.5554759-blue.svg
[zenodo-url]: https://zenodo.org/badge/latestdoi/356355994

ClimaCore.jl is a library for building PDE solvers, on the sphere, in boxes
and vertical slices, and in single columns. It provides the discretization
(continuous or discontinuous spectral elements in the horizontal, staggered
finite differences in the vertical) as composable operators that you broadcast
over fields, and the same source runs on a laptop, on an NVIDIA GPU, and across
MPI ranks.

It is the dynamical core underneath [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl)
and [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), and it is a
standalone package: nothing in it assumes you are building a climate model.

## Who this is for

  - **Writing a model, not running one.** ClimaCore provides spaces, fields,
    and operators, and stays out of the physics. A configured atmosphere
    (radiation, microphysics, turbulence and convection, surface fluxes) lives
    in [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), which is built
    on this package.
  - **Geophysical fluid dynamics without the whole Earth system model.**
    Shallow water on the sphere, barotropic instability, mountain waves,
    gravity waves, density currents, rising thermals, and tracer transport
    under prescribed winds are each a complete, tested program in
    [`examples/`](examples/), a few hundred lines end to end.
  - **Dynamical-core development.** You can change the prognostic variables,
    the flux form, the limiter, the implicit/explicit split, or the vertical
    staggering without forking a model whose discretization and equations are
    the same code.
  - **GPU without writing kernels.** A broadcast expression over fields
    compiles to one fused kernel, specialized on the polynomial degree;
    switching devices is an environment variable.

The horizontal discretization is the one used by the spectral-element
atmospheric dynamical cores: a cubed sphere of Gauss–Legendre–Lobatto elements
with direct stiffness summation, the family that CAM-SE/HOMME, NUMA, and their
performance-portable rewrites belong to. Here it is a library, so the governing
equations are not built in; you write them.

## Features

  - **Spectral-element horizontal discretizations**: continuous (CG) and discontinuous (DG) Galerkin spectral elements, on the cubed sphere and on Cartesian planes.
  - **Flexible vertical discretization**: staggered finite differences on center/face grids, with stretching and terrain-following (hypsography) coordinates.
  - **Multiple geometries**: Cartesian and spherical domains, with governing equations expressed in covariant vectors for curvilinear systems and Cartesian vectors for Euclidean spaces. The metric terms are applied by the operators.
  - **`Field` abstraction**: scalar-, vector-, or struct-valued fields carrying values, geometry, and mesh information, with flexible memory layouts (AoS, SoA, AoSoA) and useful overloads (`sum`, `norm`, ...).
  - **Composable operators via broadcasting**: differential operators (`grad`, `div`, `curl`, `interpolate`, ...) act like functions when broadcast over a `Field`, fusing operators and function calls into a single pass.
  - **Implicit solvers for the vertical**: `MatrixFields` builds banded Jacobians column by column and solves them, which is what IMEX time stepping of the vertical acoustic and diffusive terms needs.
  - **GPU acceleration**: broadcast expressions compile to custom CUDA kernels, with specialization on polynomial degree for kernel performance.
  - **Distributed by construction**: element topologies carry their halo exchange, so the same tendency function runs on one rank or hundreds.
  - **Time-stepper compatible**: `Field`s and `FieldVector`s act as the state vector for [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl), which the tests and examples here time-step with.

## Installation

ClimaCore.jl is a registered Julia package (Julia v1.10 or later):

```julia
using Pkg
Pkg.add(["ClimaCore", "ClimaComms", "ClimaTimeSteppers"])
```

## Quick start

Advect a tracer once around the sphere. The whole program is a space, two
fields, a tendency, and a time stepper:

```julia
using ClimaComms
ClimaComms.@import_required_backends
using ClimaCore.CommonSpaces
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaTimeSteppers as CTS

const R = 6.37122e6               # planet radius (m)
const u₀ = 2π * R / (12 * 86400)  # once around in 12 days
const h₀ = 1000.0                 # bell height
const r₀ = R / 3                  # bell radius
const center = Geometry.LatLongPoint(0.0, 270.0)

# 6 × 16² spectral elements on the cubed sphere, 4 × 4 GLL nodes in each
space = CubedSphereSpace(; radius = R, h_elem = 16, n_quad_points = 4)
coords = Fields.coordinate_field(space)
const global_geom = Spaces.global_geometry(space)

u = @. Geometry.UVVector(u₀ * cosd(coords.lat), 0.0)       # solid-body rotation
h = map(coords) do c
    rd = Geometry.great_circle_distance(c, center, global_geom)
    rd < r₀ ? h₀ / 2 * (1 + cospi(rd / r₀)) : 0.0          # cosine bell
end

function transport!(dh, h, u, t)
    div = Operators.Divergence()
    @. dh = -div(h * u)          # ∂h/∂t = -∇·(u h)
    Spaces.weighted_dss!(dh)     # the spectral-element gather across elements
end

prob = CTS.ODEProblem(
    CTS.ClimaODEFunction(; T_exp! = transport!),
    copy(h),
    (0.0, 12 * 86400.0),
    u,
)
sol = CTS.solve(prob, CTS.ExplicitAlgorithm(CTS.SSP33ShuOsher()); dt = 20 * 60.0)

# The bell returns to where it started. The transport is a flux divergence
# over a closed surface, so mass is conserved to roundoff (measured drift:
# 4e-15), and the L₁ error decreases by roughly a factor of 3 per halving of
# the element size.
@assert sum(sol.u[end]) ≈ sum(h)
```

Operators are matrix-free: they define the action of the operator directly on a
field, and broadcasting fuses operators and function calls into a single pass.
The metric terms come from the space, so the divergence knows it is on a sphere
without the user writing them.

## Examples

### Shallow-water on the sphere

The shallow-water system in vector-invariant form, the standard testbed for a
horizontal dynamical core, is two broadcast expressions:

```julia
using LinearAlgebra: norm, ×

# ∂h/∂t = -∇·(h u)
# ∂u/∂t = -∇(g(h + hₛ) + |u|²/2) + u × (f + ∇×u)
function shallow_water!(dY, Y, (; f, h_s, g), t)
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    curl = Operators.Curl()
    @. dY.h = -wdiv(Y.h * Y.u)
    @. dY.u = -grad(g * (Y.h + h_s) + norm(Y.u)^2 / 2) + Y.u × (f + curl(Y.u))
    Spaces.weighted_dss!(dY)
end
```

`Y.u` is a covariant vector field, `f` a contravariant one; the cross product,
the curl, and the gradient apply the cubed-sphere metric terms themselves.
Hyperviscosity is two more lines of the same kind
(`wdiv(grad(·))` applied twice).

[`examples/sphere/shallow_water.jl`](examples/sphere/shallow_water.jl) runs this
over the standard test suite of Williamson et al. (1992): steady-state
geostrophic flow, flow over a mountain, barotropic instability, and the
Rossby–Haurwitz wave, with the hyperviscosity and the error norms included.

### Column operators with built-in boundary conditions

Vertical operators carry their boundary conditions, so a Dirichlet bottom and a
prescribed-flux top are arguments rather than special-cased index arithmetic:

```julia
using ClimaCore.CommonSpaces
import ClimaCore: Fields, Geometry, Operators

space = ColumnSpace(; z_min = 0.0, z_max = 1.0, z_elem = 10, staggering = CellCenter())
T = Fields.zeros(Float64, space)

function heat!(dT, T, α, t)
    gradc2f = Operators.GradientC2F(
        bottom = Operators.SetValue(0.0),                    # T = 0 at the surface
        top = Operators.SetGradient(Geometry.WVector(1.0)),  # ∂T/∂z = 1 at the top
    )
    divf2c = Operators.DivergenceF2C()
    @. dT = α * divf2c(gradc2f(T))   # ∂T/∂t = α ∇²T, centers → faces → centers
end
```

`GradientC2F` maps cell centers to faces and `DivergenceF2C` maps back, which is
the staggering that keeps a diffusion operator compact. The same pattern carries
the vertical transport, diffusion, and surface fluxes in every CliMA component
model. See [`examples/column/`](examples/column/) for this case with its exact
solution, an Ekman spiral, hydrostatic balance, and advection with limiters.

### Benchmark test cases

Each of these is a runnable program that also asserts what the case is supposed
to show, so they double as a starting point and as a regression test of your
changes:

| Case | What it exercises | File |
|:--|:--|:--|
| Solid-body tracer transport | horizontal transport, cubed-sphere panel edges | [`sphere/solid_body_rotation.jl`](examples/sphere/solid_body_rotation.jl) |
| Shallow-water suite, Williamson et al. (1992) | vector-invariant form, hyperviscosity, conservation | [`sphere/shallow_water.jl`](examples/sphere/shallow_water.jl) |
| Bickley jet, CG and DG | barotropic instability, energy conservation, over-integration | [`plane/bickleyjet_cg.jl`](examples/plane/bickleyjet_cg.jl) |
| Deformational flow and Hadley circulation (DCMIP 2012) | filamentation, limiters, vertical–horizontal coupling | [`hybrid/sphere/deformation_flow.jl`](examples/hybrid/sphere/deformation_flow.jl), [`hadley_circulation.jl`](examples/hybrid/sphere/hadley_circulation.jl) |
| Rising thermal bubble, 2D and 3D | nonhydrostatic compressible flow in a box | [`hybrid/box/bubble_3d_invariant_rhoe.jl`](examples/hybrid/box/bubble_3d_invariant_rhoe.jl) |
| Density current | advection at a rolling-up front (Kelvin–Helmholtz billows) | [`hybrid/plane/density_current_2d_invariant_rhoe.jl`](examples/hybrid/plane/density_current_2d_invariant_rhoe.jl) |
| Mountain waves, Schär et al. (2002) and witch-of-Agnesi | terrain-following metric terms, sponge layers | [`hybrid/plane/schar_mountain.jl`](examples/hybrid/plane/schar_mountain.jl) |
| Inertial and nonhydrostatic gravity waves | IMEX splitting against linear analytic solutions | [`hybrid/plane/inertial_gravity_wave.jl`](examples/hybrid/plane/inertial_gravity_wave.jl) |
| Baroclinic wave, Ullrich et al. (2014) | the standard dry dynamical-core benchmark | [`hybrid/sphere/baroclinic_wave_rhoe.jl`](examples/hybrid/sphere/baroclinic_wave_rhoe.jl) |

### Running on CPU, GPU, and MPI

The device and the communication context come from the environment through
[ClimaComms.jl](https://github.com/CliMA/ClimaComms.jl), so a script does not
change when the hardware does:

```bash
julia --project my_solver.jl                                     # CPU
CLIMACOMMS_DEVICE=CUDA julia --project my_solver.jl              # one GPU
CLIMACOMMS_CONTEXT=MPI CLIMACOMMS_DEVICE=CUDA \
    srun --ntasks=4 julia --project my_solver.jl                 # four GPUs
```

Element topologies carry their own halo exchange, and `weighted_dss!` performs
it, so a tendency written for one rank is already the distributed one.

### Output and analysis

  - `Remapping.interpolate(field)` returns a plain (or `CuArray`) array on a
    uniform lat–long–z grid, suitable for plotting and diagnostics;
    `Remapping.Remapper` is the reusable, allocation-free version.
  - `InputOutput.HDF5Writer` / `HDF5Reader` checkpoint and restore fields and
    the spaces they live on, in serial or in parallel.
  - [`lib/`](lib/) holds the visualization and analysis companions:
    `ClimaCorePlots` and `ClimaCoreMakie` recipes, `ClimaCoreVTK` output,
    `ClimaCoreSpectra` for spherical-harmonic energy spectra, and
    `ClimaCoreTempestRemap` for conservative remapping.
  - Spaces support masks (`enable_mask = true` plus `set_mask!`), so nodes over
    the ocean (or wherever data is missing) can be skipped entirely.

## Scope: what is in the box, and what is not

**In:** spectral elements (CG/DG) on cubed spheres and rectangles; staggered
finite differences in the vertical, with stretching, terrain-following
coordinates, upwind and limited transport operators; covariant/contravariant
geometry; distributed topologies and CUDA; banded matrix fields with direct
solvers for column-coupled implicit systems; HDF5 I/O and remapping.

**Out:**

  - **Physics.** No radiation, microphysics, turbulence, or surface
    parameterizations. Those live in
    [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) and
    [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl).
  - **Time stepping.** `Field`s and `FieldVector`s are state vectors;
    [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl)
    integrates them.
  - **Global elliptic solves.** The direct solvers in `MatrixFields` are banded
    within a column, which is what vertically-implicit IMEX schemes need. There
    is no built-in pressure Poisson solver; `FieldVector`s do work with
    [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl), so a
    matrix-free iterative solve is possible, but you would write it yourself.
  - **Adaptive meshes.** No AMR; the horizontal mesh is an equiangular cubed
    sphere or a rectilinear box, refined by changing the element count or the
    polynomial degree.

### LES and DNS

LES fits. `Box3DSpace(; periodic_x = true, periodic_y = true, ...)` builds a
doubly periodic box with spectral elements in the horizontal and stretched
finite differences in the vertical, the standard layout for atmospheric
boundary-layer LES. The compressible equations to time-step in it are the ones
the [`examples/hybrid/box/`](examples/hybrid/box/) cases already solve, and
ClimaAtmos runs box LES on top of them with a Smagorinsky–Lilly closure.

Canonical DNS is a different matter. The vertical is second-order finite
differences rather than spectral, so vertical resolution has to be bought with
points rather than with polynomial order, and the fully incompressible route
needs the global pressure solve noted above. For high-Reynolds-number DNS
benchmarks in a periodic box, purpose-built spectral solvers such as
[Nek5000/nekRS](https://github.com/Nek5000/nekRS) or
[Dedalus](https://dedalus-project.org/) remain the better fit. ClimaCore is
built for stratified, rotating, thin-shell flows where the horizontal and
vertical scales differ by orders of magnitude, which is why the horizontal and
the vertical are discretized differently.

## Documentation

  - **[Stable docs](https://CliMA.github.io/ClimaCore.jl/stable/)**: installation, introduction, mathematical framework, and API reference
  - **[Dev docs](https://CliMA.github.io/ClimaCore.jl/dev/)**: latest development version
  - **[`examples/`](examples/)**: runnable examples across geometries
  - **[Operators](https://CliMA.github.io/ClimaCore.jl/dev/operators/)**, **[Matrix fields](https://CliMA.github.io/ClimaCore.jl/dev/matrix_fields/)**, and **[Remapping](https://CliMA.github.io/ClimaCore.jl/dev/remapping/)**, the pages most solvers need next

## Where to go next

Within the CliMA ecosystem:

  - [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl): atmosphere model with physics, diagnostics, and calibration on top of these operators
  - [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl): land model
  - [ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl): coupled atmosphere/land/ocean/sea-ice simulations
  - [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl): explicit, implicit, and IMEX time steppers that take `FieldVector`s
  - [ClimaComms.jl](https://github.com/CliMA/ClimaComms.jl): the device and MPI abstraction the examples above switch with

Related packages worth knowing, in and out of Julia:

  - [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl): finite-volume ocean and non-hydrostatic fluid simulations on GPUs
  - [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl): spherical-harmonic atmospheric general circulation model
  - [Trixi.jl](https://github.com/trixi-framework/Trixi.jl): adaptive high-order discontinuous Galerkin for hyperbolic conservation laws
  - [Dedalus](https://dedalus-project.org/): Python framework for global spectral methods, equations entered as text

## Contributing

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance.
