<div align="center">
  <img src="docs/src/assets/logo.svg" alt="ClimaCore.jl Logo" width="128" height="128">
</div>

# ClimaCore.jl

The dynamical core (_dycore_) of the CliMA Earth System Model: composable, GPU-capable tools for the spatial discretization of partial differential equations on the sphere and in Cartesian domains.

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

ClimaCore.jl is a library for the spatial discretization of partial
differential equations on the sphere, in boxes, in vertical slices, in
horizontal planes, and in single columns. It provides spaces, fields, and
composable differential operators (continuous or discontinuous spectral
elements in the horizontal, staggered finite differences in the vertical)
that can be broadcast over fields. The same source runs on a laptop, on an
NVIDIA GPU, and across MPI ranks.

It is the dynamical core underneath [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl)
and [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), and a standalone
package for fluid dynamics beyond climate modeling. Time integration is
provided by [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl),
which takes ClimaCore fields as its state vector.

## Who this is for

  - **Writing a model.** ClimaCore provides spaces, fields, and operators for
    spatial PDE discretization. Complex models can be built on it, e.g.,
    CliMA's atmosphere model
    [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), which adds
    physics, diagnostics, and calibration on top.
  - **Geophysical fluid dynamics.** Shallow water on the sphere, barotropic
    instability, mountain waves, gravity waves, density currents, rising
    thermals, and tracer transport under prescribed winds are included as
    tested programs in [`examples/`](examples/).
  - **Dynamical-core development.** The prognostic variables, the form of the
    equations, monotone limiters, the implicit/explicit split, and the
    vertical staggering are all choices made in user code, so each can be
    changed independently.
  - **GPU without writing kernels.** A broadcast expression over fields
    compiles to one fused kernel; devices can be switched by changing an
    environment variable.

The horizontal discretization uses spectral elements with Gauss–Legendre–Lobatto
points (Gauss–Legendre points are also available). Continuous and discontinuous
Galerkin methods are supported. The vertical discretization uses finite
differences on a Lorenz-staggered grid (vertical velocities on faces, other
variables on centers), to eliminate computational modes that can be deleterious
when horizontal scales are much larger than vertical.

## Features

  - **Spectral-element horizontal discretizations**: continuous (CG) and discontinuous (DG) Galerkin spectral elements, on the cubed sphere and on Cartesian planes.
  - **Staggered vertical discretization**: finite differences on center/face grids, with stretching and terrain-following (hypsography) coordinates, and upwind and limited transport operators.
  - **Multiple geometries**: Cartesian and spherical domains, with governing equations expressed in covariant vectors for curvilinear systems and Cartesian vectors for Euclidean spaces. The metric terms are applied implicitly by the operators.
  - **`Field` abstraction**: scalar-, vector-, or struct-valued fields carrying values, geometry, and mesh information, with flexible memory layouts and useful overloads (`sum`, `norm`, ...).
  - **Composable operators via broadcasting**: differential operators (`grad`, `div`, `curl`, `interpolate`, ...) act like functions when broadcast over a `Field`, fusing operators and function calls into a single pass.
  - **Implicit solvers for the vertical**: `MatrixFields` builds banded Jacobians column by column and solves them, as needed for IMEX time stepping of vertical acoustic and diffusive terms.
  - **GPU acceleration**: broadcast expressions compile to CUDA kernels.
  - **Distributed by construction**: element topologies carry their halo exchange, so the same tendency function runs on one rank or hundreds.
  - **Single and double precision**, and compatibility with automatic differentiation via dual numbers (e.g. [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)).
  - **I/O and remapping**: HDF5 checkpointing of fields and spaces (serial and parallel), and interpolation onto uniform lat–long–z grids for plotting and diagnostics.

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

# The bell returns to where it started, and the flux-form transport conserves
# mass to roundoff.
@assert sum(sol.u[end]) ≈ sum(h)
```

The metric terms come from the space, so the divergence knows it is on a
sphere. The same script runs on a GPU or across MPI ranks by setting the
device and communication context through
[ClimaComms.jl](https://github.com/CliMA/ClimaComms.jl):

```bash
julia --project my_solver.jl                                     # CPU
CLIMACOMMS_DEVICE=CUDA julia --project my_solver.jl              # one GPU
CLIMACOMMS_CONTEXT=MPI CLIMACOMMS_DEVICE=CUDA \
    srun --ntasks=4 julia --project my_solver.jl                 # four GPUs
```

## Examples

[`examples/`](examples/) contains example cases and explains how to run them.
The [examples documentation](https://CliMA.github.io/ClimaCore.jl/dev/examples/)
derives the
equations and discretizations for several of them.

## Documentation

  - **[Stable docs](https://CliMA.github.io/ClimaCore.jl/stable/)**: installation, introduction, mathematical framework, and API reference
  - **[Dev docs](https://CliMA.github.io/ClimaCore.jl/dev/)**: latest development version

## Contributing

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance.
