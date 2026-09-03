<div align="center">
  <img src="docs/src/assets/logo.svg" alt="ClimaCore.jl Logo" width="128" height="128">
</div>

# ClimaCore.jl

The dynamical core (_dycore_) of the CliMA Earth System Model: composable, performance-portable tools for discretizing partial differential equations on the sphere and in Cartesian domains.

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

ClimaCore.jl is the spatial discretization layer of the [Climate Modeling Alliance (CliMA)](https://clima.caltech.edu/) Earth System Model, written entirely in [Julia](https://julialang.org/). It supplies the grids, fields, and differential operators that [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) and [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl) build their equations on, and time steps them with [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl). Configurations range from a single column to large-eddy simulation on a box to a global cubed sphere, and the same model code runs on a CPU, on many nodes through MPI, and on a GPU.

## Features

- **Horizontal spectral elements**: continuous (CG) and discontinuous (DG) Galerkin discretizations on quadrilateral elements, selected by one keyword and completed across element boundaries by direct stiffness summation (CG) or numerical fluxes (DG).
- **Staggered vertical finite differences**: Lorenz staggering on cell centers and faces, with center-to-face and face-to-center operators and their boundary conditions.
- **Upwinding and limiters**: upwind-biased, FCT, and TVD reconstructions for advection, plus the quasi-monotone horizontal limiter and the vertical mass-borrowing limiter for positivity.
- **Curvilinear geometry**: covariant and contravariant bases, metric terms, and terrain-following coordinates, so operators are written once and evaluated on Cartesian and spherical domains alike.
- **Matrix-free operators via broadcasting**: differential operators act like functions when broadcast over a `Field`, fusing operators and function calls into a single pass and compiling to one CPU loop or one CUDA kernel.
- **Performance portability and scaling**: one codebase runs on CPUs and GPUs and distributes over MPI, with weak-scaling efficiency above 92% on GPUs and above 98% on CPUs, and 0.20 simulated years per day at 6 km resolution on 256 H100 GPUs ([Yatunin et al. 2026](https://CliMA.github.io/ClimaCore.jl/stable/explanation/performance/)).
- **Differentiability**: fields and operators carry `ForwardDiff` dual numbers, so a column tendency can be differentiated for Jacobians and calibration.
- **Time-stepper compatible**: `Field`s and `FieldVector`s are the state vector for [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl).

## Installation

ClimaCore.jl is a registered Julia package (Julia 1.10 or later):

```julia
using Pkg
Pkg.add("ClimaCore")
```

## Quick Example

```julia
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Domains, Meshes, Spaces, Fields, Geometry, Operators

FT = Float64

# Build a 1D column: interval domain -> mesh -> finite-difference space
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(2π),
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain; nelems = 128)
space = Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), mesh)

# Define a field over the space and differentiate it with a composed operator
z = Fields.coordinate_field(space).z
θ = sin.(z)
grad = Operators.GradientC2F(
    bottom = Operators.SetValue(FT(0)),
    top = Operators.SetValue(FT(0)),
)
∂θ = @. Geometry.WVector(grad(θ))   # face-valued vertical gradient (≈ cos(z))
```

This snippet is the [Home page](https://CliMA.github.io/ClimaCore.jl/stable/) example, which the docs build runs on every commit. More runnable examples (column, plane, and sphere configurations) are in the [`examples/`](examples/) directory.

## Documentation

- **[Stable docs](https://CliMA.github.io/ClimaCore.jl/stable/)** — tutorials, how-to guides, explanation of the numerics, and API reference
- **[Dev docs](https://CliMA.github.io/ClimaCore.jl/dev/)** — latest development version
- **[`examples/`](examples/)** — runnable examples across geometries

## Integration with CliMA models

ClimaCore.jl is the dynamical core used throughout the [CliMA](https://github.com/CliMA) ecosystem, including:

- [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) — atmosphere model
- [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl) — land model

Device and communication backends come from [ClimaComms.jl](https://github.com/CliMA/ClimaComms.jl), and time integration from [ClimaTimeSteppers.jl](https://github.com/CliMA/ClimaTimeSteppers.jl).

## Contributing

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance.
