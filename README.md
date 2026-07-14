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

[gha-ci-img]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml

[bk-ci-img]: https://badge.buildkite.com/2b63d3c49347804f61bd8e99c8b85e05871253b92612cd1af4.svg?branch=main
[bk-ci-url]: https://buildkite.com/clima/climacore-ci/builds?branch=main

[codecov-img]: https://codecov.io/gh/CliMA/ClimaCore.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaCore.jl

[dlt-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaCore&query=total_requests&label=Downloads
[dlt-url]: https://juliapkgstats.com/pkg/ClimaCore

[zenodo-img]: https://zenodo.org/badge/356355994.svg
[zenodo-url]: https://zenodo.org/badge/latestdoi/356355994

ClimaCore.jl provides the spatial discretization building blocks for the [Climate Modeling Alliance (CliMA)](https://clima.caltech.edu/) Earth System Model, which is written entirely in [Julia](https://julialang.org/). It pairs a high-level API for composing differential operators and defining flexible discretizations with low-level APIs for data layouts, specialized implementations, and threading — targeting both CPU and GPU architectures from a single codebase.

## Features

- **Spectral-element horizontal discretizations**: continuous (CG) and discontinuous (DG) Galerkin spectral elements.
- **Flexible vertical discretization**: staggered finite differences on center/face grids.
- **Multiple geometries**: Cartesian and spherical domains, with governing equations expressed in covariant vectors for curvilinear systems and Cartesian vectors for Euclidean spaces.
- **`Field` abstraction**: scalar-, vector-, or struct-valued fields carrying values, geometry, and mesh information, with flexible memory layouts (AoS, SoA, AoSoA) and useful overloads (`sum`, `norm`, ...).
- **Composable operators via broadcasting**: differential operators (`grad`, `div`, `interpolate`, ...) act like functions when broadcast over a `Field`, fusing operators and function calls into a single pass.
- **GPU acceleration**: broadcast expressions compile to custom CUDA kernels, with specialization on polynomial degree for kernel performance.
- **Time-stepper compatible**: works with [SciML](https://sciml.ai/)/OrdinaryDiffEq time steppers.

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

More runnable examples (column, plane, and sphere configurations) are in the [`examples/`](examples/) directory.

## Documentation

- **[Stable docs](https://CliMA.github.io/ClimaCore.jl/stable/)** — installation, introduction, mathematical framework, and API reference
- **[Dev docs](https://CliMA.github.io/ClimaCore.jl/dev/)** — latest development version
- **[`examples/`](examples/)** — runnable examples across geometries

## Integration with CliMA models

ClimaCore.jl is the dynamical core used throughout the [CliMA](https://github.com/CliMA) ecosystem, including:

- [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) — atmosphere model
- [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl) — land model

## Contributing

Contributors should follow the shared CliMA engineering standards in [`docs/dev-guides/`](docs/dev-guides/), which cover architecture, performance, code quality, documentation, and workflows. These are vendored from [CliMA/DeveloperGuides](https://github.com/CliMA/DeveloperGuides). The repo's [`AGENTS.md`](AGENTS.md) is a starting point for AI agents with repo-specific guidance.
