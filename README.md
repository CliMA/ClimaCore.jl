# ClimaCore.jl

|||
|---------------------:|:----------------------------------------------|
| **Docs Build**       | [![docs build][docs-bld-img]][docs-bld-url]   |
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **GHA CI**           | [![gha ci][gha-ci-img]][gha-ci-url]           |
| **Buildkite CI**     | [![buildkite ci][buildkite-ci-img]][buildkite-ci-url] |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |
| **Downloads**        | [![downloads][downloads-img]][downloads-url]  |
| **DOI**              | [![zenodo][zenodo-img]][zenodo-url]           |

[docs-bld-img]: https://github.com/CliMA/ClimaCore.jl/workflows/Documentation/badge.svg
[docs-bld-url]: https://github.com/CliMA/ClimaCore.jl/actions?query=workflow%3ADocumentation

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/ClimaCore.jl/dev/

[gha-ci-img]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml/badge.svg
[gha-ci-url]: https://github.com/CliMA/ClimaCore.jl/actions/workflows/UnitTests.yml

[buildkite-ci-img]: https://badge.buildkite.com/2b63d3c49347804f61bd8e99c8b85e05871253b92612cd1af4.svg
[buildkite-ci-url]: https://buildkite.com/clima/climacore-ci

[codecov-img]: https://codecov.io/gh/CliMA/ClimaCore.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/ClimaCore.jl

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FClimaCore&query=total_requests&suffix=%2Ftotal&label=Downloads
[downloads-url]: http://juliapkgstats.com/pkg/ClimaCore

[zenodo-img]: https://zenodo.org/badge/356355994.svg
[zenodo-url]: https://zenodo.org/badge/latestdoi/356355994

The Climate Modelling Alliance ([CliMA](https://clima.caltech.edu/)) is developing a new Earth System Model (ESM), entirely written in the [Julia](https://julialang.org/) language. The main goal of the project is to build an ESM that automatically learns from diverse data sources to produce accurate climate predictions with quantified uncertainties. The CliMA model targets both CPU and GPU architectures, using a common codebase. ClimaCore.jl constitutes the dynamical core (_dycore_) of the atmosphere and land models, providing discretization tools to solve the governing equations of the ESM component models.
ClimaCore.jl's high-level application programming interface (API) facilitates modularity and composition of differential operators and the definition of flexible discretizations. This, in turn, is coupled with low-level APIs that support different data layouts, specialized implementations, and flexible models for threading, to better face high-performance optimization, data storage, and scalability challenges on modern HPC architectures.

## Technical aims and current support
* Support both large-eddy simulation (LES) and general circulation model (GCM) configurations for the atmosphere.
* A suite of tools for constructing space discretizations.
* Horizontal spectral elements:
    - Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) spectral element discretizations.
* Flexible choice of vertical discretization (currently staggered finite differences)
* Support for different geometries (Cartesian, spherical), with governing equations discretized in terms of covariant  vectors for curvilinear, non-orthogonal systems and Cartesian vectors for Euclidean spaces.
* `Field` abstraction:
    - Scalar, vector or struct-valued
    - Stores values, geometry, and mesh information
    - Flexible memory layouts: Array-of-Structs (AoS), Struct-of-Arrays (SoA),Array-of-Struct-of-Arrays (AoSoA)
    - Useful overloads: `sum` (integral), `norm`, etc.
    - Compatible with [`DifferentialEquations.jl`](https://diffeq.sciml.ai/stable/) time steppers.
* Composable operators via broadcasting: apply a function element-wise to an array; scalar values are broadcast over arrays
* Fusion of multiple operations; can be specialized for custom functions or argument types (e.g. `CuArray` compiles and applies a custom CUDA kernel).
* Operators (`grad`, `div`, `interpolate`) are “pseudo-functions”: Act like functions when broadcasted over a `Field`; fuse operators and function calls.
* Add element node size dimensions to type domain
    - i.e., specialize on polynomial degree
    - important for GPU kernel performance.
* Flexible memory layouts allow for flexible threading models (upcoming):
    - CPU thread over elements
    - GPU thread over nodes.

Versions before and including ClimaCore.jl v0.11.7 relied on WeakValueDicts.jl, which is not thread-safe and no longer maintained.
These versions are considered unsupported, and newer versions of ClimaCore.jl should be used.
