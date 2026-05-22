# ClimaCore.jl — Repo-Specific Guide

## Package overview

ClimaCore.jl provides the dynamical core infrastructure for [CliMA](https://clima.caltech.edu/)'s Earth System Model. It supplies flexible and composable discretization tools — spectral element and finite difference operators, field abstractions, data layouts, and GPU-portable kernels — that downstream packages (ClimaAtmos.jl, ClimaLand.jl, ClimaCoupler.jl) build on to solve the governing equations of their component models.

## Directory map

| Layer | Directory | Description |
|:---|:---|:---|
| Core module | `src/ClimaCore.jl` | Top-level module, re-exports all sub-modules |
| DataLayouts | `src/DataLayouts/` | Low-level array-of-structs / struct-of-arrays data storage backends |
| Geometry | `src/Geometry/` | Coordinate types, axis tensors, covariant/contravariant transforms |
| Domains | `src/Domains/` | Abstract domain definitions (intervals, rectangles, spheres) |
| Meshes | `src/Meshes/` | Mesh generation: interval, rectangle, cubed-sphere |
| Topologies | `src/Topologies/` | Distributed topologies, DSS (direct stiffness summation) connectivity |
| Quadratures | `src/Quadratures/` | Gauss–Legendre and Gauss–Lobatto quadrature rules |
| Grids | `src/Grids/` | Spectral element and finite-difference grid types |
| Spaces | `src/Spaces/` | Function spaces built on grids (spectral element, finite-difference, extruded) |
| Fields | `src/Fields/` | `Field` type — the primary user-facing data container on a space |
| Operators | `src/Operators/` | Spectral element and finite-difference differential operators |
| MatrixFields | `src/MatrixFields/` | Banded matrix fields for implicit vertical solvers |
| Hypsography | `src/Hypsography/` | Terrain-following coordinate transforms |
| Limiters | `src/Limiters/` | Flux limiters for transport |
| Remapping | `src/Remapping/` | Interpolation and remapping between spaces |
| InputOutput | `src/InputOutput/` | HDF5-based checkpointing and restart I/O |
| CommonGrids | `src/CommonGrids/` | Pre-built convenience grid constructors |
| CommonSpaces | `src/CommonSpaces/` | Pre-built convenience space constructors |
| Utilities | `src/Utilities/` | Internal utilities (PlusHalf indexing, AutoBroadcaster, etc.) |
| DebugOnly | `src/DebugOnly/` | Debug-mode-only utilities |
| CUDA ext | `ext/ClimaCoreCUDAExt.jl`, `ext/cuda/` | CUDA GPU extension (loaded via Pkg extensions) |
| Krylov ext | `ext/KrylovExt.jl` | Krylov.jl integration for iterative solvers |
| Lib: Plots | `lib/ClimaCorePlots/` | Plots.jl recipes for ClimaCore fields |
| Lib: Makie | `lib/ClimaCoreMakie/` | Makie.jl recipes for ClimaCore fields |
| Lib: VTK | `lib/ClimaCoreVTK/` | VTK output for visualization |
| Lib: TempestRemap | `lib/ClimaCoreTempestRemap/` | TempestRemap bindings for conservative remapping |
| Lib: Spectra | `lib/ClimaCoreSpectra/` | Spectral analysis of fields on the sphere |

## Key abstractions

1. **`Field`** (`src/Fields/`) — the primary data type. A field wraps data on a space and supports broadcast, reductions, and operator application.
2. **`Space`** (`src/Spaces/`) — represents a discretized function space (spectral element, finite-difference, or extruded hybrid). Constructed from a grid and a quadrature rule.
3. **Operators** (`src/Operators/`) — lazy differential operators (gradient, divergence, curl, interpolation, restriction) that compose via Julia's broadcast system.
4. **`DataLayout`** (`src/DataLayouts/`) — the storage backends (IJFH, VIJFH, VF, etc.) that determine memory layout for CPU vs GPU performance.
5. **`MatrixFields`** (`src/MatrixFields/`) — banded-matrix field algebra used for implicit vertical solvers and Jacobian construction.

## Test groups

Tests are defined in `test/runtests.jl` using the `UnitTest` / `tabulated_tests` framework:

| Group | What it covers |
|:---|:---|
| CPU unit tests | 104 tests covering DataLayouts, Geometry, Meshes, Topologies, Quadratures, Spaces, Fields, Operators (spectral element + finite-difference), MatrixFields, Hypsography, Limiters, Remapping, InputOutput, Aqua, deprecations |
| GPU tests (`:gpu_only`) | 9 tests: CUDA kernels, compiler stress regression, DataLayout GPU ops, spectral element CUDA, finite-difference CUDA, extruded sphere/3dbox CUDA, field map-reduce CUDA |
| Buildkite CI | Heavyweight GPU integration tests on HPC cluster, defined in `.buildkite/pipeline.yml` |
| Lib CI workflows | Separate GitHub Actions per companion package: ClimaCoreMakie, ClimaCorePlots, ClimaCoreSpectra, ClimaCoreTempestRemap, ClimaCoreVTK |

## Repo-specific conventions

- **Module-per-directory**: each `src/` subdirectory is its own Julia sub-module, re-exported from `ClimaCore.jl`.
- **`lib/` companion packages**: visualization and remapping packages live as independent Julia packages under `lib/`, each with its own `Project.toml`. They have separate CI workflows.
- **`ext/` CUDA pattern**: GPU support uses Julia's package extension mechanism (`ext/ClimaCoreCUDAExt.jl`). CPU fallbacks are always provided.
- **Coding style**: `TitleCase` for types, `snake_case` for objects/functions, spaces after commas. Formatting follows [YASGuide](https://github.com/jrevels/YASGuide) loosely, enforced via `JuliaFormatter` (v1.0.62) in CI.
- **ColPrac**: the project follows the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices.
- **Tabulated test runner**: tests use the custom `UnitTest` struct and `run_unit_tests!` / `tabulate_tests` helpers from `test/tabulated_tests.jl`, with leak detection enabled.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
