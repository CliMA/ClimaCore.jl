# ClimaCore examples

Runnable dynamical-core configurations built on ClimaCore. They exist to
exercise and demonstrate the library — discretizations, operators, limiters, and
the implicit/explicit split — not to run climate simulations. For
forced-dissipative climate configurations (Held-Suarez, aquaplanet, AMIP, ...),
use [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), which owns the
physics and its parameterizations.

Every example here is run in CI (see `.buildkite/pipeline.yml`), so they are kept
working as the library changes.

## Layout

Examples are grouped by the geometry they are discretized on:

| Directory | Discretization                                                              |
|-----------|-----------------------------------------------------------------------------|
| `column/` | 1D vertical column, finite difference.                                      |
| `plane/`  | 2D horizontal plane, spectral element.                                      |
| `sphere/` | 2D cubed sphere, spectral element (shallow water and advection).            |
| `hybrid/` | Extruded domains: horizontal spectral element × vertical finite difference. |

`hybrid/` is subdivided by its horizontal domain — `plane/` (2D *x*–*z* slice),
`box/` (3D Cartesian), and `sphere/` (3D cubed sphere) — and holds the shared
model code those cases build on:

* `staggered_nonhydrostatic_model.jl`: the tendency for the staggered
  nonhydrostatic equations, in terms of density, total energy, and momentum.
* `implicit_equation_jacobian.jl`, `hyperdiffusion.jl`: the vertical implicit
  solve and the hyperdiffusion used by that model.
* `driver.jl`, `ode_config.jl`: a driver that selects a case through the
  `TEST_NAME` environment variable, and its timestepper configuration.

Files named `*_utils.jl` hold setup shared by the cases beside them (spaces,
constants, initial conditions) and define no top-level behavior of their own.

## Running an example

The examples run under the `.buildkite` project, which carries the plotting and
timestepping dependencies that ClimaCore itself does not:

```
julia --project=.buildkite -e 'using Pkg; Pkg.instantiate()'
julia --project=.buildkite examples/plane/bickleyjet_cg.jl
```

Or from the REPL:

```
julia --project=.buildkite
julia> include("examples/plane/bickleyjet_cg.jl")
```

Most examples write plots and animations to an `output/` directory beside the
example.

The `hybrid/` cases listed under `TEST_NAME` are run through `driver.jl` rather
than directly, since the driver defines the constants and spaces they expect:

```
TEST_NAME=sphere/baroclinic_wave_rhoe julia --project=.buildkite examples/hybrid/driver.jl
```

See [`hybrid/sphere/README.md`](hybrid/sphere/README.md) for the environment
variables the driver reads and for running on a cluster.
