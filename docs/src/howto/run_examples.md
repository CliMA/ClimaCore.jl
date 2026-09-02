# Run the examples

The `examples/` directory holds runnable dynamical-core configurations that
exercise the library: column, plane, sphere, and extruded (hybrid) cases. They
are run in CI, so they track the current API. Climate configurations with
physics (Held–Suarez, aquaplanet, AMIP) live in
[ClimaAtmos.jl](https://clima.github.io/ClimaAtmos.jl/stable/), not here.

## Prerequisites

A clone of the repository. The examples run under the `.buildkite` project,
which carries the plotting and time-stepping dependencies that ClimaCore itself
does not.

## Steps

1. Instantiate the `.buildkite` environment once, from the repository root:

   ```bash
   julia --project=.buildkite -e 'using Pkg; Pkg.instantiate()'
   ```

2. Run a stand-alone example:

   ```bash
   julia --project=.buildkite examples/plane/bickleyjet_cg.jl
   ```

   Most examples write plots and animations to an `output/` directory beside
   the script.

3. Run one of the extruded cases through the driver, which defines the
   constants and spaces they expect and selects the case from `TEST_NAME`:

   ```bash
   TEST_NAME=sphere/baroclinic_wave_rhoe julia --project=.buildkite examples/hybrid/driver.jl
   ```

   `examples/hybrid/sphere/README.md` lists the environment variables the
   driver reads and how to run it on a cluster.

## Layout

| Directory | Discretization                                                              |
|:----------|:----------------------------------------------------------------------------|
| `column/` | 1D vertical column, finite difference.                                      |
| `plane/`  | 2D horizontal plane, spectral element (CG and DG Bickley jets).             |
| `sphere/` | 2D cubed sphere, spectral element (shallow water and advection).            |
| `hybrid/` | Extruded domains: horizontal spectral element × vertical finite difference. |

The equations each case solves and the operators it uses are described on the
[Example gallery](../explanation/examples.md) page.
