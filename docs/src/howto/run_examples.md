# Run the examples

The [`examples/`](https://github.com/CliMA/ClimaCore.jl/tree/main/examples) directory holds runnable dynamical-core configurations that
exercise the library: column, plane, sphere, and extruded (hybrid) cases. They
run in continuous integration, so they track the current API. Climate
configurations with physics (Held–Suarez, aquaplanet, AMIP) live in
[ClimaAtmos.jl](https://clima.github.io/ClimaAtmos.jl/stable/), not here.

## Prerequisites

A clone of the repository and an environment that carries the packages the
examples import beyond ClimaCore: `ClimaComms` and `ClimaTimeSteppers` for
every case, `Plots` for the plots and animations most cases write (its recipes come from
ClimaCore's `ClimaCoreRecipesBaseExt` extension, activated when `Plots` is loaded), and `TerminalLoggers` for progress bars. A few of the extruded
cases add `JLD2`, `NCDatasets`, `QuadGK`, or `Colors`.

## Steps

 1. From the repository root, build a shared environment once with those
    packages and the ClimaCore checkout:

    ```bash
    julia --project=@climacore-examples -e '
        using Pkg
        Pkg.develop(path = ".")                     # this checkout; or Pkg.add("ClimaCore")
        Pkg.add(["ClimaComms", "ClimaTimeSteppers", "Plots", "TerminalLoggers",
                 "StaticArrays", "LazyBroadcast", "Test"])'
    ```

    `@climacore-examples` is a named shared environment that Julia stores under
    `~/.julia/environments/`; a directory path works the same way.

 2. Run a stand-alone example:

    ```bash
    julia --project=@climacore-examples examples/plane/bickleyjet.jl cg
    ```

    Most examples write plots and animations to an `output/` directory beside
    the script.

 3. Run one of the extruded cases through the driver, which defines the
    constants and spaces they expect and selects the case from `TEST_NAME`:

    ```bash
    TEST_NAME=sphere/baroclinic_wave_rhoe julia --project=@climacore-examples examples/hybrid/driver.jl
    ```

    [`examples/hybrid/sphere/README.md`](https://github.com/CliMA/ClimaCore.jl/blob/main/examples/hybrid/sphere/README.md) lists the environment variables the
    driver reads and how to run it on a cluster.

The `import` lines at the top of a script list anything further it needs. Add
those packages to the environment and run the script the same way.

## Layout

| Directory | Discretization                                                              |
|:--------- |:--------------------------------------------------------------------------- |
| `column/` | 1D vertical column, finite difference.                                      |
| `plane/`  | 2D horizontal plane, spectral element (CG and DG Bickley jets).             |
| `sphere/` | 2D cubed sphere, spectral element (shallow water and advection).            |
| `hybrid/` | Extruded domains: horizontal spectral element × vertical finite difference. |

The equations each case solves and the operators it uses are described on the
[Example gallery](../explanation/examples.md) page.
