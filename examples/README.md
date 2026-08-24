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

## Cases

Each case asserts what it is supposed to show, so it doubles as a starting
point and as a regression test. The [documentation](https://CliMA.github.io/ClimaCore.jl/dev/examples/)
derives the equations and discretizations for several of them.

| Case | What it exercises | File |
|:--|:--|:--|
| Heat equation, Ekman spiral, hydrostatic balance, limited advection | staggered vertical operators with boundary conditions | [`column/`](column/) |
| Solid-body tracer transport | horizontal transport, cubed-sphere panel edges | [`sphere/solid_body_rotation.jl`](sphere/solid_body_rotation.jl) |
| Shallow-water suite, Williamson et al. (1992) | vector-invariant form, hyperviscosity, conservation | [`sphere/shallow_water.jl`](sphere/shallow_water.jl) |
| Bickley jet, CG and DG | barotropic instability, energy conservation, over-integration | [`plane/bickleyjet_cg.jl`](plane/bickleyjet_cg.jl), [`plane/bickleyjet_dg.jl`](plane/bickleyjet_dg.jl) |
| Deformational flow and Hadley circulation (DCMIP 2012) | filamentation, limiters, vertical–horizontal coupling | [`hybrid/sphere/deformation_flow.jl`](hybrid/sphere/deformation_flow.jl), [`hadley_circulation.jl`](hybrid/sphere/hadley_circulation.jl) |
| Rising thermal bubble, 2D and 3D | nonhydrostatic compressible flow in a box | [`hybrid/box/bubble_3d_invariant_rhoe.jl`](hybrid/box/bubble_3d_invariant_rhoe.jl) |
| Density current | advection at a rolling-up front (Kelvin–Helmholtz billows) | [`hybrid/plane/density_current_2d_invariant_rhoe.jl`](hybrid/plane/density_current_2d_invariant_rhoe.jl) |
| Mountain waves, Schär et al. (2002) and witch-of-Agnesi | terrain-following metric terms, sponge layers | [`hybrid/plane/schar_mountain.jl`](hybrid/plane/schar_mountain.jl) |
| Inertial and nonhydrostatic gravity waves | IMEX splitting against linear analytic solutions | [`hybrid/plane/inertial_gravity_wave.jl`](hybrid/plane/inertial_gravity_wave.jl) |
| Baroclinic wave, Ullrich et al. (2014) | the standard dry dynamical-core benchmark | [`hybrid/sphere/baroclinic_wave_rhoe.jl`](hybrid/sphere/baroclinic_wave_rhoe.jl) |

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
