# Running the 3D sphere examples on Caltech's central cluster

The examples in this directory are driven by `examples/hybrid/driver.jl`, which
selects a case through the `TEST_NAME` environment variable. They exist to
exercise ClimaCore's dycore — the hybrid spectral-element/finite-difference
sphere discretization, its implicit/explicit split, and hyperdiffusion — not to
run climate simulations. For forced-dissipative climate configurations
(Held-Suarez, aquaplanet, AMIP, ...), use
[ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl), which owns the physics
and its parameterizations.

## Running a case

```bash
#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00

module purge
module load climacommon

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export TEST_NAME=sphere/baroclinic_wave_rhoe
export OUTPUT_DIR=$YOUR_SIMULATION_OUTPUT_DIR
#export RESTART_FILE=$YOUR_JLD2_RESTART_FILE

CC=$HOME/ClimaCore.jl
julia --project=$CC/.buildkite -e 'using Pkg; Pkg.instantiate()'
julia --project=$CC/.buildkite --threads=8 $CC/examples/hybrid/driver.jl
```

Environment variables read by the driver:

* `TEST_NAME` (required): the case to run, e.g. `sphere/baroclinic_wave_rhoe`,
  `sphere/balanced_flow_rhoe`, or `plane/inertial_gravity_wave`.
* `OUTPUT_DIR`: where JLD2 output is written.
* `RESTART_FILE`: a JLD2 file from a previous run to restart from.
* `FLOAT_TYPE`: `Float32` (default) or `Float64`.
* `DISCRETIZATION`: `CG` (default) or `DG`, the horizontal Galerkin form.
  `DG` runs `sphere/baroclinic_wave_rhoe` with the flux-form momentum
  equation of `examples/hybrid/dg_tendency.jl` and an interface numerical flux
  in place of the DSS, and without hyperdiffusion — over a shorter run than
  the CG one, for the reason the case file gives. Output goes to a
  `_dg`-suffixed directory.
* `DG_FLUX`: the DG horizontal assembly — `kg-roe` (default), `kg-rusanov`
  (Kennedy-Gruber flux differencing with a Roe or Rusanov interface flux), or
  `rusanov` (weak-form volume divergence with a Rusanov interface flux). See
  `examples/hybrid/dg_tendency.jl`.
* `MOMENTUM_FORM`: `flux` (default on DG) or `vector_invariant` (the only
  form on CG). `DISCRETIZATION=DG MOMENTUM_FORM=vector_invariant` keeps the
  velocity equation of the CG form on a DG space, its weak-form derivatives
  completed by `Operators.complete_tendency!` with central fluxes and a
  velocity-jump penalty (`examples/hybrid/dg_vector_invariant_tendency.jl`);
  output goes to a `_dg_vi`-suffixed directory.
* `T_END`, `DT`, `ODE_ALGORITHM`: override the baroclinic waves' run length,
  timestep and ClimaTimeSteppers scheme (`SSP333`, IMEX, by default; an
  explicit scheme such as `SSP33ShuOsher` needs a timestep of a few seconds).

## Moist baroclinic wave

`sphere/moist_baroclinic_wave_rhoe` adds total water and 0-moment
microphysics to the DG flux form. It needs Thermodynamics.jl and
CloudMicrophysics.jl, which ClimaCore does not depend on, so it runs from the
examples environment. The Zhang-Shu positivity limiter
(`Limiters.PositivityLimiter`) keeps `q_tot`, density and pressure positive at
every stage; `POSITIVITY_LIMITER=0` turns it off, and `ZS_RHO_MIN` and
`ZS_P_MIN` set its floors.

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
TEST_NAME=sphere/moist_baroclinic_wave_rhoe DISCRETIZATION=DG \
    julia --project=examples examples/hybrid/driver.jl
```

Resolution, timestep, and output frequency are set in the case file itself
(e.g. `sphere/baroclinic_wave_rhoe.jl`); `dt_save_to_disk = FT(0)` disables
JLD2 output. The baroclinic wave picks its timestep from `DISCRETIZATION`,
the DG form needing a smaller one for its explicit horizontal terms.

## Remapping output to a lat/lon grid

To remap CG nodal output onto a regular lat/lon grid, use
[`ClimaCoreTempestRemap`](../../../lib/ClimaCoreTempestRemap/) directly; see its
test suite for worked examples of `overlap_mesh`, `remap_weights`, and
`apply_remap`.
