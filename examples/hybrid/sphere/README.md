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
* `DISCRETIZATION`: `CG` (default) or `DG` (interface numerical fluxes, no
  hyperdiffusion; output in a `_dg`-suffixed directory).
* `DG_FLUX`: `kg-roe` (default), `kg-rusanov` (Kennedy-Gruber flux
  differencing with a Roe or Rusanov interface flux) or `rusanov` (weak form).
* `MOMENTUM_FORM`: on DG, `flux` (default; `ρuₕ`, `ρw`, `dg_tendency.jl`) or
  `vector_invariant` (`uₕ`, `w`, `dg_vector_invariant_tendency.jl`; output in
  a `_dg_vi`-suffixed directory).
* `H_ELEM`, `Z_ELEM`, `T_END`, `DT`, `ODE_ALGORITHM`: resolution, run length,
  timestep and scheme of the baroclinic waves (`SSP333` by default).

## Moist baroclinic wave

`sphere/moist_baroclinic_wave_rhoe` adds total water, 0-moment microphysics
and a Zhang-Shu positivity limiter (`POSITIVITY_LIMITER=0` turns it off) to the
DG flux form. It needs Thermodynamics.jl and CloudMicrophysics.jl, so it runs
from the examples environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
TEST_NAME=sphere/moist_baroclinic_wave_rhoe DISCRETIZATION=DG \
    julia --project=examples examples/hybrid/driver.jl
```

Defaults are set in the case files; `dt_save_to_disk = FT(0)` disables JLD2
output.

## Remapping output to a lat/lon grid

To remap CG nodal output onto a regular lat/lon grid, use
[`ClimaCoreTempestRemap`](../../../lib/ClimaCoreTempestRemap/) directly; see its
test suite for worked examples of `overlap_mesh`, `remap_weights`, and
`apply_remap`.
