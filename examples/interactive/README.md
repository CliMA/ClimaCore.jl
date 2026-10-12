# Interactive demo: stratified flow over terrain you draw

A desktop demo in the spirit of
[Dan Schroeder's fluid dynamics simulation](https://physics.weber.edu/schroeder/fluids/),
built on the physics ClimaCore is designed for: a dry, stratified,
nonhydrostatic atmosphere flowing over a mountain. Draw a terrain profile with
the mouse, press run, and watch mountain waves form; change the wind or the
stratification to move between linear waves, wave breaking, and upstream
blocking.

The dynamics are those of `examples/hybrid/plane/schar_mountain.jl`
(vector-invariant momentum, total energy, SSP33 time stepping) on a hybrid
spectral-element × finite-difference space with a terrain-following
coordinate.

## Running

Needs a desktop with OpenGL (GLMakie). From the repository root:

```
julia --project=examples/interactive -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples/interactive examples/interactive/run.jl
```

Optional arguments select the grid (`coarse`, `fine`, `finest`) and the
initial terrain preset (`flat`, `agnesi`, `schar`, `plateau`, `twin`):

```
julia --project=examples/interactive examples/interactive/run.jl fine schar
```

The headless driver runs a preset for a fixed time, prints cost and
conservation, and writes an mp4 to `examples/interactive/output/`:

```
julia --project=examples/interactive examples/interactive/run_offline.jl schar 1
```

## Controls

| Control            | Effect                                                                 |
|--------------------|------------------------------------------------------------------------|
| Run / Pause        | start or stop stepping                                                 |
| Reset              | back to the initial state on the current terrain                       |
| terrain preset     | load a preset profile and restart                                      |
| profile axis       | click-drag to draw terrain; the model restarts when the mouse is released |
| plot               | diagnostic shown: `w`, `u`, `u′`, `θ`, `θ′`, `p′`, dye `q`              |
| wind U, N          | background wind and Brunt–Väisälä frequency; applied on reset           |
| diffusion ×        | scales the second- and fourth-order diffusivities; applied live        |
| steps/frame        | model steps between redraws (model speed vs. smoothness)               |
| contrast           | shrinks the colour range                                               |
| tracers            | show passive particles advected by the resolved wind; reseed resets them to a grid |
| resolution         | rebuilds the model on another grid                                     |

The regime is set by the nondimensional mountain height `N h / U`, shown in
the status label: below about 0.5 the response is linear mountain waves, near
1 the waves steepen and break aloft, above about 1.5 the low-level flow is
blocked upstream.

## Files

- `model.jl` — the model (`TerrainFlow` module): spaces on a drawn terrain,
  initial state, tendency, surface constraint, integrator, diagnostics. No
  Makie dependency.
- `gui.jl` — the GLMakie window.
- `run.jl` — entry point. `run_offline.jl` — headless driver.
