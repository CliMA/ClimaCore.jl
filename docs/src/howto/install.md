# Install ClimaCore

ClimaCore.jl is a registered Julia package and requires Julia 1.10 or later.
[juliaup](https://github.com/JuliaLang/juliaup) installs and updates Julia.

## Install the registered release

In a Julia session, from the project environment you want to use:

```julia
using Pkg
Pkg.add("ClimaCore")
```

## Install the development version

The `main` branch carries changes that have not been released:

```julia
using Pkg
Pkg.add(url = "https://github.com/CliMA/ClimaCore.jl", rev = "main")
```

## Run the test suite

```julia
using Pkg
Pkg.test("ClimaCore")
```

The suite runs for a long time on a laptop; see `test/runtests.jl` for how to
select a subset.

## Enable GPU or MPI backends

ClimaCore reads its compute device and communication context from
[ClimaComms.jl](https://clima.github.io/ClimaComms.jl/stable/), which loads
CUDA.jl and MPI.jl on demand. Add the backend packages to your default
environment once, then select them per run with environment variables:

```bash
CLIMACOMMS_DEVICE=CUDA julia --project script.jl
CLIMACOMMS_CONTEXT=MPI mpiexec -n 4 julia --project script.jl
```

Every script that should honor these variables starts with

```julia
import ClimaComms
ClimaComms.@import_required_backends
```

The ClimaComms [how-to guides](https://clima.github.io/ClimaComms.jl/dev/howto/)
describe the environment variables and backend loading; the shared developer
guide
[running\_on\_gpu.md](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/workflow/running_on_gpu.md)
covers first-time machine setup, including CUDA driver and runtime
compatibility.
