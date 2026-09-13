# Run on a GPU

A ClimaCore script runs on an NVIDIA GPU without code changes: the device is
read from the environment, and every grid, field, and operator dispatches on
it. This page covers what is specific to ClimaCore; the environment variables
and backend loading are ClimaComms's, documented in its
[how-to guides](https://clima.github.io/ClimaComms.jl/dev/howto/), and
first-time machine setup (CUDA driver and runtime compatibility) is in the
shared developer guide
[running\_on\_gpu.md](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/workflow/running_on_gpu.md).

## Prerequisites

  - CUDA.jl in your default Julia environment (`import Pkg; Pkg.add("CUDA")`
    from a plain `julia` session). It is a weak dependency of ClimaCore: the
    `ClimaCoreCUDAExt` extension loads when both are present.
  - A script that begins with `import ClimaComms; ClimaComms.@import_required_backends`,
    which loads CUDA.jl when the environment asks for it.

## Steps

 1. Select the device for the run:

    ```bash
    CLIMACOMMS_DEVICE=CUDA julia --project script.jl
    ```

 2. Build spaces from `ClimaComms.device()` (or `ClimaComms.context()`), not
    from a hard-coded device. The `CommonSpaces` constructors do this by
    default; the low-level constructors take the device or context explicitly:

    ```julia
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    topology = Topologies.Topology2D(context, mesh)
    column_space = Spaces.CenterFiniteDifferenceSpace(device, mesh)
    ```

    Every field allocated on such a space lives in GPU memory
    (`parent(field)` is a `CuArray`), and broadcast expressions over it compile
    to CUDA kernels.

 3. Keep kernels free of scalar indexing and allocation. A broadcast over
    fields is a kernel; a loop over `parent(field)` runs on the host, and
    indexing a device array element from the host raises an error. Reductions
    (`sum`, `maximum`) run on the device and return a scalar to the host.

 4. Move data when a host-side library needs it. `ClimaCore.to_cpu(x)` returns
    a copy of a field, field vector, or space on the CPU, and
    `ClimaCore.to_device(device, x)` moves it back
    ([Move data between CPU and GPU](to_device.md)). `Remapping.interpolate`
    returns a host array, so plotting needs no explicit transfer.

## Check the device

```julia
ClimaComms.device()                 # CUDADevice()
ClimaComms.device(space)            # the device the space was built on
typeof(parent(field))               # CuArray{Float64, ...}
```

## Tuning

Four environment variables adjust the kernel configuration; the defaults are
tuned for A100- and H100-class GPUs and rarely need changing:

| Variable                     | Effect                                                            |
|:---------------------------- |:----------------------------------------------------------------- |
| `CLIMA_CUDA_MAX_WAVES`       | Upper bound on the number of thread-block waves per kernel launch |
| `CLIMA_FD_MAX_THREADS`       | Threads per block for finite-difference (column) kernels          |
| `CLIMA_DSS_MAX_THREADS`      | Threads per block for the DSS kernels                             |
| `CLIMA_COLLECT_KERNEL_STATS` | Record launch statistics for kernel-configuration sweeps          |

[`perf/sweep_kernel_configs.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/perf/sweep_kernel_configs.jl) sweeps them. GPU memory is the usual limit at
high resolution; reduce the element count or distribute the run over several
GPUs with MPI ([Run distributed with MPI](run_with_mpi.md)).

## Writing GPU-compatible code

The shared developer guide
[gpu\_performance.md](https://github.com/CliMA/ClimaCore.jl/blob/main/docs/dev-guides/performance/gpu_performance.md)
gives the rules: no allocation inside broadcasts, `ifelse` instead of
branches that diverge within a warp, type-stable closures, and no captured
non-constant globals. [Performance and portability](../explanation/performance.md)
explains how the kernels are organized.
