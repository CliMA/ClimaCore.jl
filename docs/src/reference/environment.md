# Environment variables

ClimaCore reads a small number of environment variables. The two that select
the hardware belong to ClimaComms and apply to every CliMA package; one
controls precompilation; the rest tune or instrument the CUDA kernels and are
read once when the CUDA extension loads.

| Variable                                   | Read by            | Values                                      | Effect                                                                                                                                                                                                 |
|:------------------------------------------ |:------------------ |:------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CLIMACOMMS_DEVICE`                        | ClimaComms         | `CPU` (default), `CPUMultiThreaded`, `CUDA` | The compute device returned by `ClimaComms.device()`, on which spaces and fields are allocated. Requires `ClimaComms.@import_required_backends` in the script and CUDA.jl in the load path for `CUDA`. |
| `CLIMACOMMS_CONTEXT`                       | ClimaComms         | `SINGLETON` (default), `MPI`                | The communication context returned by `ClimaComms.context()`; `MPI` distributes horizontal topologies over ranks.                                                                                      |
| `CLIMA_SKIP_PRECOMPILE_WORKLOAD`           | ClimaCore          | `true` or unset                             | Skip the precompilation workload that warms up common broadcasts, for faster development builds.                                                                                                       |
| `CLIMA_CUDA_MAX_WAVES`                     | `ClimaCoreCUDAExt` | Integer (default `1`)                       | Upper bound on the number of thread-block waves a kernel launch may use.                                                                                                                               |
| `CLIMA_FD_MAX_THREADS`                     | `ClimaCoreCUDAExt` | Integer (default `128`, at most `1024`)     | Threads per block for finite-difference (column) kernels.                                                                                                                                              |
| `CLIMA_DSS_MAX_THREADS`                    | `ClimaCoreCUDAExt` | Integer (default `256`, at most `1024`)     | Threads per block for the DSS kernels.                                                                                                                                                                 |
| `CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE` | `ClimaCoreCUDAExt` | `true` or unset                             | Name each CUDA kernel after the call site that launched it, for profilers.                                                                                                                             |
| `CLIMA_COLLECT_KERNEL_STATS`               | `ClimaCoreCUDAExt` | `true` or unset                             | Record per-launch statistics for kernel-configuration sweeps. Read at precompilation of the extension, so changing it requires recompiling.                                                            |

The `*_MAX_*` and kernel-naming variables are read when the extension loads;
changing them requires a new Julia session. [`perf/sweep_kernel_configs.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/perf/sweep_kernel_configs.jl) sweeps them. The defaults
are tuned for A100- and H100-class GPUs ([Run on a GPU](../howto/run_on_gpu.md)).

The ClimaComms [documentation](https://clima.github.io/ClimaComms.jl/dev/)
describes `CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT`, including the
`SINGLETON` context for running an MPI-launched job on one process.
