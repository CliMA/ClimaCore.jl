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
| `CLIMA_DSS_MAX_THREADS`                    | `ClimaCoreCUDAExt` | Integer (default `256`, at most `1024`)     | Threads per block for the DSS kernels.                                                                                                                                                                 |
| `CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE` | `ClimaCoreCUDAExt` | `true` or unset                             | Name each CUDA kernel after the call site that launched it, for profilers.                                                                                                                             |
| `CLIMA_COLLECT_KERNEL_STATS`               | `ClimaCoreCUDAExt` | `true` or unset                             | Record per-launch statistics for kernel-configuration sweeps. Read at precompilation of the extension, so changing it requires recompiling.                                                            |
| `CLIMA_AUTO_REGISTER_CAP`                  | `ClimaCoreCUDAExt` | `true` (default) or `false`                 | Compile a kernel with a register cap at the edge of the occupancy band it compiles into, when a second compile shows the cap is honored, buys resident warps, and costs little local memory.           |
| `CLIMA_AUTO_REGISTER_CAP_MAX_SHED`         | `ClimaCoreCUDAExt` | Integer (default `12`)                      | How many registers below its own count a kernel may be capped; a band edge further away than this is not asked for.                                                                                   |
| `CLIMA_AUTO_REGISTER_CAP_MAX_WARPS`        | `ClimaCoreCUDAExt` | Integer (default `16`)                      | Resident warps per multiprocessor above which a kernel is left alone, no longer being short of residency.                                                                                             |
| `CLIMA_AUTO_REGISTER_CAP_LOCAL_GROWTH_PCT` | `ClimaCoreCUDAExt` | Integer (default `25`)                      | How much a capped kernel's local frame may grow, as a percentage of the frame it compiled without a cap.                                                                                               |
| `CLIMA_AUTO_REGISTER_CAP_LOCAL_SLACK`      | `ClimaCoreCUDAExt` | Integer (default `32`)                      | How much that frame may grow in bytes, whichever budget is larger; most kernels compile with no local frame, where any growth is infinite in relative terms.                                          |
| `CLIMA_AUTO_REGISTER_CAP_EXEMPT`           | `ClimaCoreCUDAExt` | Comma-separated names (default empty)       | Kernels the cap is never applied to, matched as substrings of the kernel name, which requires `CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE`.                                                             |

The `*_MAX_*` and kernel-naming variables are read when the extension loads;
changing them requires a new Julia session. [`perf/sweep_kernel_configs.jl`](https://github.com/CliMA/ClimaCore.jl/blob/main/perf/sweep_kernel_configs.jl) sweeps them. The defaults
are tuned for A100- and H100-class GPUs ([Run on a GPU](../howto/run_on_gpu.md)).

The ClimaComms [documentation](https://clima.github.io/ClimaComms.jl/dev/)
describes `CLIMACOMMS_DEVICE` and `CLIMACOMMS_CONTEXT`, including the
`SINGLETON` context for running an MPI-launched job on one process.
