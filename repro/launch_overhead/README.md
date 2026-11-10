# Host overhead of kernel launches in ClimaCore 1.0.8

On a GPU, one `Adapt.adapt(CUDA.KernelAdaptor(), data)` of a data layout can take about 8.6 µs instead of about 100 ns, for every data-layout argument of every kernel launch.
In a small ClimaAtmos box this made the run bound by the host: stock ClimaAtmos `b9f85ce8` with `config/model_configs/rcemipii_box_CRM_1M.yml` on one A100 takes 0.0355 s per step, against 0.0082 s with the change on this branch.

## Cause

- The Adapt rule of a data layout calls `rebuild`, which evaluates `is_subscope(ThisHost(), ThisKernel())`.
  That query recurses over `ThisKernel`, `ThisSubBlock{128}`, ..., `ThisThread`, and `ThisSubBlock{N}()` takes its shared-memory parameter from the generated function `shmem_level_bytes` (`ext/cuda/scopes.jl`).
- If inferring that generator is the first use of the GPU in the process, it creates the CUDA context.
  CUDA.jl 6.4.1 then runs `ptxas --version` as a subprocess and waits for it.
- If another task is ready to run at that moment, the wait switches tasks.
  A task switch is not allowed inside a generator, so the generator throws, inference drops the error, and the query is inferred as not constant for the rest of the process.
  Every level of the recursion then builds its sub-block type at run time with `apply_type`.
- A ready task is common: `import Distributed` starts one (ClimaAtmos loads Distributed through Flux and MPI).
  It stays ready until the main task yields once; a write to standard output yields when the output is a pipe and does not when it is a file.
  So the same script can be slow with its log written to a file and fast when piped.

## Change on this branch

- `7a06a57b`: `is_subscope` for `ThisKernel` and `ThisSubBlock` scopes, and `points_subscope` for `ThisSubBlock` scopes, in closed form from the thread counts, so the Adapt rule constructs no sub-block type.
- `64c30cba`: a test of the closed forms against a walk of the partition chain, including sub-blocks with different shared memory.
- With the change, a ClimaAtmos run still pays part of the cost (0.0082 against 0.0050 s per step when the CUDA context exists before the model is built), probably through the remaining calls of `shmem_level_bytes` at each launch; not confirmed.
- Workaround without the change: create the CUDA context before the first launch is inferred, for example with `CUDA.zeros(1)` right after the imports.

## Reproduce

On a machine with an NVIDIA GPU, from this folder, with standard output to a file:

```
julia --project=base -e 'using Pkg; Pkg.instantiate()'
julia --project=fixed -e 'using Pkg; Pkg.instantiate()'
for env in base fixed; do
    for mode in in_function distributed ready_task; do
        julia --project=$env -t 1 repro.jl $mode > $env-$mode.txt
    done
done
cat *.txt
```

`base` is registered ClimaCore 1.0.8, and `fixed` is this branch.
`repro.jl` makes a `VIJFH{Float32, 74, 4, 4}` data layout on a `CuArray`, converts it once, and prints the inferred code size of the query and the host time of one conversion.
The modes differ only in what happens before the first conversion.

Results on an A100 (driver 595.91.07, Julia 1.13.1, CUDA.jl 6.4.1, Adapt 4.8.0, ClimaComms 0.6.13):

| Mode | Before the first conversion | 1.0.8 | This branch |
| --- | --- | --- | --- |
| `in_function` | allocation and conversion in one function | 1 statement, 98 ns | 99 ns |
| `distributed` | `import Distributed`, then the same | 48 statements, 8612 ns | 97 ns |
| `ready_task` | `@async nothing`, then the same | 48 statements, 8668 ns | 101 ns |

Julia 1.13.3 gives the same pattern.
The modes `toplevel`, `infer_first` and `spawn` of `repro.jl` do not trigger it.
