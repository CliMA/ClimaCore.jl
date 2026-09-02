# Move data between CPU and GPU

A field, field vector, or space lives on one device. `ClimaCore.to_device`
copies it to another device and `ClimaCore.to_cpu` to the CPU; both return a
copy, so the original is unchanged and `out === x` does not hold.

## Prerequisites

CUDA.jl loaded (`ClimaComms.@import_required_backends` with
`CLIMACOMMS_DEVICE=CUDA`) for anything involving `CUDADevice()`.

## Steps

 1. Build or receive an object on one device:

    ```julia
    using ClimaComms
    ClimaComms.@import_required_backends
    import ClimaCore
    using ClimaCore.CommonSpaces
    cpu_space = ColumnSpace(; z_elem = 10, z_min = 0.0, z_max = 10.0, staggering = CellCenter())
    ```

 2. Move it to the GPU. Every array inside the space's grid becomes a
    `CuArray`, and fields created on the moved space live on the GPU:

    ```julia
    cuda_space = ClimaCore.to_device(ClimaComms.CUDADevice(), cpu_space)
    ```

 3. Move a result back for host-side work (a plotting library, a test
    comparison):

    ```julia
    field_on_cpu = ClimaCore.to_cpu(field)
    ```

## Notes

  - A field moved on its own carries a moved copy of its space; a field and a
    space moved separately are on different space objects, and broadcasting
    between them raises a mismatched-spaces error. Move the state once and
    derive everything from it.
  - `Remapping.interpolate` returns a host array, so plotting from a GPU run
    needs no explicit transfer.
  - `to_device` targets `CPUSingleThreaded()` and `CUDADevice()`.
