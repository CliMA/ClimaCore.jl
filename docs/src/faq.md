# Frequently asked questions

## Moving objects between the CPU and GPU.

Occasionally, it's helpful to move data between the CPU and GPU. ClimaCore
supports this through a method, [`to_device`](@ref ClimaCore.to_device). We also
have a specialized version for converting only to the CPU: [`to_cpu`](@ref ClimaCore.to_cpu).

For example, to create a CUDA column space from a CPU space, you can use:

```julia
using ClimaComms
using ClimaCore
using ClimaCore.CommonSpaces
cpu_space = ColumnSpace(;
    z_elem = 10,
    z_min = 0,
    z_max = 10,
    staggering = CellCenter()
)
cuda_space = ClimaCore.to_device(ClimaComms.CUDADevice(), cpu_space)
```

Similarly, we can convert back with `to_cpu`:
```julia
new_cpu_space = ClimaCore.to_cpu(cuda_space)
```
