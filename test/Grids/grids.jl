using Test
using ClimaComms
using adapt

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

import ClimaCore: Grids, Spaces

@testset "todevice GPU to CPU conversions" begin
    FT = Float64
    gpu_device = ClimaComms.CUDADevice()
    gpu_context = ClimaComms.context(gpu_device)
    for gpu_space in TU.all_spaces(FT, context = gpu_context)
        gpu_grid = gpu_space.grid
        # Test backing array of initial grid
        # TODO add tests

        # Test conversion from GPU to CPU grid
        cpu_grid = Grids.todevice(Array, gpu_grid)
        # TODO add tests
    end
end

@testset "todevice CPU to GPU conversions" begin
    FT = Float64
    cpu_device = ClimaComms.CPUSingleThreaded()
    cpu_context = ClimaComms.context(cpu_device)
    for cpu_space in TU.all_spaces(FT, context = cpu_context)
        cpu_grid = cpu_space.grid
        # Test backing array of initial grid
        # TODO add tests

        # Test conversion from GPU to CPU grid
        gpu_grid = Spaces.todevice(CuArray, cpu_grid)
        # TODO add tests
    end
end
