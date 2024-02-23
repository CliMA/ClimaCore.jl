using Test
using ClimaComms
using adapt
import .TestUtilities as TU

import ClimaCore: Grids, Spaces

@testset "todevice CPU/GPU conversions" begin
    FT = Float64
    gpu_device = ClimaComms.CUDADevice()
    gpu_context = ClimaComms.context(gpu_device)
    for gpu_space in TU.all_spaces(FT, context = gpu_context)
        gpu_grid = gpu_space.grid
        cpu_grid = Grids.todevice(Array, gpu_grid)


        # Test conversion from GPU to CPU grid
        # TODO what can we test here?
        # cpu_grid.local_geometry type

        # Test conversion back from CPU to GPU grid
        # TODO this doesn't result in CuArrays
        gpu_grid2 = Spaces.todevice(CuArray, cpu_grid)
        # @test gpu_grid == gpu_grid2

    end
end
