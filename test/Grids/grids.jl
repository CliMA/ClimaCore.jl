using Test
using ClimaComms
using Adapt

import ClimaCore
include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

import ClimaCore: Grids, Spaces

@testset "todevice GPU/CPU conversions" begin
    # Set up GPU and CPU contexts and spaces
    FT = Float64
    device = ClimaComms.CUDADevice()
    gpu_context = ClimaComms.context(device)
    cpu_context = ClimaComms.context(device)
    gpu_spaces = TU.all_spaces(FT, context = gpu_context)
    cpu_spaces = TU.all_spaces(FT, context = cpu_context)

    i = 1
    for (gpu_space, cpu_space) in zip(gpu_spaces, cpu_spaces)
        @show i
        if !(gpu_space isa Spaces.PointSpace)
            gpu_grid = gpu_space.grid
            cpu_grid = cpu_space.grid

            # Test GPU to CPU
            cpu_grid_converted = Grids.todevice(Array, gpu_grid)
            # if cpu_space_converted isa Spaces.ExtrudedFiniteDifferenceSpace
            #     @show cpu_space_converted.grid.horizontal_grid.topology.context
            # elseif !(cpu_space_converted isa Spaces.PointSpace)
            #     @show cpu_space_converted.grid.topology.context
            # end
            if cpu_grid != cpu_grid_converted
                @show typeof(cpu_space)
            end
            @test cpu_grid == cpu_grid_converted

            # Test CPU to GPU
            gpu_grid_converted = Grids.todevice(CuArray, cpu_grid)
            @test gpu_grid == gpu_grid_converted
        end
        i += 1
    end
    # for gpu_space in TU.all_spaces(FT, context = gpu_context)
    #     # Test backing array of initial space
    #     @test parent(gpu_space.grid.local_geometry) isa CuArray

    #     # Test conversion from GPU to CPU space
    #     # TODO context doesn't actually get changed - implement that?
    #     cpu_space = Spaces.todevice(Array, gpu_space)
    #     @test parent(cpu_space.grid.local_geometry) isa Array
    # end
end


# @testset "todevice GPU to CPU conversions" begin
#     FT = Float64
#     gpu_device = ClimaComms.CUDADevice()
#     gpu_context = ClimaComms.context(gpu_device)
#     for gpu_space in TU.all_spaces(FT, context = gpu_context)
#         gpu_grid = gpu_space.grid
#         # Test backing array of initial grid
#         # TODO add tests

#         # Test conversion from GPU to CPU grid
#         cpu_grid = Grids.todevice(Array, gpu_grid)
#         # TODO add tests
#     end
# end

# @testset "todevice CPU to GPU conversions" begin
#     FT = Float64
#     cpu_device = ClimaComms.CPUSingleThreaded()
#     cpu_context = ClimaComms.context(cpu_device)
#     for cpu_space in TU.all_spaces(FT, context = cpu_context)
#         cpu_grid = cpu_space.grid
#         # Test backing array of initial grid
#         # TODO add tests

#         # Test conversion from GPU to CPU grid
#         gpu_grid = Spaces.todevice(CuArray, cpu_grid)
#         # TODO add tests
#     end
# end
