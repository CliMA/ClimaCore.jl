using Test
using CUDA
using ClimaComms
import ClimaCore: Device

@testset "check device detection on GPU" begin
    device = Device.device()
    cuda_context = ClimaComms.SingletonCommsContext(device)
    DA = Device.device_array_type(cuda_context.device)

    @test device isa ClimaComms.CUDA
    @test cuda_context.device == ClimaComms.CUDA()
    @test DA == CuArray

    override_device = Device.device(disablegpu = true)
    override_cuda_context = ClimaComms.SingletonCommsContext(override_device)
    DA = Device.device_array_type(override_cuda_context.device)

    @test override_device isa ClimaComms.CPU
    @test override_cuda_context.device == ClimaComms.CPU()
    @test DA == Array
end
