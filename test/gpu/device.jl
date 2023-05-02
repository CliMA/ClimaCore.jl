using Test
using CUDA
using ClimaComms

@testset "check device detection on GPU" begin
    device = ClimaComms.device()
    cuda_context = ClimaComms.SingletonCommsContext(device)
    DA = ClimaComms.array_type(cuda_context.device)

    @test device isa ClimaComms.CUDADevice
    @test cuda_context.device == ClimaComms.CUDADevice()
    @test DA == CuArray

    override_device = ClimaComms.device(disablegpu = true)
    override_cuda_context = ClimaComms.SingletonCommsContext(override_device)
    DA = ClimaComms.array_type(override_cuda_context.device)

    @test override_device isa ClimaComms.CPUDevice
    @test override_cuda_context.device == ClimaComms.CPUDevice()
    @test DA == Array
end
