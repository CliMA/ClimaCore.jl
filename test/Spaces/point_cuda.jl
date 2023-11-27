import ClimaCore
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, column, Fields
import ClimaComms
using Test

compare(cpu, gpu) = all(parent(cpu) .≈ Array(parent(gpu)))
compare(cpu, gpu, sym) =
    all(parent(getproperty(cpu, sym)) .≈ Array(parent(getproperty(gpu, sym))))

@testset "CuArray-backed point spaces" begin
    cpu_context =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    gpu_context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())

    point = Geometry.ZPoint(1.0)

    cpuspace = Spaces.PointSpace(cpu_context, point)
    gpuspace = Spaces.PointSpace(gpu_context, point)

    # Test that all geometries match with CPU version:
    @test compare(cpuspace, gpuspace, :local_geometry)

    @test ClimaComms.device(gpuspace) == ClimaComms.CUDADevice()
    @test ClimaComms.context(gpuspace) == gpu_context

end
