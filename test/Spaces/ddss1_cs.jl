using Test
using ClimaComms
import ClimaCore:
    Device,
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 3, serial run)" begin
    device = Device.device() #ClimaComms.CUDADevice()
    context = ClimaComms.SingletonCommsContext(device)
    context_cpu = ClimaComms.SingletonCommsContext(ClimaComms.CPUDevice())

    println("running test on $device device")

    domain = Domains.SphereDomain(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    quad = Spaces.Quadratures.GLL{4}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    space_cpu = Spaces.SpectralElementSpace2D(topology_cpu, quad)

    x = ones(space)
    x_cpu = ones(space_cpu)

    Spaces.weighted_dss!(x)
    Spaces.weighted_dss!(x_cpu)

    @test parent(x_cpu) ≈ Array(parent(x))
end
