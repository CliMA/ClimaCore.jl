using Test
using CUDA
using ClimaComms
using ClimaCommsMPI
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

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 3, 4 process run)" begin
    device = Device.device() #ClimaComms.CUDA()
    context_cuda = ClimaCommsMPI.MPICommsContext(device)
    context_cpu = ClimaCommsMPI.MPICommsContext(ClimaComms.CPU())

    pid_cuda, nprocs_cuda = ClimaComms.init(context_cuda)
    pid_cpu, nprocs_cpu = ClimaComms.init(context_cpu)

    @assert pid_cuda == pid_cpu "pids different for CUDA and CPU contexts"
    @assert nprocs_cuda == nprocs_cpu "nprocs different for CUDA and CPU contexts"
    pid, nprocs = pid_cuda, nprocs_cuda

    if pid == 1
        println("running tests on $device device and CPU with $nprocs procs")
    end
    domain = Domains.SphereDomain(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology_cuda = Topologies.Topology2D(context_cuda, mesh)
    topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    quad = Spaces.Quadratures.GLL{4}()
    space_cuda = Spaces.SpectralElementSpace2D(topology_cuda, quad)
    space_cpu = Spaces.SpectralElementSpace2D(topology_cpu, quad)
    x_cuda = ones(space_cuda)
    x_cpu = ones(space_cpu)

    Spaces.weighted_dss!(x_cuda)
    Spaces.weighted_dss!(x_cpu)

    @test parent(x_cpu) ≈ Array(parent(x_cuda))
end
