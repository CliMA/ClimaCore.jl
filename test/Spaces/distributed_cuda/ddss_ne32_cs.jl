using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies

# Widens a scalar field into a field of `K` identical components. Broadcasting a
# callable struct keeps `K` a compile-time constant, which the GPU kernels need.
struct Replicate{K} end
(::Replicate{K})(x) where {K} = ntuple(_ -> x, Val(K))

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 32)" begin
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    context_cuda = ClimaComms.MPICommsContext(device)
    context_cpu = ClimaComms.MPICommsContext(ClimaComms.CPUSingleThreaded())

    pid_cuda, nprocs_cuda = ClimaComms.init(context_cuda)
    pid_cpu, nprocs_cpu = ClimaComms.init(context_cpu)

    @assert pid_cuda == pid_cpu "pids different for CUDA and CPU contexts"
    @assert nprocs_cuda == nprocs_cpu "nprocs different for CUDA and CPU contexts"
    pid, nprocs = pid_cuda, nprocs_cuda

    if pid == 1
        println("running tests on $device device and CPU with $nprocs procs")
    end
    domain = Domains.SphereDomain(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 32)
    topology_cuda = Topologies.Topology2D(context_cuda, mesh)
    topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    quad = Quadratures.GLL{4}()
    space_cuda = Spaces.SpectralElementSpace2D(topology_cuda, quad)
    space_cpu = Spaces.SpectralElementSpace2D(topology_cpu, quad)
    x_cuda = ones(space_cuda)
    x_cpu = ones(space_cpu)

    Spaces.weighted_dss!(x_cuda)
    Spaces.weighted_dss!(x_cpu)

    @test parent(x_cpu) ≈ Array(parent(x_cuda))


    field_cuda = Geometry.Covariant12Vector.(ones(space_cuda), ones(space_cuda))
    field_cpu = Geometry.Covariant12Vector.(ones(space_cpu), ones(space_cpu))

    Spaces.weighted_dss!(field_cuda)
    Spaces.weighted_dss!(field_cpu)

    @test parent(field_cpu) ≈ Array(parent(field_cuda))

    # Weighted DSS leaves a constant field unchanged, whatever its number of
    # components. Unlike the comparisons above, which only establish that the
    # CPU and GPU implementations agree, this is a check against ground truth,
    # so it also catches a bug that both of them share.
    #
    # The number of components is swept because it sets the size of the
    # messages exchanged between processes: at ne = 32 with GLL{4} and two
    # processes, each neighbour is sent 576 Float64 per component. The
    # `Covariant12Vector` test above is the only multi-component case in this
    # file, and it is therefore also the only large-message case, which leaves
    # "the Nf > 1 code path is wrong" and "the ghost exchange drops large
    # messages" indistinguishable. These cases carry no vectors at all, so a
    # failure here isolates the latter.
    for K in (1, 3, 8), space in (space_cuda, space_cpu)
        field = Replicate{K}().(ones(space))
        Spaces.weighted_dss!(field)
        @test maximum(abs, Array(parent(field)) .- 1) < 1e-12
    end
end
