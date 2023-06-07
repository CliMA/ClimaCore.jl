using Test
using ClimaComms
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 3, serial run)" begin
    device = ClimaComms.device() #ClimaComms.CUDADevice()
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
    wrong_field = map(Fields.coordinate_field(space)) do cf
        (; a = Float64(0))
    end
    wrong_buffer = Spaces.create_dss_buffer(wrong_field)
    @test_throws ErrorException("Incorrect buffer eltype") Spaces.weighted_dss!(
        x,
        wrong_buffer,
    )
    @test_throws ErrorException("Incorrect buffer eltype") Spaces.weighted_dss_start!(
        x,
        wrong_buffer,
    )
    @test_throws ErrorException("Incorrect buffer eltype") Spaces.weighted_dss_internal!(
        x,
        wrong_buffer,
    )
    @test_throws ErrorException("Incorrect buffer eltype") Spaces.weighted_dss_ghost!(
        x,
        wrong_buffer,
    )
end
