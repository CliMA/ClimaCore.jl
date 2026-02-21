#=
julia --project
using Revise; include(joinpath("test", "Spaces", "ddss1_cs.jl"))
=#
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
    Topologies,
    DataLayouts

function get_space_cs(::Type{FT}; context, R = 300.0) where {FT}
    domain = Domains.SphereDomain{FT}(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    quad = Quadratures.GLL{4}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    return space
end

function get_space_and_buffers(::Type{FT}; context) where {FT}
    init_state_covariant12(local_geometry, p) =
        Geometry.Covariant12Vector(1.0, -1.0)
    init_state_covariant123(local_geometry, p) =
        Geometry.Covariant123Vector(1.0, -1.0, 1.0)

    R = FT(6.371229e6)
    npoly = 2
    z_max = FT(30e3)
    z_elem = 3
    h_elem = 2
    device = ClimaComms.device(context)
    @info "running dss-Covariant123Vector test on $(device)" h_elem z_elem npoly R z_max FT
    # horizontal space
    domain = Domains.SphereDomain{FT}(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)
    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zero(z_max)),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    args = (Fields.local_geometry_field(space), Ref(nothing))
    y12 = init_state_covariant12.(args...)
    y123 = init_state_covariant123.(args...)
    dss_buffer12 = Spaces.create_dss_buffer(y12)
    dss_buffer123 = Spaces.create_dss_buffer(y123)
    return (; space, y12, y123, dss_buffer12, dss_buffer123)
end

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 3, serial run)" begin
    FT = Float64
    device = ClimaComms.device()
    context = ClimaComms.SingletonCommsContext(device)
    println("running test on $device device")
    space = get_space_cs(FT; context)
    space_cpu = get_space_cs(FT; context)
    x = ones(space)
    Spaces.weighted_dss!(x)

    @test Array(parent(x)) ≈ ones(size(parent(x))) # TODO: improve the quality of this test
end

@testset "DSS of Covariant12Vector & Covariant123Vector on extruded Cubed Sphere mesh (ne = 3, serial run)" begin
    FT = Float64
    device = ClimaComms.device()
    nt = get_space_and_buffers(FT; context = ClimaComms.context(device))

    # test DSS for a Covariant12Vector
    # ensure physical velocity is continous across SE boundary for initial state
    Spaces.weighted_dss!(nt.y12 => nt.dss_buffer12)
    init = copy(nt.y12)
    Spaces.weighted_dss!(nt.y12, nt.dss_buffer12)
    @test init ≈ nt.y12
    # ensure physical velocity is continous across SE boundary for initial state
    Spaces.weighted_dss!(nt.y123, nt.dss_buffer123)
    init = copy(nt.y123)
    Spaces.weighted_dss!(nt.y123, nt.dss_buffer123)
    @test init ≈ nt.y123
end

# TODO: remove once the quality of the above test is improved
(ClimaComms.device() isa ClimaComms.CUDADevice) &&
    @testset "GPU-vs-CPU test: DSS of Covariant12Vector & Covariant123Vector on extruded Cubed Sphere mesh (ne = 3, serial run)" begin
        FT = Float64
        cpu_device = ClimaComms.CPUSingleThreaded()
        gpu_device = ClimaComms.CUDADevice()
        gpu =
            get_space_and_buffers(FT; context = ClimaComms.context(gpu_device))
        cpu =
            get_space_and_buffers(FT; context = ClimaComms.context(cpu_device))

        # test DSS for a Covariant12Vector
        # ensure physical velocity is continous across SE boundary for initial state
        Spaces.weighted_dss!(cpu.y12 => cpu.dss_buffer12)
        Spaces.weighted_dss!(gpu.y12 => gpu.dss_buffer12)

        inity12 = (; cpu = copy(cpu.y12), gpu = copy(gpu.y12))

        Spaces.weighted_dss!(cpu.y12, cpu.dss_buffer12)
        Spaces.weighted_dss!(gpu.y12, gpu.dss_buffer12)

        @test inity12.cpu ≈ cpu.y12
        @test inity12.gpu ≈ gpu.y12
        @test parent(cpu.y12) ≈ Array(parent(gpu.y12))

        # ensure physical velocity is continous across SE boundary for initial state
        Spaces.weighted_dss!(cpu.y123, cpu.dss_buffer123)
        Spaces.weighted_dss!(gpu.y123, gpu.dss_buffer123)

        inity123 = (; cpu = copy(cpu.y123), gpu = copy(gpu.y123))

        Spaces.weighted_dss!(cpu.y123, cpu.dss_buffer123)
        Spaces.weighted_dss!(gpu.y123, gpu.dss_buffer123)

        @test inity123.cpu ≈ cpu.y123
        @test inity123.gpu ≈ gpu.y123
        @test parent(cpu.y123) ≈ Array(parent(gpu.y123))
    end
