using Test
using ClimaComms
if pkgversion(ClimaComms) >= v"0.6"
    ClimaComms.@import_required_backends
end
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

@testset "DSS on Equiangular Cubed Sphere mesh (ne = 3, serial run)" begin
    device = ClimaComms.device() #ClimaComms.CUDADevice()
    context = ClimaComms.SingletonCommsContext(device)
    context_cpu =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())

    println("running test on $device device")

    domain = Domains.SphereDomain(300.0)
    mesh = Meshes.EquiangularCubedSphere(domain, 3)
    topology = Topologies.Topology2D(context, mesh)
    topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    quad = Quadratures.GLL{4}()
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

@testset "DSS of Covarinat12Vector & Covariant123Vector on extruded Cubed Sphere mesh (ne = 3, serial run)" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    context_cpu =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()) # CPU context for comparison
    R = FT(6.371229e6)

    npoly = 4
    z_max = FT(30e3)
    z_elem = 10
    h_elem = 4
    println(
        "running dss-Covariant123Vector test on $(context.device); h_elem = $h_elem; z_elem = $z_elem; npoly = $npoly; R = $R; z_max = $z_max; FT = $FT",
    )
    # horizontal space
    domain = Domains.SphereDomain(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    horizontal_topology_cpu = Topologies.Topology2D(
        context_cpu,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)
    h_space_cpu = Spaces.SpectralElementSpace2D(horizontal_topology_cpu, quad)

    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_topology_cpu = Topologies.IntervalTopology(context_cpu, z_mesh)

    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    z_center_space_cpu = Spaces.CenterFiniteDifferenceSpace(z_topology_cpu)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)

    hv_center_space_cpu =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space_cpu, z_center_space_cpu)

    # test DSS for a Covariant12Vector
    init_state_covariant12(local_geometry, p) =
        Geometry.Covariant12Vector(1.0, -1.0)

    y12 =
        init_state_covariant12.(
            Fields.local_geometry_field(hv_center_space),
            Ref(nothing),
        )
    y12_cpu =
        init_state_covariant12.(
            Fields.local_geometry_field(hv_center_space_cpu),
            Ref(nothing),
        )

    dss_buffer12 = Spaces.create_dss_buffer(y12)
    dss_buffer12_cpu = Spaces.create_dss_buffer(y12_cpu)
    # ensure physical velocity is continous across SE boundary for initial state
    Spaces.weighted_dss!(y12 => dss_buffer12)
    Spaces.weighted_dss!(y12_cpu => dss_buffer12_cpu)

    yinit12 = copy(y12)
    yinit12_cpu = copy(y12_cpu)

    Spaces.weighted_dss!(y12, dss_buffer12)
    Spaces.weighted_dss!(y12_cpu, dss_buffer12_cpu)
    @test yinit12 ≈ y12
    @test yinit12_cpu ≈ y12_cpu
    @test parent(y12_cpu) ≈ Array(parent(y12))

    # test DSS for a Covariant123Vector
    init_state_covariant123(local_geometry, p) =
        Geometry.Covariant123Vector(1.0, -1.0, 1.0)

    y123 =
        init_state_covariant123.(
            Fields.local_geometry_field(hv_center_space),
            Ref(nothing),
        )
    y123_cpu =
        init_state_covariant123.(
            Fields.local_geometry_field(hv_center_space_cpu),
            Ref(nothing),
        )

    dss_buffer123 = Spaces.create_dss_buffer(y123)
    dss_buffer123_cpu = Spaces.create_dss_buffer(y123_cpu)

    # ensure physical velocity is continous across SE boundary for initial state
    Spaces.weighted_dss!(y123, dss_buffer123)
    Spaces.weighted_dss!(y123_cpu, dss_buffer123_cpu)

    yinit123 = copy(y123)
    yinit123_cpu = copy(y123_cpu)

    Spaces.weighted_dss!(y123, dss_buffer123)
    Spaces.weighted_dss!(y123_cpu, dss_buffer123_cpu)

    @test yinit123 ≈ y123
    @test yinit123_cpu ≈ y123_cpu
    @test parent(y123_cpu) ≈ Array(parent(y123))
end
