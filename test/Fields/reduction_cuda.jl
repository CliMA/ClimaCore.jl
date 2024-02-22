using Test
using CUDA
using ClimaComms
using Statistics
using LinearAlgebra

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

include("reduction_cuda_utils.jl")

@testset "test cuda reduction op on surface of sphere" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    context_cpu =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()) # CPU context for comparison

    # Set up discretization
    ne = 72
    Nq = 4
    ndof = ne * ne * 6 * Nq * Nq
    println(
        "running reduction test on $(context.device); problem size Ne = $ne; Nq = $Nq; ndof = $ndof; FT = $FT",
    )
    R = FT(6.37122e6) # radius of earth
    domain = Domains.SphereDomain(R)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    quad = Quadratures.GLL{Nq}()
    grid_topology = Topologies.Topology2D(context, mesh)
    grid_topology_cpu = Topologies.Topology2D(context_cpu, mesh)
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)
    space_cpu = Spaces.SpectralElementSpace2D(grid_topology_cpu, quad)

    coords = Fields.coordinate_field(space)
    Y = set_initial_condition(space)
    Y_cpu = set_initial_condition(space_cpu)

    h₀ = FT(200)
    Z = set_elevation(space, h₀)
    Z_cpu = set_elevation(space_cpu, h₀)

    result = Base.sum(Y)
    result_cpu = Base.sum(Y_cpu)

    local_max = Base.maximum(identity, Z)
    local_max_cpu = Base.maximum(identity, Z_cpu)

    local_min = Base.minimum(identity, Z)
    local_min_cpu = Base.minimum(identity, Z_cpu)
    # test weighted sum
    @test result ≈ 4 * pi * R^2 rtol = 1e-5
    @test result ≈ result_cpu
    # test maximum
    @test local_max ≈ h₀
    @test local_max ≈ local_max_cpu
    # test minimum
    @test local_min ≈ FT(0)
    @test local_min ≈ local_min_cpu
    # testing mean
    meanz = Statistics.mean(Z)
    meanz_cpu = Statistics.mean(Z_cpu)
    @test meanz ≈ meanz_cpu
    # testing norm
    norm1z = LinearAlgebra.norm(Z, 1)
    norm1z_cpu = LinearAlgebra.norm(Z_cpu, 1)
    @test norm1z ≈ norm1z_cpu

    norm2z = LinearAlgebra.norm(Z, 2)
    norm2z_cpu = LinearAlgebra.norm(Z_cpu, 2)
    @test norm2z ≈ norm2z_cpu

    norm3z = LinearAlgebra.norm(Z, 3)
    norm3z_cpu = LinearAlgebra.norm(Z_cpu, 3)
    @test norm3z ≈ norm3z_cpu

    norminfz = LinearAlgebra.norm(Z, Inf)
    norminfz_cpu = LinearAlgebra.norm(Z_cpu, Inf)
    @test norminfz ≈ norminfz_cpu
end

@testset "test cuda reduction op for extruded 3D domain (hollow sphere)" begin
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
        "running reduction test on $(context.device); h_elem = $h_elem; z_elem = $z_elem; npoly = $npoly; R = $R; z_max = $z_max; FT = $FT",
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
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_topology_cpu = Topologies.IntervalTopology(context_cpu, z_mesh)

    z_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    z_center_space_cpu = Spaces.CenterFiniteDifferenceSpace(z_topology_cpu)

    z_face_space = Spaces.FaceFiniteDifferenceSpace(z_topology)
    z_face_space_cpu = Spaces.FaceFiniteDifferenceSpace(z_topology_cpu)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    hv_center_space_cpu =
        Spaces.ExtrudedFiniteDifferenceSpace(h_space_cpu, z_center_space_cpu)
    hv_face_space_cpu =
        Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space_cpu)

    Yc = set_initial_condition(hv_center_space)
    Yf = set_initial_condition(hv_face_space)

    Yc_cpu = set_initial_condition(hv_center_space_cpu)
    Yf_cpu = set_initial_condition(hv_face_space_cpu)

    resultc = Base.sum(Yc)
    resultc_cpu = Base.sum(Yc_cpu)

    resultf = Base.sum(Yf)
    resultf_cpu = Base.sum(Yf_cpu)

    @test resultc_cpu ≈ resultc
    @test resultf_cpu ≈ resultf

    @test resultc ≈ (4 / 3) * π * ((R + z_max)^3 - R^3) rtol = 1e-2
    @test resultf ≈ (4 / 3) * π * ((R + z_max)^3 - R^3) rtol = 1e-2


    Yc = set_simple_field(hv_center_space)
    Yc_cpu = set_simple_field(hv_center_space_cpu)

    @test Base.maximum(identity, Yc) ≈ Base.maximum(identity, Yc_cpu)
    @test Base.minimum(identity, Yc) ≈ Base.minimum(identity, Yc_cpu)

    @test Statistics.mean(Yc) ≈ Statistics.mean(Yc_cpu)

    @test LinearAlgebra.norm(Yc, 1) ≈ LinearAlgebra.norm(Yc_cpu, 1)
    @test LinearAlgebra.norm(Yc, 2) ≈ LinearAlgebra.norm(Yc_cpu, 2)
    @test LinearAlgebra.norm(Yc, 3) ≈ LinearAlgebra.norm(Yc_cpu, 3)
    @test LinearAlgebra.norm(Yc, Inf) ≈ LinearAlgebra.norm(Yc_cpu, Inf)

    Yf = set_simple_field(hv_face_space)
    Yf_cpu = set_simple_field(hv_face_space_cpu)

    @test Base.maximum(identity, Yf) ≈ Base.maximum(identity, Yf_cpu)
    @test Base.minimum(identity, Yf) ≈ Base.minimum(identity, Yf_cpu)

    @test Statistics.mean(Yf) ≈ Statistics.mean(Yf_cpu)

    @test LinearAlgebra.norm(Yf, 1) ≈ LinearAlgebra.norm(Yf_cpu, 1)
    @test LinearAlgebra.norm(Yf, 2) ≈ LinearAlgebra.norm(Yf_cpu, 2)
    @test LinearAlgebra.norm(Yf, 3) ≈ LinearAlgebra.norm(Yf_cpu, 3)
    @test LinearAlgebra.norm(Yf, Inf) ≈ LinearAlgebra.norm(Yf_cpu, Inf)
end

@testset "test cuda reduction op for single column" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext(ClimaComms.CUDADevice())
    context_cpu =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()) # CPU context for comparison

    z0 = FT(0)
    z_max = FT(2 * π)
    z_nelems = 16

    println(
        "running column reduction test on $(context.device); z_nelems = $z_nelems; z0 = $z0; z_max = $z_max; FT = $FT",
    )

    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(z0),
        Geometry.ZPoint{FT}(z_max);
        boundary_tags = (:left, :right),
    )

    z_mesh = Meshes.IntervalMesh(domain; nelems = z_nelems)
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    z_topology_cpu = Topologies.IntervalTopology(context_cpu, z_mesh)


    center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    center_space_cpu = Spaces.CenterFiniteDifferenceSpace(z_topology_cpu)
    face_space_cpu = Spaces.FaceFiniteDifferenceSpace(center_space_cpu)

    Yc = ones(FT, center_space)
    Yc_cpu = ones(FT, center_space_cpu)

    Yf = ones(FT, face_space)
    Yf_cpu = ones(FT, face_space_cpu)

    resultc_cpu = Base.sum(Yc_cpu)
    resultc = Base.sum(Yc)

    resultf_cpu = Base.sum(Yf_cpu)
    resultf = Base.sum(Yf)
    # test weighted sum
    @test resultc ≈ resultc_cpu
    @test resultc ≈ z_max - z0

    @test resultf ≈ resultf_cpu
    @test resultf ≈ z_max - z0

    Yc = sin.(getproperty(Fields.coordinate_field(center_space), :z)) .+ 1
    Yc_cpu =
        sin.(getproperty(Fields.coordinate_field(center_space_cpu), :z)) .+ 1


    @test Base.maximum(identity, Yc) ≈ Base.maximum(identity, Yc_cpu)
    @test Base.minimum(identity, Yc) ≈ Base.minimum(identity, Yc_cpu)

    @test Statistics.mean(Yc) ≈ Statistics.mean(Yc_cpu)

    @test LinearAlgebra.norm(Yc, 1) ≈ LinearAlgebra.norm(Yc_cpu, 1)
    @test LinearAlgebra.norm(Yc, 2) ≈ LinearAlgebra.norm(Yc_cpu, 2)
    @test LinearAlgebra.norm(Yc, 3) ≈ LinearAlgebra.norm(Yc_cpu, 3)

    @test LinearAlgebra.norm(Yc, Inf) ≈ LinearAlgebra.norm(Yc_cpu, Inf)
end

@testset "test cuda reduction on a field with multiple variables" begin
    comms_ctx = ClimaComms.SingletonCommsContext(ClimaComms.device())
    comms_ctx_cpu =
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    FT = Float64
    println("running multi-variable reduction test on $(comms_ctx.device)")
    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint{FT}(0.0),
            Geometry.XPoint{FT}(1.0);
            periodic = false,
            boundary_names = (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint{FT}(0.0),
            Geometry.YPoint{FT}(1.0);
            periodic = false,
            boundary_names = (:south, :north),
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, 3, 3)
    topology = Topologies.Topology2D(comms_ctx, mesh)
    topology_cpu = Topologies.Topology2D(comms_ctx_cpu, mesh)
    quad = Quadratures.GLL{5}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    space_cpu = Spaces.SpectralElementSpace2D(topology_cpu, quad)
    coords = Fields.coordinate_field(space)
    coords_cpu = Fields.coordinate_field(space_cpu)

    q₀(coords, x_scale, y_scale) =
        (x = x_scale * coords.x, y = y_scale * coords.y)

    q = @. q₀(coords, 1.2, 1.5)
    q_cpu = @. q₀(coords_cpu, 1.2, 1.5)

    @test [sum(q)...] ≈ [sum(q_cpu)...]
end
