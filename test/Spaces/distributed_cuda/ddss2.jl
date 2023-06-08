using Logging
using Test

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies

using ClimaComms
using CUDA

# initializing MPI
const device = ClimaComms.device()
const context = ClimaComms.MPICommsContext(device)
pid, nprocs = ClimaComms.init(context)
#=
 _
|1|
|_|
|2|
|=|
|3|
|_|
|4|
|_|
=#
@testset "4x1 element mesh with periodic boundaries on 2 processes" begin
    n1, n2 = 4, 1
    x1periodic, x2periodic = true, true
    Nq, Nv = 3, 1
    x1min, x1max = -2π, 2π
    x2min, x2max = -2π, 2π
    # initializing MPI
    device = ClimaComms.device()
    context = ClimaComms.MPICommsContext(device)
    pid, nprocs = ClimaComms.init(context)
    iamroot = ClimaComms.iamroot(context)
    if iamroot
        println("running test on $device device with $nprocs processes")
    end

    domain = Domains.RectangleDomain(
        Domains.IntervalDomain(
            Geometry.XPoint(x1min),
            Geometry.XPoint(x1max),
            periodic = x1periodic,
            boundary_names = x1periodic ? nothing : (:west, :east),
        ),
        Domains.IntervalDomain(
            Geometry.YPoint(x2min),
            Geometry.YPoint(x2max),
            periodic = x2periodic,
            boundary_names = x2periodic ? nothing : (:north, :south),
        ),
    )
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    topology = Topologies.Topology2D(context, mesh, Meshes.elements(mesh))
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)


    @test Topologies.nlocalelems(Spaces.topology(space)) == 2

    @test Topologies.local_neighboring_elements(space.topology, 1) == [2]
    @test Topologies.local_neighboring_elements(space.topology, 2) == [1]

    @test Topologies.ghost_neighboring_elements(space.topology, 1) == [2]
    @test Topologies.ghost_neighboring_elements(space.topology, 2) == [1]

    init_state(local_geometry, p) = (ρ = 1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))
    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .=
        reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel)) .+
        (pid - 1) * Nq * Nq * nel
    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer) # DSS2
    #=
    [18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5,
     18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
    =#
#! format: off
    if pid == 1
        @test Array(yarr[:]) == [18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 18.5, 5.0, 9.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5, 9.5, 14.0, 18.5]
    else
        @test Array(yarr[:]) == [18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 18.5, 23.0, 27.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5, 27.5, 32.0, 18.5]
    end
#! format: on
    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    iamroot && @show p

    #testing weighted dss on a vector field
    init_vectorstate(local_geometry, p) = Geometry.Covariant12Vector(1.0, -1.0)
    v0 = init_vectorstate.(Fields.local_geometry_field(space), Ref(nothing))
    vx = copy(v0)

    dss_vbuffer = Spaces.create_dss_buffer(v0)
    Spaces.weighted_dss!(v0, dss_vbuffer)

    @test parent(v0) ≈ parent(vx)

    p = @allocated Spaces.weighted_dss!(v0, dss_vbuffer)
    iamroot && @show p
end
