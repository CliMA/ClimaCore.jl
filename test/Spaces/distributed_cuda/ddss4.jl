using Logging
using Test

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies, Quadratures

using ClimaComms
using CUDA

# initializing MPI
const device = ClimaComms.device()
const context = ClimaComms.MPICommsContext(device)
pid, nprocs = ClimaComms.init(context)
#=
 _ _
|3|4|
 - - 
|1|2|
 - -  
=#
@testset "2x2 element mesh with periodic boundaries on 4 processes" begin
    n1, n2 = 2, 2
    x1periodic, x2periodic = true, true
    Nq, Nv = 4, 1
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
    quad = Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)
    init_state(local_geometry, p) = (ρ = 1.0)
    y0 = init_state.(Fields.local_geometry_field(space), Ref(nothing))
    nel = Topologies.nlocalelems(Spaces.topology(space))
    yarr = parent(y0)
    yarr .=
        reshape(1:(Nq * Nq * nel), (Nq, Nq, 1, nel)) .+
        (pid - 1) * Nq * Nq * nel
    dss_buffer = Spaces.create_dss_buffer(y0)
    Spaces.weighted_dss!(y0, dss_buffer) # DSS2
    passed = 0
    #=
    output from single process run:
    [32.5  24.0  25.0  32.5  14.5  6.0  7.0  14.5  18.5  10.0  11.0  18.5  32.5  24.0  25.0  32.5]
    [32.5  40.0  41.0  32.5  14.5  22.0  23.0  14.5  18.5  26.0  27.0  18.5  32.5  40.0  41.0  32.5]
    [32.5  24.0  25.0  32.5  46.5  38.0  39.0  46.5  50.5  42.0  43.0  50.5  32.5  24.0  25.0  32.5]
    [32.5  40.0  41.0  32.5  46.5  54.0  55.0  46.5  50.5  58.0  59.0  50.5  32.5  40.0  41.0  32.5]
    =#
#! format: off
    if pid == 1
        if Array(yarr)[:] == [32.5, 24.0, 25.0, 32.5, 14.5, 6.0, 7.0, 14.5, 18.5, 10.0, 11.0, 18.5, 32.5, 24.0, 25.0,  32.5]
            passed += 1
        end
    elseif pid == 2
        if Array(yarr)[:] == [32.5, 40.0, 41.0, 32.5, 14.5, 22.0, 23.0, 14.5, 18.5, 26.0, 27.0, 18.5, 32.5, 40.0, 41.0, 32.5]
            passed += 1
        end
    elseif pid == 3
        if Array(yarr)[:] == [32.5, 24.0, 25.0, 32.5, 46.5, 38.0, 39.0, 46.5, 50.5, 42.0, 43.0, 50.5, 32.5, 24.0, 25.0, 32.5]
            passed += 1
        end
    else
        if Array(yarr)[:] == [32.5, 40.0, 41.0, 32.5, 46.5, 54.0, 55.0, 46.5, 50.5, 58.0, 59.0, 50.5, 32.5, 40.0, 41.0, 32.5]
            passed += 1
        end
    end
#! format: on
    passed = ClimaComms.reduce(context, passed, +)
    if pid == 1
        @test passed == 4#8
    end
    p = @allocated Spaces.weighted_dss!(y0, dss_buffer)
    if pid == 1
        @show p
    end

end
