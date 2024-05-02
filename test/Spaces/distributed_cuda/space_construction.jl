using Logging
using Test
using ClimaComms
ClimaComms.@import_required_backends
using CUDA

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    Quadratures

@testset "Distributed extruded CUDA space" begin
    # initializing MPI
    device = ClimaComms.device()
    context = ClimaComms.MPICommsContext(device)
    vcontext = ClimaComms.SingletonCommsContext(device)
    pid, nprocs = ClimaComms.init(context)
    iamroot = ClimaComms.iamroot(context)
    if iamroot
        println("running test on $device device with $nprocs processes")
    end

    FT = Float64
    radius = FT(128)
    zlim = (0, 1)
    helem = 4
    Nq = 4
    zelem = 10
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vtopology = Topologies.IntervalTopology(vcontext, vertmesh)
    vspace = Spaces.CenterFiniteDifferenceSpace(vtopology)

    hdomain = Domains.SphereDomain(radius)
    hmesh = Meshes.EquiangularCubedSphere(hdomain, helem)
    htopology = Topologies.Topology2D(context, hmesh)
    quad = Quadratures.GLL{Nq}()
    hspace = Spaces.SpectralElementSpace2D(htopology, quad)
    space = Spaces.ExtrudedFiniteDifferenceSpace(hspace, vspace)

end
