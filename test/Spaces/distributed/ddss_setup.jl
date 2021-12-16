using Logging
using Printf
using Test

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies

using ClimaComms
using ClimaCommsMPI
const Context = ClimaCommsMPI.MPICommsContext
const pid, nprocs = ClimaComms.init(Context)

puts(s...) = ccall(:puts, Cint, (Cstring,), string("$(pid)> ", s...))

# log output only from root process
logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

function distributed_space(
    (n1, n2),
    (x1periodic, x2periodic),
    (Nq, Nv, Nf);
    x1min = -2π,
    x1max = 2π,
    x2min = -2π,
    x2max = 2π,
)
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
    topology = Topologies.DistributedTopology2D(mesh, Context)
    quad = Spaces.Quadratures.GLL{Nq}()
    comms_ctx = Spaces.setup_comms(Context, topology, quad, Nv, Nf)
    space = Spaces.SpectralElementSpace2D(topology, quad, comms_ctx)

    return (space, comms_ctx)
end

function show_elements(Nq, y, nel)
    for el in 1:nel
        s = @sprintf "element %d: %s" el repr(
            "text/plain",
            reshape(parent(y)[:, :, 1, el], Nq, Nq),
        )
        puts(s)
    end
end
