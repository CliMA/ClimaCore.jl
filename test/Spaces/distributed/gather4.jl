using Test
using LinearAlgebra

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Topologies,
    DataLayouts

using Logging

ENV["CLIMACORE_DISTRIBUTED"] = "MPI"

using ClimaComms
using ClimaCommsMPI
const comms_ctx = ClimaCommsMPI.MPICommsContext()
const pid, nprocs = ClimaComms.init(comms_ctx)

# log output only from root process
logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull

prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(-2π),
        Geometry.XPoint(2π),
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(-2π),
        Geometry.YPoint(2π),
        periodic = true,
    ),
)
n1, n2 = 16, 16
Nq = 4
quad = Spaces.Quadratures.GLL{Nq}()
mesh = Meshes.RectilinearMesh(domain, n1, n2)

grid_topology = Topologies.DistributedTopology2D(comms_ctx, mesh)
global_grid_topology = Topologies.Topology2D(mesh)
Nf = 4
Nv = 1
space = Spaces.SpectralElementSpace2D(grid_topology, quad)
global_space = Spaces.SpectralElementSpace2D(global_grid_topology, quad)

gathered_coord = DataLayouts.gather(
    comms_ctx,
    Fields.field_values(Fields.coordinate_field(space)),
)

@testset "gathering coordinate field" begin
    if pid == 1
        diff = maximum(
            abs.(
                parent(gathered_coord) .-
                parent(Fields.coordinate_field(global_space)),
            ),
        )
        @test diff ≈ 0.0
    end
end
