using IntervalSets
using Logging
using StaticArrays
using Test

using ClimaComms
using ClimaCommsMPI
const Context = ClimaCommsMPI.MPICommsContext
const pid, nprocs = ClimaComms.init(Context)
comms_ctx = Context(ClimaComms.Neighbor[])

import ClimaCore: Domains, Meshes, Topologies
import ClimaCore.Geometry: Geometry

# log output only from root
logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

function distributed_grid(
    n1,
    n2,
    x1periodic,
    x2periodic;
    x1min = 0.0,
    x1max = 1.0,
    x2min = 0.0,
    x2max = 1.0,
)
    domain = Domains.RectangleDomain(
        Geometry.XPoint(x1min) .. Geometry.XPoint(x1max),
        Geometry.YPoint(x2min) .. Geometry.YPoint(x2max),
        x1periodic = x1periodic,
        x2periodic = x2periodic,
        x1boundary = x1periodic ? nothing : (:west, :east),
        x2boundary = x2periodic ? nothing : (:south, :north),
    )
    mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
    return Topologies.DistributedTopology2D(mesh, Context)
end

#=
 _ _ _ _
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
=#
@testset "4x4 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(4, 4, false, false)
    ifaces = sort(collect(Topologies.interior_faces(dtopo)))
    gfaces = sort(collect(Topologies.ghost_faces(dtopo)))
    passed = 0
    if pid == 1
        if length(ifaces) == 3
            passed += 1
        end
        if ifaces ==
           sort([(1, 2, 2, 4, true), (2, 2, 3, 4, true), (3, 2, 4, 4, true)])
            passed += 1
        end
        if length(gfaces) == 4
            passed += 1
        end
        if gfaces == sort([
            (1, 3, 5, 1, true),
            (2, 3, 6, 1, true),
            (3, 3, 7, 1, true),
            (4, 3, 8, 1, true),
        ])
            passed += 1
        end
    elseif pid == 2
        if length(ifaces) == 3
            passed += 1
        end
        if ifaces ==
           sort([(5, 2, 6, 4, true), (6, 2, 7, 4, true), (7, 2, 8, 4, true)])
            passed += 1
        end
        if length(gfaces) == 8
            passed += 1
        end
        if gfaces == sort([
            (5, 1, 1, 3, true),
            (5, 3, 9, 1, true),
            (6, 1, 2, 3, true),
            (6, 3, 10, 1, true),
            (7, 1, 3, 3, true),
            (7, 3, 11, 1, true),
            (8, 1, 4, 3, true),
            (8, 3, 12, 1, true),
        ])
            passed += 1
        end
    elseif pid == 3
        if length(ifaces) == 3
            passed += 1
        end
        if ifaces == sort([
            (9, 2, 10, 4, true),
            (10, 2, 11, 4, true),
            (11, 2, 12, 4, true),
        ])
            passed += 1
        end
        if length(gfaces) == 8
            passed += 1
        end
        if gfaces == sort([
            (9, 1, 5, 3, true),
            (9, 3, 13, 1, true),
            (10, 1, 6, 3, true),
            (10, 3, 14, 1, true),
            (11, 1, 7, 3, true),
            (11, 3, 15, 1, true),
            (12, 1, 8, 3, true),
            (12, 3, 16, 1, true),
        ])
            passed += 1
        end
    else
        if length(ifaces) == 3
            passed += 1
        end
        if ifaces == sort([
            (13, 2, 14, 4, true),
            (14, 2, 15, 4, true),
            (15, 2, 16, 4, true),
        ])
            passed += 1
        end
        if length(gfaces) == 4
            passed += 1
        end
        if gfaces == sort([
            (13, 1, 9, 3, true),
            (14, 1, 10, 3, true),
            (15, 1, 11, 3, true),
            (16, 1, 12, 3, true),
        ])
            passed += 1
        end
    end
    passed = ClimaComms.reduce(comms_ctx, passed, +)
    if pid == 1
        @test passed == 16
    end
end

#=
 _ _
|_|_|
|_|_|
=#
@testset "2x2 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(2, 2, false, false)
    ivs = collect(Topologies.interior_vertices(dtopo))
    gvs = sort(collect(Topologies.ghost_vertices(dtopo)))
    passed = 0
    if pid == 1
        if length(ivs) == 1
            passed += 1
        end
        if ivs[1] == [(1, 1)]
            passed += 1
        end
        if length(gvs) == 3
            passed += 1
        end
        if gvs == [[(1, 2), (2, 1)], [(1, 3), (2, 4), (3, 2), (4, 1)], [(1, 4), (3, 1)]]
            passed += 1
        end
    elseif pid == 2
        if length(ivs) == 1
            passed += 1
        end
        if ivs[1] == [(2, 2)]
            passed += 1
        end
        if length(gvs) == 3
            passed += 1
        end
        if gvs == [[(1, 2), (2, 1)], [(1, 3), (2, 4), (3, 2), (4, 1)], [(2, 3), (4, 2)]]
            passed += 1
        end
    elseif pid == 3
        if length(ivs) == 1
            passed += 1
        end
        if ivs[1] == [(3, 4)]
            passed += 1
        end
        if length(gvs) == 3
            passed += 1
        end
        if gvs == [[(1, 3), (2, 4), (3, 2), (4, 1)], [(1, 4), (3, 1)], [(3, 3), (4, 4)]]
            passed += 1
        end
    else
        if length(ivs) == 1
            passed += 1
        end
        if ivs[1] == [(4, 3)]
            passed += 1
        end
        if length(gvs) == 3
            passed += 1
        end
        if gvs == [[(1, 3), (2, 4), (3, 2), (4, 1)], [(2, 3), (4, 2)], [(3, 3), (4, 4)]]
            passed += 1
        end
    end
    passed = ClimaComms.reduce(comms_ctx, passed, +)
    if pid == 1
        @test passed == 16
    end
end

#=
 _ _  _ _  _ _  _ _
|_|_||_|_||_|_||_|_|
|_|_||_|_||_|_||_|_|
=#
@testset "2x8 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(2, 8, false, false)
    foi(e) = dtopo.orderindex[e]
    real_elems = map(foi, dtopo.real_elems)
    fse(se) = begin
        (id, elems) = se
        (id, sort(map(foi, elems)))
    end
    send_elems = sort(map(fse, dtopo.send_elems))
    ghost_elems = sort(map(fse, dtopo.ghost_elems))
    passed = 0
    if pid == 1
        if real_elems == [1, 2, 3, 4]
            passed += 1
        end
        if send_elems == [(2, [3, 4])]
            passed += 1
        end
        if ghost_elems == [(2, [5, 6])]
            passed += 1
        end
    elseif pid == 2
        if real_elems == [5, 6, 7, 8]
            passed += 1
        end
        if send_elems == [(1, [5, 6]), (3, [7, 8])]
            passed += 1
        end
        if ghost_elems == [(1, [3, 4]), (3, [9, 10])]
            passed += 1
        end
    elseif pid == 3
        if real_elems == [9, 10, 11, 12]
            passed += 1
        end
        if send_elems == [(2, [9, 10]), (4, [11, 12])]
            passed += 1
        end
        if ghost_elems == [(2, [7, 8]), (4, [13, 14])]
            passed += 1
        end
    else
        if real_elems == [13, 14, 15, 16]
            passed += 1
        end
        if send_elems == [(3, [13, 14])]
            passed += 1
        end
        if ghost_elems == [(3, [11, 12])]
            passed += 1
        end
    end
    passed = ClimaComms.reduce(comms_ctx, passed, +)
    if pid == 1
        @test passed == 12
    end
end