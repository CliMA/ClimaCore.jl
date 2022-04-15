using IntervalSets
using Logging
using Test

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies

using ClimaComms
using ClimaCommsMPI
comms_ctx = ClimaCommsMPI.MPICommsContext()
pid, nprocs = ClimaComms.init(comms_ctx)

# log output only from root process
logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull
prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
atexit() do
    global_logger(prev_logger)
end

function distributed_grid(
    comms_ctx,
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
    mesh = Meshes.RectilinearMesh(domain, n1, n2)
    return Topologies.DistributedTopology2D(comms_ctx, mesh)
end

function strip_extra_gface_info(gfacesin)
    gfacesout = Tuple{Int, Int, Int, Int, Bool}[]
    for gfaces in gfacesin
        push!(gfacesout, gfaces[1:5])
    end
    return gfacesout
end

function strip_extra_gvert_info(gvertsin)
    gvertsout = Vector{Tuple{Int, Int}}[]
    for verts in gvertsin
        gvert_group = Tuple{Int, Int}[]
        for (e, vertex_num, _, _) in verts
            push!(gvert_group, (e, vertex_num))
        end
        push!(gvertsout, gvert_group)
    end
    return gvertsout
end

#=
 _ _ _ _
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
|_|_|_|_|
=#
@testset "4x4 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(comms_ctx, 4, 4, false, false)
    ifaces = sort(collect(Topologies.interior_faces(dtopo)))
    gfaces = sort(collect(Topologies.ghost_faces(dtopo)))
    gfacesout = strip_extra_gface_info(gfaces)

    @test length(ifaces) == 3
    @test ifaces ==
          sort([(1, 2, 2, 4, true), (2, 2, 3, 4, true), (3, 2, 4, 4, true)])
    if pid == 1 || pid == 4
        @test length(gfaces) == 4
    else
        @test length(gfaces) == 8
    end
    if pid == 1
        @test gfacesout == sort([
            (1, 3, 1, 1, true),
            (2, 3, 2, 1, true),
            (3, 3, 3, 1, true),
            (4, 3, 4, 1, true),
        ])
    elseif pid == 2 || pid == 3
        @test gfacesout == sort([
            (1, 1, 1, 3, true),
            (2, 1, 2, 3, true),
            (3, 1, 3, 3, true),
            (4, 1, 4, 3, true),
            (1, 3, 5, 1, true),
            (2, 3, 6, 1, true),
            (3, 3, 7, 1, true),
            (4, 3, 8, 1, true),
        ])
    else
        @test gfacesout == sort([
            (1, 1, 1, 3, true),
            (2, 1, 2, 3, true),
            (3, 1, 3, 3, true),
            (4, 1, 4, 3, true),
        ])
    end
end

#=
 _ _
|_|_|
|_|_|
=#
@testset "2x2 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(comms_ctx, 2, 2, false, false)
    ivs = map(sort ∘ collect, Topologies.local_vertices(dtopo))
    gvs = map(sort ∘ collect, Topologies.ghost_vertices(dtopo))

    if pid == 1
        # 2 3
        # x 1
        @test ivs == [[(1, 1)]]
        @test gvs == [
            sort([(false, 1, 2), (true, 1, 1)]),
            sort([(false, 1, 3), (true, 1, 4), (true, 2, 2), (true, 3, 1)]),
            sort([(false, 1, 4), (true, 2, 1)]),
        ]
    elseif pid == 2
        # 2 3
        # 1 x
        @test ivs == [[(1, 2)]]
        @test gvs == [
            sort([(false, 1, 1), (true, 1, 2)]),
            sort([(false, 1, 3), (true, 3, 2)]),
            sort([(false, 1, 4), (true, 1, 3), (true, 2, 2), (true, 3, 1)]),
        ]
    elseif pid == 3
        # x 3
        # 1 2
        @test ivs == [[(1, 4)]]
        @test gvs == [
            sort([(false, 1, 1), (true, 1, 4)]),
            sort([(false, 1, 2), (true, 1, 3), (true, 2, 4), (true, 3, 1)]),
            sort([(false, 1, 3), (true, 3, 4)]),
        ]
    else
        # 3 x
        # 1 2
        @test ivs == [[(1, 3)]]
        @test gvs == [
            sort([(false, 1, 1), (true, 1, 3), (true, 2, 4), (true, 3, 2)]),
            sort([(false, 1, 2), (true, 2, 3)]),
            sort([(false, 1, 4), (true, 3, 3)]),
        ]
    end
end

#=
 _ _
|_|_|
|_|_|
|_|_|
|_|_|
|_|_|
|_|_|
|_|_|
|_|_|
=#
@testset "2x8 element quad mesh with non-periodic boundaries" begin
    dtopo = distributed_grid(comms_ctx, 2, 8, false, false)
    local_elems = dtopo.local_elem_gidx
    send_elems = dtopo.local_elem_gidx[dtopo.send_elem_lidx]
    ghost_elems = dtopo.recv_elem_gidx
    if pid == 1
        @test local_elems == [1, 2, 3, 4]
        @test send_elems == [3, 4]
        @test ghost_elems == [5, 6]
    elseif pid == 2
        @test local_elems == [5, 6, 7, 8]
        @test send_elems == [5, 6, 7, 8]
        @test ghost_elems == [3, 4, 9, 10]
    elseif pid == 3
        @test local_elems == [9, 10, 11, 12]
        @test send_elems == [9, 10, 11, 12]
        @test ghost_elems == [7, 8, 13, 14]
    else
        @test local_elems == [13, 14, 15, 16]
        @test send_elems == [13, 14]
        @test ghost_elems == [11, 12]
    end
end
