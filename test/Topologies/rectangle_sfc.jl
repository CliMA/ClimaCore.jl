using IntervalSets
using Test
using ClimaComms

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies

function rectilinear_grid(
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
    return Meshes.RectilinearMesh(domain, n1, n2)
end

@testset "Space Filling Curve tests for a 4x4 element quad mesh with non-periodic boundaries" begin
    mesh = rectilinear_grid(4, 4, false, false)
    sfc_elemorder = Topologies.spacefillingcurve(mesh)
    sfc_orderindex = Meshes.linearindices(sfc_elemorder)
    nelems = Meshes.nelements(mesh)

    @testset "connectivity test" begin
        vconn = Meshes.vertex_connectivity_matrix(
            mesh,
            sfc_elemorder,
            sfc_orderindex,
        )

        connected = true

        for i in 1:(nelems - 1)
            oi = sfc_orderindex[sfc_elemorder[i]]  # order index for current element on SFC
            neigh_oi = sfc_orderindex[sfc_elemorder[i + 1]]  # order index for next element on SFC
            !vconn[oi, neigh_oi] && (connected = false)
        end
        @test connected
    end

    @testset "inclusiveness test" begin
        elemorder = Meshes.elements(mesh)
        @test sort(sfc_elemorder[:]) == sort(elemorder[:])
    end

    @testset "linearindices test" begin
        sfc_orderindex = Meshes.linearindices(sfc_elemorder)
        for (order, cartindex) in enumerate(sfc_elemorder)
            @test sfc_orderindex[cartindex] == order
        end
    end
end

@testset "uses_spacefillingcurve tests for rectangular mesh" begin
    device = ClimaComms.CPUSingleThreaded()
    context = ClimaComms.SingletonCommsContext(device)

    @testset "rectangular mesh with space-filling curve" begin
        mesh = rectilinear_grid(4, 4, false, false)
        topology_sfc = Topologies.Topology2D(
            context,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        @test Topologies.uses_spacefillingcurve(topology_sfc) == true
    end

    @testset "rectangular mesh with default ordering" begin
        mesh = rectilinear_grid(4, 4, false, false)
        topology_default = Topologies.Topology2D(context, mesh)
        @test Topologies.uses_spacefillingcurve(topology_default) == false
    end
end
