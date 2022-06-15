using Test

import ClimaCore:
    Domains, Fields, Geometry, Meshes, Operators, Spaces, Topologies

@testset "Space Filling Curve tests for a cubed sphere mesh with (16) elements per edge" begin
    mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain(1.0), 16)
    sfc_elemorder = Topologies.spacefillingcurve(mesh)
    sfc_orderindex = Meshes.linearindices(sfc_elemorder)

    @testset "connectivity test" begin
        vconn = Meshes.vertex_connectivity_matrix(
            mesh,
            sfc_elemorder,
            sfc_orderindex,
        )
        nelems = Meshes.nelements(mesh)

        connected = true

        for i in 1:(nelems - 1)
            oi = sfc_orderindex[sfc_elemorder[i]]  # order index for current element on SFC
            neigh_oi = sfc_orderindex[sfc_elemorder[i + 1]]  # order index for next element on SFC
            !vconn[oi, neigh_oi] && (connected = false)
        end
        @test connected
    end

    @testset "inclusivity test" begin
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
