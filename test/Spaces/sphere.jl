using LinearAlgebra, IntervalSets
using ClimaComms
import ClimaCore:
    Domains, Topologies, Meshes, Grids, Spaces, Geometry, column, Quadratures

using Test

@testset "Sphere" begin
    for FT in (Float64, Float32)
        context = ClimaComms.SingletonCommsContext()
        device = ClimaComms.device(context)
        radius = FT(3)
        ne = 4
        Nq = 4
        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, ne)
        topology = Topologies.Topology2D(context, mesh)
        quad = Quadratures.GLL{Nq}()
        space = Spaces.SpectralElementSpace2D(topology, quad)

        @test Spaces.n_elements_per_panel_direction(space) == ne

        # surface area
        @test sum(ones(space)) ≈ FT(4pi * radius^2) rtol = 1e11 * eps(FT)

        enable_bubble = false
        no_bubble_space =
            Spaces.SpectralElementSpace2D(topology, quad; enable_bubble)

        # Now check that constructor with enable_buble = false falls back on existing behavior
        @test sum(ones(no_bubble_space)) ≈ FT(4pi * radius^2) rtol =
            1e11 * eps(FT)

        # Now check constructor with bubble enabled
        enable_bubble = true
        bubble_space =
            Spaces.SpectralElementSpace2D(topology, quad; enable_bubble)
        @test sum(ones(bubble_space)) ≈ FT(4pi * radius^2) rtol = 1e3 * eps(FT)

        # vertices with multiplicity 3
        nn3 = 8 # corners of cube
        # vertices with multiplicity 4
        nn4 = 6 * ne^2 - 6 # (6*ne^2*4 - 8*3)/4
        # internal nodes on edges: multiplicity 2
        nn2 = 6 * ne^2 * (Nq - 2) * 2
        # total nodes
        nn = 6 * ne^2 * Nq^2
        # unique nodes
        if device isa ClimaComms.AbstractCPUDevice
            @test length(collect(Spaces.unique_nodes(space))) ==
                  nn - nn2 - 2 * nn3 - 3 * nn4
        end

        point_space = column(space, 1, 1, 1)
        @test point_space isa Spaces.PointSpace
        ClimaComms.allowscalar(device) do
            @test Spaces.coordinates_data(point_space)[] ==
                  column(Spaces.coordinates_data(space), 1, 1, 1)[]
        end
    end
end

@testset "Bubble correction Nq robustness" begin

    for FT in (Float64, Float32)
        # Reference rtols without bubble
        rtols = [FT(35) * FT(0.5)^(FT(2.5) * Nq) for Nq in 2:10]


        for (k, Nq) in enumerate(2:10)
            context = ClimaComms.SingletonCommsContext()
            radius = FT(3)
            ne = 1
            domain = Domains.SphereDomain(radius)
            mesh = Meshes.EquiangularCubedSphere(domain, ne)
            topology = Topologies.Topology2D(context, mesh)
            quad = Quadratures.GLL{Nq}()
            no_bubble_space = Spaces.SpectralElementSpace2D(topology, quad)
            # check surface area
            @test sum(ones(no_bubble_space)) ≈ FT(4pi * radius^2) rtol =
                rtols[k]

            bubble_space = Spaces.SpectralElementSpace2D(
                topology,
                quad;
                enable_bubble = true,
            )
            @test sum(ones(bubble_space)) ≈ FT(4pi * radius^2) rtol = rtols[k]
        end
    end

end

@testset "Volume of a spherical shell" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext()
    radius = FT(10)
    zlim = (0, 1)
    helem = 4
    zelem = 10
    Nq = 4

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    device = ClimaComms.device(context)
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vertgrid = Grids.FiniteDifferenceGrid(device, vertmesh)

    horzdomain = Domains.SphereDomain(radius)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{Nq}()
    horzgrid = Grids.SpectralElementGrid2D(horztopology, quad)
    horzspace = Spaces.SpectralElementSpace2D(horzgrid)
    @test Spaces.node_horizontal_length_scale(horzspace) ≈
          sqrt((4 * pi * radius^2) / (helem^2 * 6 * (Nq - 1)^2))

    # "shallow atmosphere" spherical shell: volume = surface area * height
    shallow_grid = Grids.ExtrudedFiniteDifferenceGrid(horzgrid, vertgrid)

    @test sum(ones(Spaces.CenterExtrudedFiniteDifferenceSpace(shallow_grid))) ≈
          4pi * radius^2 * (zlim[2] - zlim[1]) rtol = 1e-3
    @test sum(ones(Spaces.FaceExtrudedFiniteDifferenceSpace(shallow_grid))) ≈
          4pi * radius^2 * (zlim[2] - zlim[1]) rtol = 1e-3

    deep_grid =
        Grids.ExtrudedFiniteDifferenceGrid(horzgrid, vertgrid; deep = true)

    @test sum(ones(Spaces.CenterExtrudedFiniteDifferenceSpace(deep_grid))) ≈
          4pi / 3 * ((zlim[2] + radius)^3 - (zlim[1] + radius)^3) rtol = 1e-3
    @test sum(ones(Spaces.FaceExtrudedFiniteDifferenceSpace(deep_grid))) ≈
          4pi / 3 * ((zlim[2] + radius)^3 - (zlim[1] + radius)^3) rtol = 1e-3

end
