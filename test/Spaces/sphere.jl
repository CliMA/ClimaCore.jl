using LinearAlgebra, IntervalSets, UnPack
using ClimaComms
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, column

using Test

@testset "Sphere" begin
    for FT in (Float64, Float32)
        context = ClimaComms.SingletonCommsContext()
        radius = FT(3)
        ne = 4
        Nq = 4
        domain = Domains.SphereDomain(radius)
        mesh = Meshes.EquiangularCubedSphere(domain, ne)
        topology = Topologies.Topology2D(context, mesh)
        quad = Spaces.Quadratures.GLL{Nq}()
        space = Spaces.SpectralElementSpace2D(topology, quad)

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
        @test length(collect(Spaces.unique_nodes(space))) ==
              nn - nn2 - 2 * nn3 - 3 * nn4

        point_space = column(space, 1, 1, 1)
        @test point_space isa Spaces.PointSpace
        @test Spaces.coordinates_data(point_space)[] ==
              column(Spaces.coordinates_data(space), 1, 1, 1)[]
    end
end

@testset "Bubble correction Nq robustness" begin

    for FT in (Float64, Float32)
        # Reference rtols without bubble
        no_bubble_rtols_f64 = (
            FT(0.64),
            FT(0.19),
            FT(0.027),
            FT(0.0049),
            FT(0.0008),
            FT(0.00014),
            FT(2.4e-5),
            FT(3.97e-6),
            FT(6.77e-7),
        )
        no_bubble_rtols_f32 = (
            FT(0.64f0),
            FT(0.19f0),
            FT(0.027f0),
            FT(0.0049f0),
            FT(0.0008f0),
            FT(0.00014f0),
            FT(2.4f-5),
            FT(3.65f-6),
            FT(4.05f-7),
        )
        # Reference rtols with bubble
        bubble_rtols_f32 = (
            FT(0.027),
            FT(0.0008),
            FT(2.4f-5),
            FT(4.05f-7),
            FT(1.35f-6),
            FT(6.75f-8),
            FT(1.08f-6),
            FT(4.05f-7),
            FT(6.75f-8),
        )
        bubble_rtols_f64 = (
            FT(0.027),
            FT(0.0008),
            FT(2.33e-5),
            FT(6.77e-7),
            FT(1.98e-8),
            FT(5.78e-10),
            FT(1.7e-11),
            FT(4.94e-13),
            FT(1.53e-14),
        )

        for (k, Nq) in enumerate(2:10)
            context = ClimaComms.SingletonCommsContext()
            radius = FT(3)
            ne = 1
            domain = Domains.SphereDomain(radius)
            mesh = Meshes.EquiangularCubedSphere(domain, ne)
            topology = Topologies.Topology2D(context, mesh)
            quad = Spaces.Quadratures.GLL{Nq}()
            no_bubble_space = Spaces.SpectralElementSpace2D(topology, quad)
            # check surface area
            if FT == Float32
                @test sum(ones(no_bubble_space)) ≈ FT(4pi * radius^2) rtol =
                    no_bubble_rtols_f32[k]
            else
                @test sum(ones(no_bubble_space)) ≈ FT(4pi * radius^2) rtol =
                    no_bubble_rtols_f64[k]
            end

            bubble_space = Spaces.SpectralElementSpace2D(
                topology,
                quad;
                enable_bubble = true,
            )

            if FT == Float32
                @test sum(ones(bubble_space)) ≈ FT(4pi * radius^2) rtol =
                    bubble_rtols_f32[k]
            else
                @test sum(ones(bubble_space)) ≈ FT(4pi * radius^2) rtol =
                    bubble_rtols_f64[k]
            end
        end
    end

end

@testset "Volume of a spherical shell" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext()
    radius = FT(128)
    zlim = (0, 1)
    helem = 4
    zelem = 10
    Nq = 4

    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(radius)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    # "shallow atmosphere" spherical shell: volume = surface area * height
    @test sum(ones(hv_center_space)) ≈ 4pi * radius^2 * (zlim[2] - zlim[1]) rtol =
        1e-3
end
