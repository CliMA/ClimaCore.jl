using LinearAlgebra, IntervalSets, UnPack
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, column

using Test

@testset "Sphere" begin
    FT = Float64
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(topology, quad)

    # surface area
    @test sum(ones(space)) ≈ 4pi * radius^2 rtol = 1e-3

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

@testset "Volume of a spherical shell" begin
    FT = Float64
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
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)

    # "shallow atmosphere" spherical shell: volume = surface area * height
    @test sum(ones(hv_center_space)) ≈ 4pi * radius^2 * (zlim[2] - zlim[1]) rtol =
        1e-3
end
