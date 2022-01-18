using LinearAlgebra, IntervalSets, UnPack
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry

using Test

@testset "Surface area of a sphere" begin
    FT = Float64
    radius = FT(3)
    ne = 4
    Nq = 4
    domain = Domains.SphereDomain(radius)
    mesh = Meshes.EquiangularCubedSphere(domain, ne)
    grid_topology = Topologies.Topology2D(mesh)
    quad = Spaces.Quadratures.GLL{Nq}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    @test sum(ones(space)) ≈ 4pi * radius^2 rtol = 1e-3
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
