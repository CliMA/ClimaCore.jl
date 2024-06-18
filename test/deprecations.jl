#=
julia --project
using Revise; include(joinpath("test", "deprecations.jl"))
=#
using Test
using ClimaCore: Geometry, Quadratures, Domains, Meshes, Topologies, Spaces
import ClimaCore.Spaces:
    CenterFiniteDifferenceSpace,
    FaceFiniteDifferenceSpace,
    FiniteDifferenceSpace
import ClimaCore.Grids: FiniteDifferenceGrid
import ClimaComms
ClimaComms.@import_required_backends

@testset "Deprecations" begin
    FT = Float64
    context = ClimaComms.SingletonCommsContext()
    R = FT(6.371229e6)
    npoly = 3
    z_max = FT(30e3)
    z_elem = 64
    h_elem = 30
    # horizontal space
    domain = Domains.SphereDomain(R)
    horizontal_mesh = Meshes.EquiangularCubedSphere(domain, h_elem)
    horizontal_topology = Topologies.Topology2D(
        context,
        horizontal_mesh,
        Topologies.spacefillingcurve(horizontal_mesh),
    )
    quad = Quadratures.GLL{npoly + 1}()
    h_space = Spaces.SpectralElementSpace2D(horizontal_topology, quad)

    # vertical space
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)

    # deprecated methods:
    @test_deprecated Topologies.IntervalTopology(z_mesh)
    @test_deprecated FaceFiniteDifferenceSpace(z_mesh)
    @test_deprecated CenterFiniteDifferenceSpace(z_mesh)
    @test_deprecated FiniteDifferenceGrid(z_mesh)
end

nothing
