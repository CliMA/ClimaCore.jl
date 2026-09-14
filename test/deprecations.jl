using Test
import ClimaCore
using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces, Grids
import ClimaCore.Spaces: CenterFiniteDifferenceSpace, FiniteDifferenceSpace
import ClimaComms
ClimaComms.@import_required_backends

@testset "Deprecations" begin
    FT = Float64
    z_max = FT(30e3)
    z_elem = 64
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, nelems = z_elem)

    device = ClimaComms.device()
    grid = Grids.FiniteDifferenceGrid(Topologies.IntervalTopology(device, z_mesh))

    @test_deprecated CenterFiniteDifferenceSpace(z_mesh)
    @test CenterFiniteDifferenceSpace(z_mesh) ==
          FiniteDifferenceSpace(grid, Grids.CellCenter())
end

nothing
