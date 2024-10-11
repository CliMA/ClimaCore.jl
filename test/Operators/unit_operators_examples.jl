import ClimaCore:
    Domains,
    Meshes,
    Geometry,
    Grids,
    Spaces,
    Topologies,
    Hypsography,
    Fields,
    Operators,
    Utilities
import ClimaComms

ClimaComms.@import_required_backends
device = ClimaComms.device()
comms_ctx = ClimaComms.context(device)

h_domain = Domains.RectangleDomain(
    Domains.IntervalDomain(
        Geometry.XPoint(0.0),
        Geometry.XPoint(1.0);
        periodic = true,
    ),
    Domains.IntervalDomain(
        Geometry.YPoint(0.0),
        Geometry.YPoint(1.0);
        periodic = true,
    ),
)
h_mesh = Meshes.RectilinearMesh(h_domain, 10, 10)
h_grid = Spaces.grid(
    Spaces.SpectralElementSpace2D(
        Topologies.DistributedTopology2D(
            comms_ctx,
            h_mesh,
            Topologies.spacefillingcurve(h_mesh),
        ),
        Spaces.Quadratures.GLL{4}(),
    ),
)
z_domain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(1.0);
    boundary_names = (:bottom, :top),
)
z_grid = Grids.FiniteDifferenceGrid(
    Topologies.IntervalTopology(
        comms_ctx,
        Meshes.IntervalMesh(z_domain, Meshes.Uniform(); nelems = 10),
    ),
)
grid = Grids.ExtrudedFiniteDifferenceGrid(
    h_grid,
    z_grid,
    Hypsography.Flat();
    deep = false,
)
center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)

# Create fields and show that it fails
ᶜgradᵥ = Operators.GradientF2C()

level_field = Fields.level(Fields.Field(Float64, face_space), Utilities.half)
ᶠscalar_field = Fields.Field(Float64, face_space)
# Does not work:
using Test

@testset "Broken broadcast expression on GPUs" begin
    if device isa ClimaComms.CUDADevice
        @test_broken begin
            @. ᶜgradᵥ(level_field + ᶠscalar_field)
        end
    else
        @. ᶜgradᵥ(level_field + ᶠscalar_field)
    end
end
