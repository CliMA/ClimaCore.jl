using Logging
using Test

import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Meshes,
    Operators,
    Spaces,
    Quadratures,
    Topologies,
    DataLayouts,
    Grids

using ClimaComms
ClimaComms.@import_required_backends
device = ClimaComms.device()
using ClimaComms
using IntervalSets

FT = Float64
xlim = (FT(0), FT(2π))
zlim = (FT(0), FT(1))
helem = 5
velem = 5
npoly = 5
stretch = Meshes.Uniform()
comms_context = ClimaComms.SingletonCommsContext(device)
FT = eltype(xlim)

# Horizontal Grid Construction
quad = Quadratures.GLL{npoly + 1}()
horzdomain = Domains.RectangleDomain(
    Geometry.XPoint{FT}(xlim[1]) .. Geometry.XPoint{FT}(xlim[2]),
    Geometry.YPoint{FT}(xlim[1]) .. Geometry.YPoint{FT}(xlim[2]),
    x1periodic = true,
    x2periodic = true,
)
# Assume same number of elems (helem) in (x,y) directions
horzmesh = Meshes.RectilinearMesh(horzdomain, helem, helem)
horz_topology = Topologies.Topology2D(
    comms_context,
    horzmesh,
    Topologies.spacefillingcurve(horzmesh),
);
h_space =
    Spaces.SpectralElementSpace2D(horz_topology, quad, enable_bubble = true);
horz_grid = Spaces.grid(h_space)

# Vertical Grid Construction
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(zlim[1]),
    Geometry.ZPoint{FT}(zlim[2]);
    boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, stretch, nelems = velem)
vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)
vert_topology = Topologies.IntervalTopology(
    ClimaComms.SingletonCommsContext(device),
    vertmesh,
)
vert_grid = Grids.FiniteDifferenceGrid(vert_topology)

grid = Grids.ExtrudedFiniteDifferenceGrid(horz_grid, vert_grid)
cent_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
ccoords = Fields.coordinate_field(cent_space)
fcoords = Fields.coordinate_field(face_space)

∇ = Operators.Gradient();

η = @. sin(ccoords.x) + cos(ccoords.y)
η_test1 = @. Geometry.project(Geometry.UVAxis(), ∇(η)).components.data.:1
η_test2 = @. Geometry.project(Geometry.UVAxis(), ∇(η)).components.data.:2
Spaces.weighted_dss!(η_test1)
Spaces.weighted_dss!(η_test2)

𝒻₁ = @. Geometry.UVVector(η, 2η)
𝒻₂ = @. Geometry.UVWVector(η, 2η, 3η)

∇η = @. ∇(η)
∇𝒻₁ = @. Geometry.project(Geometry.UVAxis(), ∇(𝒻₁))
for ii in 1:4
    Spaces.weighted_dss!(∇𝒻₁.components.data.:($ii))
end

# Check against known solution component-wise
device isa ClimaComms.CUDADevice ? CUDA.allowscalar(true) : nothing
@test parent(η_test1) ≈ parent(∇𝒻₁.components.data.:1)
@test parent(η_test2) ≈ parent(∇𝒻₁.components.data.:2)
@test parent(2 .* η_test1) ≈ parent(∇𝒻₁.components.data.:3)
@test parent(2 .* η_test2) ≈ parent(∇𝒻₁.components.data.:4)

∇𝒻₂ = @. Geometry.project(Geometry.UVAxis(), ∇(𝒻₂))
for ii in 1:6
    Spaces.weighted_dss!(∇𝒻₂.components.data.:($ii))
end

# Check against known solution component-wise
@test parent(η_test1) ≈ parent(∇𝒻₂.components.data.:1)
@test parent(η_test2) ≈ parent(∇𝒻₂.components.data.:2)
@test parent(2 .* η_test1) ≈ parent(∇𝒻₂.components.data.:3)
@test parent(2 .* η_test2) ≈ parent(∇𝒻₂.components.data.:4)
@test parent(3 .* η_test1) ≈ parent(∇𝒻₂.components.data.:5)
@test parent(3 .* η_test2) ≈ parent(∇𝒻₂.components.data.:6)
