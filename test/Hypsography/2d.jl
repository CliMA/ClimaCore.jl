using Test
import ClimaCore
import ClimaComms
import ClimaCore:
    ClimaCore,
    slab,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Quadratures,
    Grids,
    Fields,
    Operators,
    InputOutput,
    Hypsography
using ClimaCore.Geometry

FT = Float64
context = ClimaComms.context()
device = ClimaComms.device(context)
vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(4);
    boundary_names = (:bottom, :top),
)
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 40)
z_topology = Topologies.IntervalTopology(context, vertmesh)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(z_topology)



horzdomain = Domains.IntervalDomain(
    Geometry.XPoint{FT}(0),
    Geometry.XPoint{FT}(4pi);
    periodic = true,
)
horzmesh = Meshes.IntervalMesh(horzdomain, nelems = 20)
horztopology = Topologies.IntervalTopology(device, horzmesh)
quad = Quadratures.GLL{4 + 1}()
horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

z_surface = Geometry.ZPoint.(sin.(Fields.coordinate_field(horzspace).x) .+ 1)

hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
    horzspace,
    vert_center_space,
    Hypsography.LinearAdaption(z_surface),
)
hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

grad_h = Operators.Gradient()
grad_v = Operators.GradientF2C()

center_z = Fields.coordinate_field(hv_center_space).z
face_z = Fields.coordinate_field(hv_face_space).z
∇z_h = Geometry.UWVector.(grad_h.(center_z))
∇z_v = Geometry.UWVector.(grad_v.(face_z))
∇z = ∇z_h .+ ∇z_v

@test maximum(u -> abs(u.u), ∇z) < 1e-4
@test map(u -> u.w, ∇z) ≈ ones(hv_center_space)

center_x = Fields.coordinate_field(hv_center_space).x
face_x = Fields.coordinate_field(hv_face_space).x
∇x_h = Geometry.UWVector.(grad_h.(center_x))
∇x_v = Geometry.UWVector.(grad_v.(face_x))
∇x = ∇x_h .+ ∇x_v

@test map(u -> u.u, ∇x) ≈ ones(hv_center_space)
@test maximum(u -> abs(u.w), ∇x) < 1e-10

# Check transformations back and forth
z_ref1 = Geometry.ZPoint{FT}(40.0)
z_top1 = Geometry.ZPoint{FT}(400.0)
z_surface1 = Geometry.ZPoint{FT}(10.0)

point_space = Spaces.PointSpace(z_surface1)
z_surface_point = Fields.coordinate_field(point_space)

@test Hypsography.physical_z_to_ref_z(
    Grids.Flat(),
    Hypsography.ref_z_to_physical_z(Grids.Flat(), z_ref1, z_surface1, z_top1),
    z_surface1,
    z_top1,
).z ≈ z_ref1.z

# Linear adaption
linear_adaption = Hypsography.LinearAdaption(z_surface_point)
@test Hypsography.physical_z_to_ref_z(
    linear_adaption,
    Hypsography.ref_z_to_physical_z(
        linear_adaption,
        z_ref1,
        z_surface1,
        z_top1,
    ),
    z_surface1,
    z_top1,
).z ≈ z_ref1.z
