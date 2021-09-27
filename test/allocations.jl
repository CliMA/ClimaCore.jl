import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators

using Test

Nv = 100

vertdomain = Domains.IntervalDomain(
    Geometry.ZPoint(0.0),
    Geometry.ZPoint(1000.0),
    boundary_tags = (:bottom, :top),
)

vertmesh = Meshes.IntervalMesh(vertdomain, nelems = Nv)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
vert_face_space = Spaces.FaceFiniteDifferenceSpace(vert_center_space)

x = map(Fields.coordinate_field(vert_face_space)) do z
    Geometry.Cartesian3Vector(1.0)
end
divx = ones(Float64, vert_center_space)

function f!(divx, x)
    divf = Operators.DivergenceF2C()
    # works without negation
    @. divx = -divf(x)
    return nothing
end

f!(divx, x)
divx = ones(Float64, vert_center_space)
if VERSION < v"1.7.0-beta1"
    @test_broken @allocated(f!(divx, x)) == 0
else
    @test @allocated(f!(divx, x)) == 0
end

function g!(divx, x)
    divf = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(1.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(1.0)),
    )
    divx .= divf.(x)
    return nothing
end

divx = ones(Float64, vert_center_space)
g!(divx, x)
divx = ones(Float64, vert_center_space)
@test @allocated(g!(divx, x)) == 0
