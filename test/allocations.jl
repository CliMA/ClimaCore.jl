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

vertdomain = Domains.IntervalDomain(0.0, 1000.0; x3boundary = (:bottom, :top))
vertmesh = Meshes.IntervalMesh(vertdomain, nelems = Nv)
vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

x = map(Fields.coordinate_field(vert_center_space)) do z
    Geometry.Cartesian3Vector(1.0)
end
divx = ones(Float64, Spaces.FaceFiniteDifferenceSpace(vert_center_space))

function f!(divx, x)
    divf = Operators.DivergenceC2F(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(1.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(1.0)),
    )
    divx .= divf.(x)
    return nothing
end

f!(divx, x)
@test @allocated(f!(divx, x)) == 0
