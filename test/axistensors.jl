using Test
using ClimaCore: Geometry, DataLayouts
using LinearAlgebra, StaticArrays



x = Geometry.Covariant12Vector(1.0, 2.0)
y = Geometry.Contravariant12Vector(1.0, 4.0)

M = Geometry.AxisTensor(
    (Geometry.Cartesian12Axis(), Geometry.Covariant12Axis()),
    SMatrix{2, 2}(1.0, 0.5, 0.0, 2.0),
)

@test dot(x, y) == x' * y == 9.0
@test dot(y, x) == y' * x == 9.0

@test x == x

@test M * y == Geometry.Cartesian12Vector(1.0, 8.5)

@test_throws DimensionMismatch dot(x, x)

@test DataLayouts.basetype(typeof(x)) == Float64
@test DataLayouts.typesize(Float64, typeof(x)) == 2
