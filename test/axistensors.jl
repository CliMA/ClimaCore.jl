using Test
using ClimaCore: Geometry, DataLayouts
using LinearAlgebra



x = Geometry.Covariant12Vector(1.0,2.0)
y = Geometry.Contravariant12Vector(1.0,3.0)

@test dot(x,y) == x'*y == 7.0
@test dot(y,x) == y'*x == 7.0

@test x == x

@test_throws DimensinoMismatch dot(x,x)

@test DataLayouts.basetype(typeof(x)) == Float64
@test DataLayouts.typesize(Float64, typeof(x)) == 2