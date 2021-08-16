using Test
using ClimaCore.Geometry, ClimaCore.DataLayouts
using LinearAlgebra, StaticArrays


x = Geometry.Covariant12Vector(1.0, 2.0)
y = Geometry.Contravariant12Vector(1.0, 4.0)

M = Geometry.Axis2Tensor(
    (Geometry.Cartesian12Axis(), Geometry.Covariant12Axis()),
    [1.0 0.0; 0.5 2.0],
)

@test dot(x, y) == x' * y == 9.0
@test dot(y, x) == y' * x == 9.0

@test x == x
@test x != Geometry.components(x)
@test x != Geometry.Contravariant12Vector(Geometry.components(x)...)

@test x[1] == 1.0
@test y[2] == 4.0
@test M[2] == 0.5
@test M[2, 1] == 0.5
@test M[:, 1] == Geometry.Cartesian12Vector(1.0, 0.5)
@test M[1, :] == Geometry.Covariant12Vector(1.0, 0.0)

@test x * y' ==
      x ⊗ y ==
      Geometry.AxisTensor(
          (axes(x, 1), axes(y, 1)),
          Geometry.components(x) * Geometry.components(y)',
      )

@test Geometry.components(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
@test Geometry.components(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

@test M * y == Geometry.Cartesian12Vector(1.0, 8.5)
@test M \ Geometry.Cartesian12Vector(1.0, 8.5) == y

@test_throws DimensionMismatch dot(x, x)
@test_throws DimensionMismatch M * x
@test_throws DimensionMismatch M \ x

@test DataLayouts.basetype(typeof(x)) == Float64
@test DataLayouts.typesize(Float64, typeof(x)) == 2
