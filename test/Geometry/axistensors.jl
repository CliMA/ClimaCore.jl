using Test
using ClimaCore.Geometry, ClimaCore.DataLayouts
using LinearAlgebra, StaticArrays
import ClimaCore
ClimaCore.Geometry.assert_exact_transform() = true

@testset "AxisTensors" begin
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

    @test x + zero(x) == x
    @test x' + zero(x') == x'

    @test -x + x * 2 - x / 2 == -x + 2 * x - 2 \ x == x / 2
    @test -x' + x' * 2 - x' / 2 == -x' + 2 * x' - 2 \ x' == (x / 2)'

    @test x * y' ==
          x ⊗ y ==
          Geometry.AxisTensor(
              (axes(x, 1), axes(y, 1)),
              Geometry.components(x) * Geometry.components(y)',
          )

    @test Geometry.components(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
    @test Geometry.components(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

    @test x ⊗ 3 == Geometry.Covariant12Vector(3.0, 6.0)
    @test x ⊗ (1, (a = 2, b = 3)) == (
        Geometry.Covariant12Vector(1.0, 2.0),
        (
            a = Geometry.Covariant12Vector(2.0, 4.0),
            b = Geometry.Covariant12Vector(3.0, 6.0),
        ),
    )


    @test Geometry.components(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
    @test Geometry.components(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

    @test M * y == Geometry.Cartesian12Vector(1.0, 8.5)
    @test M \ Geometry.Cartesian12Vector(1.0, 8.5) == y

    @test_throws DimensionMismatch dot(x, x)
    @test_throws DimensionMismatch M * x
    @test_throws DimensionMismatch M \ x

    @test DataLayouts.typesize(Float64, typeof(x)) == 2
end

@testset "Printing" begin
    # https://github.com/CliMA/ClimaCore.jl/issues/768
    T = Geometry.AxisTensor{
        Float64,
        2,
        Tuple{Geometry.LocalAxis{(1, 2)}, Geometry.CovariantAxis{(1, 2)}},
        SMatrix{2, 2, Float64, 4},
    }
    components = SMatrix{2, 2, Float64, 4}([4.0 0.0; 0.0 5.0])
    axes_T = (Geometry.LocalAxis{(1, 2)}(), Geometry.CovariantAxis{(1, 2)}())
    ats = T(axes_T, components)
    s = sprint(show, ats)
    s = replace(s, "StaticArraysCore." => "")
    s = replace(s, "ClimaCore.Geometry." => "")
    if !Sys.iswindows()
        @test s ==
              "AxisTensor{Float64, 2, Tuple{LocalAxis{(1, 2)}, CovariantAxis{(1, 2)}}, SMatrix{2, 2, Float64, 4}}((LocalAxis{(1, 2)}(), CovariantAxis{(1, 2)}()), [4.0 0.0; 0.0 5.0])"
    end
end

@testset "transform" begin
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)
    # @test_throws InexactError Geometry.transform(
    #     Geometry.Covariant12Axis(),
    #     Geometry.Covariant13Vector(2.0, 2.0),
    # )


    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0) * Geometry.Cartesian1Vector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)'
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)'
    # @test_throws InexactError Geometry.transform(
    #     Geometry.Covariant12Axis(),
    #     Geometry.Covariant13Vector(2.0, 2.0) * Geometry.Cartesian1Vector(1.0)',
    # )
end

@testset "project" begin
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant12Vector(2.0, 2.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant123Vector(2.0, 2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant123Vector(2.0, 2.0, 1.0),
    ) == Geometry.Covariant12Vector(2.0, 2.0)


    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 2.0),
    ) == Geometry.Covariant12Vector(2.0, 0.0)

    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0) * Geometry.Cartesian1Vector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 2.0) * Geometry.Cartesian1Vector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.Cartesian1Vector(1.0)'
end


@testset "cross product" begin
    M = @SMatrix [
        4.0 1.0
        0.5 2.0
    ]
    J = det(M)
    local_geom = Geometry.LocalGeometry(
        Geometry.XYPoint(0.0, 0.0),
        J,
        J,
        Geometry.Axis2Tensor(
            (Geometry.UVAxis(), Geometry.Covariant12Axis()),
            M,
        ),
    )

    u = Geometry.UVVector(1.0, 2.0)
    v = Geometry.WVector(3.0)
    @test u × v == -v × u == Geometry.UVVector(6.0, -3.0)
    uⁱ = Geometry.ContravariantVector(u, local_geom)
    vⁱ = Geometry.ContravariantVector(v, local_geom)
    @test Geometry.UVVector(Geometry._cross(uⁱ, vⁱ, local_geom), local_geom) ==
          Geometry.UVVector(6.0, -3.0)
end


@testset "project" begin
    M = @SMatrix [
        2.0 0.0
        0.0 1.0
    ]
    J = det(M)

    local_geom = Geometry.LocalGeometry(
        Geometry.XYPoint(0.0, 0.0),
        J,
        J,
        Geometry.Axis2Tensor(
            (Geometry.UVAxis(), Geometry.Covariant12Axis()),
            M,
        ),
    )

    @test Geometry.project(
        Geometry.Contravariant12Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant12Vector(0.25, 1.0)
    @test Geometry.project(
        Geometry.Contravariant1Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant1Vector(0.25)
    @test Geometry.project(
        Geometry.Contravariant2Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant2Vector(1.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant12Vector(1.0, 1.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 0.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant123Vector(1.0, 1.0, 1.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 1.0)


    @test Geometry.project(
        Geometry.Contravariant12Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant12Vector(0.25, 1.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant1Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant1Vector(0.25) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant2Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant2Vector(1.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant12Vector(1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 0.0) ⊗ Covariant12Vector(2.0, 8.0)
    @test Geometry.project(
        Geometry.Contravariant123Axis(),
        Covariant123Vector(1.0, 1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0),
        local_geom,
    ) == Contravariant123Vector(0.25, 1.0, 1.0) ⊗ Covariant12Vector(2.0, 8.0)
end
