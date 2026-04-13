using Test, JET
using ClimaCore.Geometry, ClimaCore.DataLayouts
using LinearAlgebra, StaticArrays
import ClimaCore

@testset "Tensors" begin
    x = Geometry.Covariant12Vector(1.0, 2.0)
    y = Geometry.Contravariant12Vector(1.0, 4.0)

    @test x.u₁ === 1.0
    @test x.u₂ === 2.0
    @test x.u₃ === 0.0

    f(x) = x.u₁ + x.u₂ + x.u₃
    @test_opt f(x)

    ref = Ref(zero(x))
    ref[] = Geometry.Covariant12Vector(1, 2) # Int components instead of Float64
    @test ref[] == x

    M = Geometry.Tensor(
        [1.0 0.0; 0.5 2.0],
        (Geometry.UVAxis(), Geometry.Covariant12Axis()),
    )

    @test dot(x, y) == x' * y == 9.0
    @test dot(y, x) == y' * x == 9.0

    @test x == x
    @test x != parent(x)
    @test x != Geometry.Contravariant12Vector(parent(x)...)

    @test x[1] == 1.0
    @test y[2] == 4.0
    @test M[2] == 0.5
    @test M[2, 1] == 0.5
    @test M[:, 1] == Geometry.UVVector(1.0, 0.5)
    @test M[1, :] == Geometry.Covariant12Vector(1.0, 0.0)

    @test x + zero(x) == x
    @test x' + zero(x') == x'

    @test -x + x * 2 - x / 2 == -x + 2 * x - 2 \ x == x / 2
    @test -x' + x' * 2 - x' / 2 == -x' + 2 * x' - 2 \ x' == (x / 2)'

    @test x * y' ==
          x ⊗ y ==
          Geometry.Tensor(
              parent(x) * parent(y)',
              (axes(x, 1), axes(y, 1)),
          )

    @test parent(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
    @test parent(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

    @test x ⊗ 3 == Geometry.Covariant12Vector(3.0, 6.0)
    @test x ⊗ (1, (a = 2, b = 3)) == (
        Geometry.Covariant12Vector(1.0, 2.0),
        (
            a = Geometry.Covariant12Vector(2.0, 4.0),
            b = Geometry.Covariant12Vector(3.0, 6.0),
        ),
    )


    @test parent(M * inv(M)) == @SMatrix [1.0 0.0; 0.0 1.0]
    @test parent(inv(M) * M) == @SMatrix [1.0 0.0; 0.0 1.0]

    @test M * y == Geometry.UVVector(1.0, 8.5)
    @test M \ Geometry.UVVector(1.0, 8.5) == y

    @test_throws DimensionMismatch dot(x, x)
    @test_throws DimensionMismatch M * x
    @test_throws DimensionMismatch M \ x

    @test DataLayouts.num_basetypes(Float64, typeof(x)) == 2
end

@testset "Printing" begin
    # https://github.com/CliMA/ClimaCore.jl/issues/768
    T = Geometry.Tensor{
        2,
        Float64,
        Tuple{
            Geometry.Basis{Geometry.Orthonormal, (1, 2)},
            Geometry.Basis{Geometry.Covariant, (1, 2)},
        },
        SMatrix{2, 2, Float64, 4},
    }
    components = SMatrix{2, 2, Float64, 4}([4.0 0.0; 0.0 5.0])
    bases = (
        Geometry.Basis{Geometry.Orthonormal, (1, 2)}(),
        Geometry.Basis{Geometry.Covariant, (1, 2)}(),
    )
    ats = T(components, bases)
    s = sprint(show, ats)
    s = replace(s, "StaticArraysCore." => "")
    s = replace(s, "ClimaCore.Geometry." => "")
    if !Sys.iswindows()
        @test occursin("Tensor(", s)
        @test occursin("Orthonormal", s)
        @test occursin("Covariant", s)
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

    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant1Vector(2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.transform(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
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
        Geometry.Covariant1Vector(2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 0.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.Covariant13Vector(2.0, 2.0) * Geometry.UVector(1.0)',
    ) == Geometry.Covariant12Vector(2.0, 0.0) * Geometry.UVector(1.0)'

    # Test projection over rightmost axis
    x_C12 = Geometry.Covariant12Vector(2.0, 2.0)
    x_Cart123 = Geometry.UVWVector(1.0, 1.0, 1.0)
    @test Geometry.project(x_C12 * x_Cart123', Geometry.WAxis()) ==
          x_C12 * Geometry.WVector(1.0)'
    @test Geometry.project(x_C12 * x_Cart123', Geometry.VWAxis()) ==
          x_C12 * Geometry.VWVector(1.0, 1.0)'

    # Test projection over both axes
    @test Geometry.project(
        Geometry.Covariant12Axis(),
        x_C12 * x_Cart123',
        Geometry.UVWAxis(),
    ) == x_C12 * x_Cart123'
    @test Geometry.project(
        Geometry.Covariant2Axis(),
        x_C12 * x_Cart123',
        Geometry.UWAxis(),
    ) == Geometry.Covariant2Vector(2.0) * Geometry.UWVector(1.0, 1.0)'
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
        Geometry.Tensor(M, (Geometry.UVAxis(), Geometry.Covariant12Axis())),
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
        Geometry.Tensor(M, (Geometry.UVAxis(), Geometry.Covariant12Axis())),
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
