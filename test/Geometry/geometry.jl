using Test
using ClimaCore.Geometry
using LinearAlgebra, StaticArrays

@testset "1D X,Y,Z Points" begin

    for FT in (Float32, Float64, BigFloat)
        for pt in (
            Geometry.XPoint(one(FT)),
            Geometry.YPoint(one(FT)),
            Geometry.ZPoint(one(FT)),
            Geometry.LatPoint(one(FT)),
            Geometry.LongPoint(one(FT)),
            Geometry.PPoint(one(FT)),
        )
            @test Geometry.ncomponents(pt) == 1
            @test Geometry.components(pt) isa SVector{1, FT}
            @test Geometry.component(pt, 1) isa FT
            @test Geometry.component(pt, 1) == one(FT)
            @test_throws Exception Geometry.component(pt, 2)
            @test Geometry.coordinate(pt, 1) === pt
            @test_throws Exception Geometry.coordinate(pt, 2)
            @test pt * FT(2) == typeof(pt)(one(FT) * FT(2))
            @test pt == pt
            @test pt != pt * FT(2)
            @test !(pt < pt)
            @test pt < pt * FT(2)
            @test pt <= pt
            @test pt <= pt * FT(2)
            @test Geometry.tofloat(pt) == one(FT)
        end
    end

    @test Geometry.LongPoint(1.0) == Geometry.LongPoint(1.0f0)
    @test Geometry.LongPoint(0.0) != Geometry.LongPoint(1.0f0)
    @test Geometry.LongPoint(1.0) != Geometry.LatPoint(1.0)
    @test Geometry.LongPoint(1.0) != Geometry.LatPoint(1.0f0)
    @test Geometry.LongPoint(0.0) < Geometry.LongPoint(1.0f0)
    @test Geometry.LongPoint(1.0) <= Geometry.LongPoint(1.0f0)
end

@testset "2D XY,XZ Points" begin

    for FT in (Float32, Float64, BigFloat)
        for pt in (
            Geometry.XYPoint(one(FT), zero(FT)),
            Geometry.XZPoint(one(FT), zero(FT)),
        )
            @test Geometry.ncomponents(pt) == 2
            @test Geometry.components(pt) isa SVector{2, FT}
            @test Geometry.component(pt, 1) isa FT
            @test Geometry.component(pt, 2) isa FT
            @test Geometry.component(pt, 1) == one(FT)
            @test Geometry.component(pt, 2) == zero(FT)
            @test_throws Exception Geometry.component(pt, 3)
            @test Geometry.coordinate(pt, 1) == Geometry.XPoint(one(FT))
            if pt isa Geometry.XYPoint
                @test Geometry.coordinate(pt, 2) == Geometry.YPoint(zero(FT))
            elseif pt isa Geometry.XZPoint
                @test Geometry.coordinate(pt, 2) == Geometry.ZPoint(zero(FT))
            end
            @test_throws Exception Geometry.coordinate(pt, 3)
            @test pt * FT(2) == typeof(pt)(one(FT) * FT(2), zero(FT))
        end
    end
end

@testset "3D XYZ Points" begin

    for FT in (Float32, Float64, BigFloat)
        for pt in (Geometry.XYZPoint(one(FT), zero(FT), -one(FT)),)
            @test Geometry.ncomponents(pt) == 3
            @test Geometry.components(pt) isa SVector{3, FT}
            @test Geometry.component(pt, 1) isa FT
            @test Geometry.component(pt, 2) isa FT
            @test Geometry.component(pt, 3) isa FT
            @test Geometry.component(pt, 1) == one(FT)
            @test Geometry.component(pt, 2) == zero(FT)
            @test Geometry.component(pt, 3) == -one(FT)
            @test_throws Exception Geometry.component(pt, 4)
            @test Geometry.coordinate(pt, 1) == Geometry.XPoint(one(FT))
            @test Geometry.coordinate(pt, 2) == Geometry.YPoint(zero(FT))
            @test Geometry.coordinate(pt, 3) == Geometry.ZPoint(-one(FT))
            @test_throws Exception Geometry.coordinate(pt, 4)
            @test pt * FT(2) ==
                  typeof(pt)(one(FT) * FT(2), zero(FT), -2 * one(FT))
        end
    end
end

@testset "3D LatLongP Points" begin

    for FT in (Float32, Float64, BigFloat)
        for pt in (
            Geometry.LatLongPPoint(one(FT), zero(FT), one(FT)),
        )
            @test Geometry.ncomponents(pt) == 3
            @test Geometry.components(pt) isa SVector{3, FT}
            @test Geometry.component(pt, 1) isa FT
            @test Geometry.component(pt, 2) isa FT
            @test Geometry.component(pt, 3) isa FT
            @test Geometry.component(pt, 1) == one(FT)
            @test Geometry.component(pt, 2) == zero(FT)
            @test Geometry.component(pt, 3) == one(FT)
            @test_throws Exception Geometry.component(pt, 4)
            @test pt * FT(2) == typeof(pt)(one(FT) * FT(2), zero(FT), one(FT) * FT(2))
        end
    end
end

@testset "Vectors" begin
    wᵢ = Geometry.Covariant12Vector(1.0, 2.0)
    vʲ = Geometry.Contravariant12Vector(3.0, 4.0)

    @test wᵢ[1] == 1.0
    @test wᵢ[2] == 2.0
    @test vʲ[1] == 3.0
    @test vʲ[2] == 4.0

    @test wᵢ * 3 == Geometry.Covariant12Vector(3.0, 6.0)
    @test wᵢ + wᵢ == Geometry.Covariant12Vector(2.0, 4.0)


    uᵏ = Geometry.Contravariant12Vector(1.0, 2.0)
    @test_throws Exception wᵢ * uᵏ

    T = uᵏ ⊗ uᵏ
    @test uᵏ ⊗ uᵏ isa Geometry.Contravariant2Tensor

    @test T * wᵢ == Geometry.Contravariant12Vector(5.0, 10.0)
    @test_throws Exception T * uᵏ
end

@testset "Sample flux calculation" begin
    state = (ρ = 2.0, ρu = Geometry.UVVector(1.0, 2.0), ρθ = 0.5)

    function flux(state, g)
        (; ρ, ρu, ρθ) = state
        u = ρu / ρ
        return (ρ = ρu, ρu = (ρu ⊗ u) + (g * ρ^2 / 2) * I, ρθ = ρθ * u)
    end
    @test flux(state, 10.0) == (
        ρ = Geometry.UVVector(1.0, 2.0),
        ρu = Geometry.Axis2Tensor(
            (Geometry.UVAxis(), Geometry.UVAxis()),
            SMatrix{2, 2}(0.5 + 20.0, 1.0, 1.0, 2.0 + 20.0),
        ),
        ρθ = Geometry.UVVector(0.25, 0.5),
    )
end

@testset "XYZ -> Cartesian coordinate conversions" begin

    global_geom = Geometry.CartesianGlobalGeometry()

    @test Geometry.CartesianPoint(Geometry.XYPoint(1.0, 2.0), global_geom) ==
          Geometry.Cartesian12Point(1.0, 2.0)
    @test Geometry.Cartesian123Point(Geometry.XYPoint(1.0, 2.0), global_geom) ==
          Geometry.Cartesian123Point(1.0, 2.0, 0.0)

    @test Geometry.CartesianPoint(Geometry.XPoint(1.0), global_geom) ==
          Geometry.Cartesian1Point(1.0)
    @test Geometry.Cartesian123Point(Geometry.XPoint(1.0), global_geom) ==
          Geometry.Cartesian123Point(1.0, 0.0, 0.0)

    @test Geometry.CartesianPoint(Geometry.ZPoint(3.0), global_geom) ==
          Geometry.Cartesian3Point(3.0)
    @test Geometry.Cartesian123Point(Geometry.ZPoint(3.0), global_geom) ==
          Geometry.Cartesian123Point(0.0, 0.0, 3.0)

    @test Geometry.CartesianPoint(Geometry.XZPoint(1.0, 3.0), global_geom) ==
          Geometry.Cartesian13Point(1.0, 3.0)
    @test Geometry.Cartesian123Point(Geometry.XZPoint(1.0, 3.0), global_geom) ==
          Geometry.Cartesian123Point(1.0, 0.0, 3.0)
end


@testset "LatLong -> Cartesian coordinate conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(2.0, 0.0, 0.0)
    @test Geometry.Cartesian123Point(
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(2.0, 0.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(0.0, 90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 2.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(0.0, -90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, -2.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(0.0, 180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-2.0, 0.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(0.0, -180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-2.0, 0.0, 0.0)

    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(45.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(sqrt(2.0), 0.0, sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(45.0, 90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, sqrt(2.0), sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(45.0, -90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, -sqrt(2.0), sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(45.0, 180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-sqrt(2.0), 0.0, sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(45.0, -180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-sqrt(2.0), 0.0, sqrt(2.0))

    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-45.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(sqrt(2.0), 0.0, -sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-45.0, 90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, sqrt(2.0), -sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-45.0, -90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, -sqrt(2.0), -sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-45.0, 180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-sqrt(2.0), 0.0, -sqrt(2.0))
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-45.0, -180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-sqrt(2.0), 0.0, -sqrt(2.0))

    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(90.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, 2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(90.0, 90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, 2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(90.0, -90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, 2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(90.0, 180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, 2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(90.0, -180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, 2.0)

    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-90.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, -2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-90.0, 90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, -2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-90.0, -90.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, -2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-90.0, 180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, -2.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongPoint(-90.0, -180.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 0.0, -2.0)
end


@testset "LatLongZ -> Cartesian coordinate conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.Cartesian123Point(2.0, 0.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, 0.0, 3.0),
        global_geom,
    ) == Geometry.Cartesian123Point(5.0, 0.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, 90.0, 3.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, 5.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, -90.0, 3.0),
        global_geom,
    ) == Geometry.Cartesian123Point(0.0, -5.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, 180.0, 3.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-5.0, 0.0, 0.0)
    @test Geometry.CartesianPoint(
        Geometry.LatLongZPoint(0.0, -180.0, 3.0),
        global_geom,
    ) == Geometry.Cartesian123Point(-5.0, 0.0, 0.0)
end


@testset "Cartesian -> LatLong coordinate conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(2.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongPoint(0.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, 2.0, 0.0),
        global_geom,
    ) == Geometry.LatLongPoint(0.0, 90.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongPoint(0.0, 180.0)
    # check we handle the branch cut smoothly
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, -0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongPoint(0.0, -180.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, -2.0, 0.0),
        global_geom,
    ) == Geometry.LatLongPoint(0.0, -90.0)

    # should be invariant to radius
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(2.0, 0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(45.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, 2.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(45.0, 90.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, 0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(45.0, 180.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, -0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(45.0, -180.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, -2.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(45.0, -90.0)

    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(2.0, 0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-45.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, 2.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-45.0, 90.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, 0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-45.0, 180.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-2.0, -0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-45.0, -180.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, -2.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-45.0, -90.0)

    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, 0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, -0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-0.0, 0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-0.0, -0.0, 2.0),
        global_geom,
    ) == Geometry.LatLongPoint(90.0, 0.0)

    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, 0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(0.0, -0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-0.0, 0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-90.0, 0.0)
    @test Geometry.LatLongPoint(
        Geometry.Cartesian123Point(-0.0, -0.0, -2.0),
        global_geom,
    ) == Geometry.LatLongPoint(-90.0, 0.0)
end


@testset "Cartesian -> LatLongZ coordinate conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(2.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 0.0, 0.0)
    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(5.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 0.0, 3.0)
    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(0.0, 2.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 90.0, 0.0)
    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(0.0, 5.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 90.0, 3.0)
    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(-2.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 180.0, 0.0)
    @test Geometry.LatLongZPoint(
        Geometry.Cartesian123Point(-5.0, 0.0, 0.0),
        global_geom,
    ) == Geometry.LatLongZPoint(0.0, 180.0, 3.0)
end


@testset "great circle distance" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) == 0.0
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(22.0, 32.0),
        Geometry.LatLongPoint(22.0, 32.0),
        global_geom,
    ) == 0.0

    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, 1e-20),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, -1e-20),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 1e-20),
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, -1e-20),
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, 10.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 30.0),
        Geometry.LatLongPoint(0.0, 40.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(1e-20, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(-1e-20, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 1e-20),
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, -1e-20),
        Geometry.LatLongPoint(0.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(10.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 30.0),
        Geometry.LatLongPoint(1e-20, 30.0),
        global_geom,
    ) ≈ 2 * deg2rad(1e-20) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 30.0),
        Geometry.LatLongPoint(10.0, 30.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

    # points near poles
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(89.99, 30.0),
        Geometry.LatLongPoint(90.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(90.0 - 89.99) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(89.99, 90.0),
        Geometry.LatLongPoint(89.99, -90.0),
        global_geom,
    ) ≈ 2 * 2 * deg2rad(90.0 - 89.99) rtol = 2eps()

    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(-89.99, 30.0),
        Geometry.LatLongPoint(-90.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(90.0 - 89.99) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(-89.99, 90.0),
        Geometry.LatLongPoint(-89.99, -90.0),
        global_geom,
    ) ≈ 2 * 2 * deg2rad(90.0 - 89.99) rtol = 2eps()

    # antipodal pairs
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, 180.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 0.0),
        Geometry.LatLongPoint(0.0, -180.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(0.0, 90.0),
        Geometry.LatLongPoint(0.0, -90.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(20.0, 0.0),
        Geometry.LatLongPoint(-20.0, 180.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(20.0, 0.0),
        Geometry.LatLongPoint(-20.0, -180.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(20.0, 0.0),
        Geometry.LatLongPoint(-20.0, 180.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(20.0, 90.0),
        Geometry.LatLongPoint(-20.0, -90.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(89.99, 90.0),
        Geometry.LatLongPoint(-89.99, -90.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(90.0, 0.0),
        Geometry.LatLongPoint(-90.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()
    @test Geometry.great_circle_distance(
        Geometry.LatLongPoint(-90.0, 0.0),
        Geometry.LatLongPoint(90.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(180.0) rtol = 2eps()

end


@testset "shallow great circle distance" begin
    global_geom = Geometry.ShallowSphericalGlobalGeometry(2.0)

    # test between two LatLongZPoints
    @test Geometry.great_circle_distance(
        Geometry.LatLongZPoint(0.0, 30.0, 0.0),
        Geometry.LatLongZPoint(0.0, 40.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

    @test Geometry.great_circle_distance(
        Geometry.LatLongZPoint(0.0, 30.0, 10.0),
        Geometry.LatLongZPoint(0.0, 40.0, 10.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

end

@testset "deep great circle distance" begin
    global_geom = Geometry.DeepSphericalGlobalGeometry(2.0)

    # test between two LatLongZPoints
    @test Geometry.great_circle_distance(
        Geometry.LatLongZPoint(0.0, 30.0, 0.0),
        Geometry.LatLongZPoint(0.0, 40.0, 0.0),
        global_geom,
    ) ≈ 2 * deg2rad(10.0) rtol = 2eps()

    @test Geometry.great_circle_distance(
        Geometry.LatLongZPoint(0.0, 30.0, 10.0),
        Geometry.LatLongZPoint(0.0, 40.0, 10.0),
        global_geom,
    ) ≈ (10 + 2) * deg2rad(10.0) rtol = 2eps()

end
@testset "1D XPoint Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.XPoint(one(FT))
        pt_2 = Geometry.XPoint(zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "1D YPoint Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.YPoint(one(FT))
        pt_2 = Geometry.YPoint(zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "Opposite 1D X,Y,Z Points" begin
    for FT in (Float32, Float64, BigFloat)
        pt_x_1 = Geometry.XPoint(one(FT))
        pt_x_2 = Geometry.XPoint(-one(FT))
        pt_y_1 = Geometry.YPoint(one(FT))
        pt_y_2 = Geometry.YPoint(-one(FT))
        pt_z_1 = Geometry.ZPoint(one(FT))
        pt_z_2 = Geometry.ZPoint(-one(FT))
        # Check 1D geometry points
        @test pt_x_1 ≈ -pt_x_2 rtol = 2eps()
        @test pt_y_1 ≈ -pt_y_2 rtol = 2eps()
        @test pt_z_1 ≈ -pt_z_2 rtol = 2eps()
        # Check components
        @test (pt_x_1).x ≈ -(pt_x_2).x rtol = 2eps()
        @test (pt_y_1).y ≈ -(pt_y_2).y rtol = 2eps()
        @test (pt_z_1).z ≈ -(pt_z_2).z rtol = 2eps()
    end
end

@testset "Add and subtract 1D X,Y,Z Points" begin
    for FT in (Float32, Float64, BigFloat)
        pt_x_1 = Geometry.XPoint(one(FT))
        pt_x_2 = Geometry.XPoint(FT(2))
        pt_y_1 = Geometry.YPoint(one(FT))
        pt_y_2 = Geometry.YPoint(FT(2))
        pt_z_1 = Geometry.ZPoint(one(FT))
        pt_z_2 = Geometry.ZPoint(FT(2))
        # Check 1D geometry points
        @test pt_x_1 + pt_x_2 ≈ Geometry.XPoint(FT(3)) rtol = 2eps()
        @test pt_x_1 - pt_x_2 ≈ Geometry.XPoint(FT(-1)) rtol = 2eps()
        @test pt_y_1 + pt_y_2 ≈ Geometry.YPoint(FT(3)) rtol = 2eps()
        @test pt_y_1 - pt_y_2 ≈ Geometry.YPoint(FT(-1)) rtol = 2eps()
        @test pt_z_1 + pt_z_2 ≈ Geometry.ZPoint(FT(3)) rtol = 2eps()
        @test pt_z_1 - pt_z_2 ≈ Geometry.ZPoint(FT(-1)) rtol = 2eps()
        # Check components
        @test (pt_x_1 + pt_x_2).x ≈ FT(3) rtol = 2eps()
        @test (pt_x_1 - pt_x_2).x ≈ FT(-1) rtol = 2eps()
        @test (pt_y_1 + pt_y_2).y ≈ FT(3) rtol = 2eps()
        @test (pt_y_1 - pt_y_2).y ≈ FT(-1) rtol = 2eps()
        @test (pt_z_1 + pt_z_2).z ≈ FT(3) rtol = 2eps()
        @test (pt_z_1 - pt_z_2).z ≈ FT(-1) rtol = 2eps()
    end
end

@testset "1D ZPoint Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.ZPoint(one(FT))
        pt_2 = Geometry.ZPoint(zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "2D XYPoint Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.XYPoint(one(FT), one(FT))
        pt_2 = Geometry.XYPoint(zero(FT), zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "2D XZPoint Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.XZPoint(one(FT), one(FT))
        pt_2 = Geometry.XZPoint(zero(FT), zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "3D Euclidean distance" begin
    for FT in (Float32, Float64, BigFloat)
        pt_1 = Geometry.XYZPoint(one(FT), one(FT), one(FT))
        pt_2 = Geometry.XYZPoint(zero(FT), zero(FT), zero(FT))
        @test Geometry.euclidean_distance(pt_1, pt_2) ≈
              hypot((Geometry.components(pt_1) .- Geometry.components(pt_2))...) rtol =
            2eps()
    end
end

@testset "UVW -> Cartesian spherical vector conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, 1.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 0.0),
    ) == Geometry.Cartesian123Vector(1.0, 0.0, 0.0)

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(45.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(45.0, 0.0),
    ) == Geometry.Cartesian123Vector(-sqrt(0.5), 0.0, sqrt(0.5))
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(45.0, 0.0),
    ) == Geometry.Cartesian123Vector(sqrt(0.5), 0.0, sqrt(0.5))

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-45.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-45.0, 0.0),
    ) == Geometry.Cartesian123Vector(sqrt(0.5), 0.0, sqrt(0.5))
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(-45.0, 0.0),
    ) == Geometry.Cartesian123Vector(sqrt(0.5), 0.0, -sqrt(0.5))

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 90.0),
    ) == Geometry.Cartesian123Vector(-1.0, 0.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 90.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, 1.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 90.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 180.0),
    ) == Geometry.Cartesian123Vector(0.0, -1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 180.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, 1.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(0.0, 180.0),
    ) == Geometry.Cartesian123Vector(-1.0, 0.0, 0.0)

    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, -180.0),
    ) == Geometry.Cartesian123Vector(0.0, -1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(0.0, -180.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, 1.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(0.0, -180.0),
    ) == Geometry.Cartesian123Vector(-1.0, 0.0, 0.0)

    # north pole
    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.Cartesian123Vector(-1.0, 0.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, 1.0)

    # south pole
    @test Geometry.CartesianVector(
        Geometry.UVWVector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 1.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.Cartesian123Vector(1.0, 0.0, 0.0)
    @test Geometry.CartesianVector(
        Geometry.UVWVector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.Cartesian123Vector(0.0, 0.0, -1.0)
end

@testset "Cartesian -> UVW spherical vector conversions" begin
    global_geom = Geometry.SphericalGlobalGeometry(2.0)

    # north pole
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.UVWVector(1.0, 0.0, 0.0)
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(-1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.UVWVector(0.0, 1.0, 0.0)
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(0.0, 0.0, 1.0),
        global_geom,
        Geometry.LatLongPoint(90.0, 0.0),
    ) == Geometry.UVWVector(0.0, 0.0, 1.0)

    # south pole
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(0.0, 1.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.UVWVector(1.0, 0.0, 0.0)
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(1.0, 0.0, 0.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.UVWVector(0.0, 1.0, 0.0)
    @test Geometry.LocalVector(
        Geometry.Cartesian123Vector(0.0, 0.0, -1.0),
        global_geom,
        Geometry.LatLongPoint(-90.0, 0.0),
    ) == Geometry.UVWVector(0.0, 0.0, 1.0)

end
