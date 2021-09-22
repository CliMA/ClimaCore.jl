using Test
using ClimaCore.Geometry
using LinearAlgebra, UnPack, StaticArrays

@testset "1D X,Y,Z Points" begin

    for FT in (Float32, Float64, BigFloat)
        for pt in (
            Geometry.XPoint(one(FT)),
            Geometry.YPoint(one(FT)),
            Geometry.ZPoint(one(FT)),
        )
            @test Geometry.ncomponents(pt) == 1
            @test Geometry.components(pt) isa SVector{1, FT}
            @test Geometry.component(pt, 1) isa FT
            @test Geometry.component(pt, 1) == one(FT)
            @test_throws Exception Geometry.component(pt, 2)
            @test Geometry.coordinate(pt, 1) === pt
            @test_throws Exception Geometry.coordinate(pt, 2)
            @test pt * FT(2) == typeof(pt)(one(FT) * FT(2))
        end
    end
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
    state = (ρ = 2.0, ρu = Geometry.Cartesian12Vector(1.0, 2.0), ρθ = 0.5)

    function flux(state, g)
        @unpack ρ, ρu, ρθ = state
        u = ρu / ρ
        return (ρ = ρu, ρu = (ρu ⊗ u) + (g * ρ^2 / 2) * I, ρθ = ρθ * u)
    end
    @test flux(state, 10.0) == (
        ρ = Geometry.Cartesian12Vector(1.0, 2.0),
        ρu = Geometry.Axis2Tensor(
            (Geometry.Cartesian12Axis(), Geometry.Cartesian12Axis()),
            SMatrix{2, 2}(0.5 + 20.0, 1.0, 1.0, 2.0 + 20.0),
        ),
        ρθ = Geometry.Cartesian12Vector(0.25, 0.5),
    )
end
