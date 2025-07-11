using JET
using Test

import ClimaCore as CC
using ClimaCore.RecursiveApply
using ClimaCore.Geometry

@static if @isdefined(var"@test_opt") # v1.7 and higher
    @testset "RecursiveApply optimization test" begin
        for x in [
            1.0,
            1.0f0,
            (1.0, 2.0),
            (1.0f0, 2.0f0),
            (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
            (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
        ]
            @test_opt 2 ⊠ x
            @test_opt x ⊞ x
            @test_opt RecursiveApply.rdiv(x, 3)
        end
    end
end

@testset "RecursiveApply nary ops" begin
    for x in [
        1.0,
        1.0f0,
        (1.0, 2.0),
        (1.0f0, 2.0f0),
        (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
        (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
    ]
        FT = eltype(x[1])
        @test RecursiveApply.rmul(x, one(FT), one(FT), one(FT)) == x
        @test RecursiveApply.rmul(x, one(FT), x, one(FT)) ==
              RecursiveApply.rmul(x, x)
        @test RecursiveApply.radd(x, zero(FT), zero(FT), zero(FT)) == x
        @test RecursiveApply.radd(x, zero(FT), x, zero(FT)) ==
              RecursiveApply.rmul(x, FT(2))
    end
end

@testset "Highly nested types" begin
    FT = Float64
    nested_types = [
        FT,
        Tuple{FT, FT},
        NamedTuple{(:ϕ, :ψ), Tuple{FT, FT}},
        Tuple{
            NamedTuple{(:ϕ, :ψ), Tuple{FT, FT}},
            NamedTuple{(:ϕ, :ψ), Tuple{FT, FT}},
        },
        Tuple{FT, FT},
        NamedTuple{
            (:ρ, :uₕ, :ρe_tot, :ρq_tot, :sgs⁰, :sgsʲs),
            Tuple{
                FT,
                Tuple{FT, FT},
                FT,
                FT,
                NamedTuple{(:ρatke,), Tuple{FT}},
                Tuple{NamedTuple{(:ρa, :ρae_tot, :ρaq_tot), Tuple{FT, FT, FT}}},
            },
        },
        NamedTuple{
            (:u₃, :sgsʲs),
            Tuple{Tuple{FT}, Tuple{NamedTuple{(:u₃,), Tuple{Tuple{FT}}}}},
        },
    ]
    for nt in nested_types
        rz = RecursiveApply.rmap(RecursiveApply.rzero, nt)
        @test typeof(rz) == nt
        @inferred RecursiveApply.rmap(RecursiveApply.rzero, nt)

        rz = RecursiveApply.rmap((x, y) -> RecursiveApply.rzero(x), nt, nt)
        @test typeof(rz) == nt
        @inferred RecursiveApply.rmap((x, y) -> RecursiveApply.rzero(x), nt, nt)

        rz = RecursiveApply.rmaptype(identity, nt)
        @test rz == nt
        @inferred RecursiveApply.rmaptype(zero, nt)

        rz = RecursiveApply.rmaptype((x, y) -> identity(x), nt, nt)
        @test rz == nt
        @inferred RecursiveApply.rmaptype((x, y) -> zero(x), nt, nt)
    end
end

@testset "NamedTuples and axis tensors" begin
    FT = Float64
    nt = (; a = FT(1), b = FT(2))
    uv = Geometry.UVVector(FT(1), FT(2))
    rz = RecursiveApply.rmap(*, nt, uv)
    @test typeof(rz) == NamedTuple{(:a, :b), Tuple{UVVector{FT}, UVVector{FT}}}
    @inferred RecursiveApply.rmap(*, nt, uv)
    @test rz.a.u == 1
    @test rz.a.v == 2
    @test rz.b.u == 2
    @test rz.b.v == 4
end

@testset "NamedTuple subset functionality" begin
    # Test basic subset functionality
    X = (a=1, b=2.0, d=[1, 2, 3])
    Y = (a=10, b=3.0)
    
    result = RecursiveApply.rmap(+, X, Y)
    @test result.a == 11  # 1 + 10
    @test result.b == 5.0  # 2.0 + 3.0
    @test result.d == [1, 2, 3]  # unchanged from X
    
    # Test with nested NamedTuples
    X_nested = (a=(x=1, y=2), b=(z=3, w=4))
    Y_nested = (a=(x=10,),)
    
    result_nested = RecursiveApply.rmap(+, X_nested, Y_nested)
    @test result_nested.a.x == 11  # 1 + 10
    @test result_nested.a.y == 2  # unchanged from X
    @test result_nested.b.z == 3  # unchanged from X
    @test result_nested.b.w == 4  # unchanged from X
    
    # Test error case (Y has names not in X)
    Y_error = (a=1, e=5)  # 'e' is not in X
    @test_throws ArgumentError RecursiveApply.rmap(+, X, Y_error)
    
    # Test type stability
    @test_opt RecursiveApply.rmap(+, X, Y)
    @test_opt RecursiveApply.rmap(+, X_nested, Y_nested)

    FT = Float64
    domain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint{FT}(0),
        CC.Geometry.ZPoint{FT}(1),
        boundary_names = (:bottom, :top),
    )
    mesh = CC.Meshes.IntervalMesh(domain, nelems = 2)
    space = CC.Spaces.CenterFiniteDifferenceSpace(mesh)
    coord = CC.Fields.coordinate_field(space)

    X_nt = (; a = 1.0, b = 2.0, c = 3.0)
    X_field = map(Returns(X_nt), coord)

    Y_nt = (; a = 10.0, b = 3.0)
    Y_field = map(Returns(Y_nt), coord)

    result_field = @. RecursiveApply.rmap(+, X_field, Y_field)
    @test all(==(11.0), parent(result_field.a))
    @test all(==(5.0), parent(result_field.b))
    @test all(==(3.0), parent(result_field.c))
end
