using JET
using Test

using ClimaCore.Utilities: enable_auto_broadcasting
using ClimaCore.Geometry: UVVector

@static if @isdefined(var"@test_opt") # v1.7 and higher
    @testset "AutoBroadcaster optimization test" begin
        for x in [
            1.0,
            1.0f0,
            (1.0, 2.0),
            (1.0f0, 2.0f0),
            (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
            (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
        ]
            @test_opt 2 * enable_auto_broadcasting(x)
            @test_opt enable_auto_broadcasting(x) + x
            @test_opt enable_auto_broadcasting(x) / 3
        end
    end
end

@testset "AutoBroadcaster nary ops" begin
    for x in [
        1.0,
        1.0f0,
        (1.0, 2.0),
        (1.0f0, 2.0f0),
        (a = 1.0, b = (x1 = 2.0, x2 = 3.0)),
        (a = 1.0f0, b = (x1 = 2.0f0, x2 = 3.0f0)),
    ]
        @test enable_auto_broadcasting(x) * 1 * 1 * 1 === enable_auto_broadcasting(x)
        @test enable_auto_broadcasting(x) * 1 * x * 1 === enable_auto_broadcasting(x) * x
        @test enable_auto_broadcasting(x) + 0 + 0 + 0 === enable_auto_broadcasting(x)
        @test enable_auto_broadcasting(x) + 0 + x + 0 === enable_auto_broadcasting(x) + x
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
        T = Base.promote_op(enable_auto_broadcasting, nt)

        @test typeof(zero(T)) == T
        @inferred zero(T)

        @test min(zero(T), one(T)) === zero(T)
        @inferred min(zero(T), one(T))

        @test broadcast(Returns(FT(1)), zero(T), zero(T), zero(T)) === one(T)
        @inferred broadcast(Returns(FT(1)), zero(T), zero(T), zero(T))
    end
end

@testset "NamedTuples and axis tensors" begin
    FT = Float64
    nt = (; a = FT(1), b = FT(2))
    nmm = @inferred enable_auto_broadcasting(nt) * UVVector(FT(1), FT(2))
    @test nmm.a.u == 1
    @test nmm.a.v == 2
    @test nmm.b.u == 2
    @test nmm.b.v == 4
end
