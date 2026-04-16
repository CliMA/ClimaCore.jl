using JET
using Test

using ClimaCore.Utilities: add_auto_broadcasters
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
            @test_opt 2 * add_auto_broadcasters(x)
            @test_opt add_auto_broadcasters(x) + x
            @test_opt add_auto_broadcasters(x) / 3
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
        @test add_auto_broadcasters(x) * 1 * 1 * 1 === add_auto_broadcasters(x)
        @test add_auto_broadcasters(x) * 1 * x * 1 === add_auto_broadcasters(x) * x
        @test add_auto_broadcasters(x) + 0 + 0 + 0 === add_auto_broadcasters(x)
        @test add_auto_broadcasters(x) + 0 + x + 0 === add_auto_broadcasters(x) + x
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
    for X in map(add_auto_broadcasters, nested_types)
        @test typeof(zero(X)) == X
        @inferred zero(X)

        @test min(zero(X), one(X)) === zero(X)
        @inferred min(zero(X), one(X))

        @test broadcast(Returns(FT(1)), zero(X), zero(X), zero(X)) === one(X)
        @inferred broadcast(Returns(FT(1)), zero(X), zero(X), zero(X))
    end
end

@testset "NamedTuples and axis tensors" begin
    FT = Float64
    nt = (; a = FT(1), b = FT(2))
    nmm = @inferred add_auto_broadcasters(nt) * UVVector(FT(1), FT(2))
    @test nmm.a.u == 1
    @test nmm.a.v == 2
    @test nmm.b.u == 2
    @test nmm.b.v == 4
end
