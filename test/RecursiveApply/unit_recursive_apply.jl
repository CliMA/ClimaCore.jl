using JET
using Test

using ClimaCore.Utilities: nested_math_mapper
using ClimaCore.Geometry: UVVector

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
            @test_opt 2 * nested_math_mapper(x)
            @test_opt nested_math_mapper(x) + x
            @test_opt nested_math_mapper(x) / 3
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
        @test nested_math_mapper(x) * 1 * 1 * 1 === nested_math_mapper(x)
        @test nested_math_mapper(x) * 1 * x * 1 === nested_math_mapper(x) * x
        @test nested_math_mapper(x) + 0 + 0 + 0 === nested_math_mapper(x)
        @test nested_math_mapper(x) + 0 + x + 0 === nested_math_mapper(x) + x
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
        T = Base.promote_op(nested_math_mapper, nt)

        @test typeof(zero(T)) == T
        @inferred zero(T)

        @test map(identity, zero(T)) === zero(T)
        @inferred map(identity, zero(T))

        @test map(max, zero(T), one(T)) === one(T)
        @inferred map(max, zero(T), one(T))

        @test sum(zero(T)) == 0
        @inferred sum(zero(T))

        @test sum(one(T)) == sizeof(T) / sizeof(FT)
        @inferred sum(one(T))

        @test mapreduce(+, max, zero(T), one(T), one(T)) == 2
        @inferred mapreduce(+, max, zero(T), one(T), one(T))
    end
end

@testset "NamedTuples and axis tensors" begin
    FT = Float64
    nt = (; a = FT(1), b = FT(2))
    nmm = @inferred nested_math_mapper(nt) * UVVector(FT(1), FT(2))
    @test nmm.a.u == 1
    @test nmm.a.v == 2
    @test nmm.b.u == 2
    @test nmm.b.v == 4
end
