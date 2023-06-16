using JET
using Test

using ClimaCore.RecursiveApply

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
    end
    for nt in nested_types
        rz = RecursiveApply.rmaptype(identity, nt)
        @test rz == nt
        @inferred RecursiveApply.rmaptype(zero, nt)
    end
end
