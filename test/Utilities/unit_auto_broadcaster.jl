using Test

using ClimaCore.Utilities: add_auto_broadcasters, nested_broadcast
using ClimaCore.Geometry: UVVector

@testset "Simple AutoBroadcasters" begin
    for itr in (1, (1, 2), (a = 1, b = (c = 2, d = 3)))
        x = @inferred add_auto_broadcasters(itr) + 0 + 0 + 0
        y = @inferred add_auto_broadcasters(itr) + 0 + itr + 0
        @test x + itr === y

        x = @inferred add_auto_broadcasters(itr) * 1 * 1 * 1
        y = @inferred add_auto_broadcasters(itr) * 1 * itr * 1
        @test x * itr === y
    end
end

@testset "AutoBroadcasters of AxisTensors" begin
    x = @inferred add_auto_broadcasters((; a = 1, b = 2)) * UVVector(1, 2)
    y = @inferred add_auto_broadcasters((; a = UVVector(1, 2), b = UVVector(2, 4)))
    @test x === y
end

@testset "Highly nested AutoBroadcasters" begin
    FT = Float64
    # Coverage instrumentation is superlinear in the compiled function sizes
    # these depths produce (511 s at 20/10/5 vs 60 s at 16/8/4 on Julia 1.10),
    # so the instrumented job runs the same testset at reduced depths, still
    # well above the nesting that trips an unlifted recursion limit.
    (d1, d2, d3) = Base.JLOptions().code_coverage == 0 ? (20, 10, 5) : (16, 8, 4)
    for T in (
        typeof(∘(ntuple(Returns(tup -> (tup,)), d1)...)(zero(FT))),
        typeof(∘(ntuple(Returns(tup -> (tup, tup)), d2)...)(zero(FT))),
        typeof(∘(ntuple(Returns(tup -> (tup, tup, tup)), d3)...)(zero(FT))),
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
        }, # similar to the prognostic state used in ClimaAtmos.jl
    )
        X = @inferred add_auto_broadcasters(T)
        @test zero(X) isa X
        for x in (
            (@inferred zero(X)),
            (@inferred FT(Integer(zero(X)))),
            (@inferred min(Integer(zero(X)), cos(zero(X)), abs(eps(X)))),
            (@inferred nested_broadcast(Returns(-), zero(X))(Int(one(X)), one(FT))),
            (@inferred nested_broadcast(Returns(zero(FT)), ntuple(Returns(one(X)), 40)...)),
        )
            @test x === zero(X)
        end
    end
end
