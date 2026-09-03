using Test
import StaticArrays: SVector
import ClimaCore.Utilities: static_select

@testset "static_select" begin
    v = SVector(10.0, 20.0, 30.0)
    t = (1, 2, 3, 4)

    @test static_select(v, 1) === 10.0
    @test static_select(v, 2) === 20.0
    @test static_select(v, 3) === 30.0
    for i in 1:4
        @test static_select(t, i) === t[i]
    end
    # single-element collections and Int32 indices
    @test static_select((7.5f0,), 1) === 7.5f0
    @test static_select(v, Int32(3)) === 30.0
    # out-of-range indices fall back to the first element (documented)
    @test static_select(v, 0) === 10.0
    @test static_select(v, 4) === 10.0
    # matches getindex for every in-range run-time index, and infers concretely
    idx = collect(1:3)
    @test all(static_select(v, i) == v[i] for i in idx)
    @test @inferred(static_select(v, idx[2])) === 20.0
    @test @inferred(static_select(t, idx[3])) === 3
    # ForwardDiff-style element types pass through unchanged
    @test static_select((1 + 2im, 3 + 4im), 2) === 3 + 4im
end
