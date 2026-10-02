using Test
using CUDA

@test CUDA.functional()
@test length(collect(CUDA.devices())) >= 1

import ClimaCore.Geometry

@testset "rand(::CUDA.RNG, ::Type{<:Tensor})" begin
    rng = CUDA.default_rng()
    u = Geometry.Covariant12Vector(1.0, 2.0)
    v = Geometry.Contravariant12Vector(1.0, 2.0)
    for T in (
        Geometry.Covariant123Vector{Float64},
        Geometry.Contravariant12Vector{Float32},
        typeof(u * v'),
    )
        x = rand(rng, T)
        @test x isa T
        @test all(c -> 0 <= c < 1, parent(x))
    end
end
