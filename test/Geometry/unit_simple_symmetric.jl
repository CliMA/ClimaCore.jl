#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Geometry", "unit_simple_symmetric.jl"))
=#
using Test
using StaticArrays
using ClimaCore.Geometry: SimpleSymmetric
import ClimaCore.Geometry
using JET
simple_symmetric(A::Matrix) = SimpleSymmetric(SMatrix{size(A)..., eltype(A)}(A))

@testset "SimpleSymmetric" begin
    A = simple_symmetric([1 2; 3 4]) # pass in non-symmetric matrix
    @test A[1, 1] == 1
    @test A[1, 2] == 2
    @test A[2, 1] == 2 # Do not access below diagonal
    @test A[2, 2] == 4

    B₀ = @SMatrix [1 2; 2 4]
    A = SimpleSymmetric(B₀)
    C = A * B₀
    @test C isa SMatrix
    @test C == B₀ * B₀

    A = @SMatrix [1 2; 2 4]
    @test SimpleSymmetric(A) * 2 === SimpleSymmetric(A * 2)

    A = @SMatrix [1 2; 2 4]
    @test SimpleSymmetric(A) / 2 === SimpleSymmetric(A / 2)
    @test_opt SimpleSymmetric(A)
    @test Geometry.tail_params(typeof(@SMatrix Float32[1 2; 2 4])) ==
          (Float32, SMatrix{2, 2, Float32, 4}, 2, 3)
end

@testset "sizs" begin
    for N in (1, 2, 3, 5, 8, 10)
        simple_symmetric(rand(N, N)) # pass in non-symmetric matrix
    end
end
