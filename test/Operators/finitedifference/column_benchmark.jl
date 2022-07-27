include("column_benchmark_utils.jl")

@testset "Benchmark operators" begin
    benchmark_operators(1000, Float64)
end

nothing
