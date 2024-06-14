#=
julia --project=test
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_column.jl"))
=#
include("benchmark_column_utils.jl")

@testset "Benchmark operators" begin
    benchmark_operators(1000, Float64)
end

nothing
