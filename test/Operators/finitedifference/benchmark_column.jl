#=
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_column.jl"))
=#
include("benchmark_column_utils.jl")

@testset "Benchmark operators" begin
    benchmark_operators(Float64; z_elems = 63, helem = 30, Nq = 4)
end

nothing
