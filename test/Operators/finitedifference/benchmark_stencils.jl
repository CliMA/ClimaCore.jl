#=
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_stencils.jl"))
=#
include("benchmark_stencils_utils.jl")

@testset "Benchmark operators" begin
    benchmark_operators(Float64; z_elems = 63, helem = 30, Nq = 4)
end

nothing
