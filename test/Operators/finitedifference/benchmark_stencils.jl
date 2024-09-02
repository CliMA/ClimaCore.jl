#=
julia --project
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_stencils.jl"))
=#
include("benchmark_stencils_utils.jl")

#! format: off
@testset "Benchmark operators" begin
    # benchmark_operators_column(Float64; z_elems = 63, helem = 30, Nq = 4, compile = true)
    benchmark_operators_column(Float64; z_elems = 63, helem = 30, Nq = 4)

    # benchmark_operators_sphere(Float64; z_elems = 63, helem = 30, Nq = 4, compile = true)
    benchmark_operators_sphere(Float64; z_elems = 63, helem = 30, Nq = 4)
end
#! format: on

nothing
