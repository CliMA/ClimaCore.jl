include("column_benchmark_utils.jl")

@testset "Benchmark operators" begin
    # (; cfield, ffield, vars_contig) = get_fields(1000, Float64, false)
    # benchmark_arrays(vars_contig)
    # benchmark_operators(cfield, ffield, vars_contig.has_h_space)
    # (; cfield, ffield, vars_contig) = get_fields(1000, Float64, true)
    # benchmark_operators(cfield, ffield, vars_contig.has_h_space)
    benchmark_operators(1000, Float64)
end

nothing
