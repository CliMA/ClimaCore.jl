include("column_benchmark_utils.jl")

@testset "Benchmark cases" begin
    (; cfield, ffield, vars_contig) = get_fields(1000, Float64)
    benchmark_cases(vars_contig, cfield, ffield)
end

nothing
