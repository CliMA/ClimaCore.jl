include("column_benchmark_utils.jl")

@testset "Gradient benchmark" begin
    (; cfield, ffield, vars_contig) = get_fields(1000, Float64)
    benchmark_grad(vars_contig, cfield, ffield)
end

@testset "Benchmark cases" begin
    (; cfield, ffield, vars_contig) = get_fields(1000, Float64)
    benchmark_cases(vars_contig, cfield, ffield)
end
