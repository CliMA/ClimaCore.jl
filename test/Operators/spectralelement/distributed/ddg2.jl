include("ddg_setup.jl")

@testset "distributed DG operators (2 ranks)" begin
    run_ddg_tests(Float64)
    run_ddg_tests(Float32)
end
