using Test
using CUDA

# Regression tests for compiler stress behavior near known failure thresholds.
# These use compile-only mode for speed while still compiling GPU kernels.

include(joinpath(@__DIR__, "..", "..", "perf", "stress_test_compiler.jl"))

function _find_stress_test(name::String)
    idx = findfirst(t -> t.name == name, ALL_TESTS)
    @test !isnothing(idx)
    return ALL_TESTS[idx]
end

function _run_compile_mode(test_name::String)
    old_slurm_job_id = get(ENV, "SLURM_JOB_ID", nothing)
    # Force local subprocess execution in tests (avoid nested `srun`).
    ENV["SLURM_JOB_ID"] = "climacore-test"
    try
        return run_test(_find_stress_test(test_name), "compile")
    finally
        if isnothing(old_slurm_job_id)
            delete!(ENV, "SLURM_JOB_ID")
        else
            ENV["SLURM_JOB_ID"] = old_slurm_job_id
        end
    end
end

@testset "GPU compiler stress regressions" begin
    @test CUDA.functional()

    # Near-threshold pass should continue to compile.
    div12 = _run_compile_mode("div_12_ops")
    @test div12.success
    @test !isnothing(div12.cuda_profile_summary)
    @test div12.cuda_profile_summary.registers >= 48

    # Known brink failures should remain explicit failures (not silent passes).
    div14 = _run_compile_mode("div_14_ops")
    @test !div14.success

    curl14 = _run_compile_mode("curl_14_ops")
    @test !curl14.success

    # Nested lazy-broadcast case should still compile in compile-only mode.
    lazy_d4_b2 = _run_compile_mode("lazy_broadcast_d4_b2")
    @test lazy_d4_b2.success
    @test !isnothing(lazy_d4_b2.llvm_analysis_summary)
    @test lazy_d4_b2.llvm_analysis_summary.invoke_count == 0
end
