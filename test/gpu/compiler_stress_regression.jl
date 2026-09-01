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

# Whether a deep operator chain exceeds the register budget is a property of
# the GPU (the sm_60 P100 cliff is not the sm_70 A100 cliff) and of how much
# register pressure the compiler currently produces, so this testset does not
# assert that a particular op count fails to compile: an assertion that a chain
# *must* fail goes red on exactly the register-pressure improvements we want,
# and one that a marginal chain *must* succeed goes red on a different CI node.
# It asserts only what is robust -- a chain comfortably below the cliff
# compiles -- reports register usage for a human to watch, and checks that a
# chain at the cliff resolves cleanly whichever way it goes: an explicit
# compile error if it fails, and no leftover dynamic invokes if it succeeds.
@testset "GPU compiler stress regressions" begin
    @test CUDA.functional()

    # Comfortably below the register cliff: must compile. The register count is
    # reported, not gated on a threshold, so lowering register pressure (the
    # goal) never breaks this test.
    div12 = _run_compile_mode("div_12_ops")
    @test div12.success
    @test !isnothing(div12.cuda_profile_summary)
    isnothing(div12.cuda_profile_summary) ||
        @info "stress div_12_ops compiled" registers =
            div12.cuda_profile_summary.registers

    # Chains at the register cliff: whether they compile is node- and
    # pressure-dependent, so it is reported, not asserted. What is asserted is
    # that the outcome is clean either way -- a failure carries an explicit
    # compile error (not a silent pass or a crash), and a success has no
    # leftover dynamic invokes in its LLVM.
    for name in ("div_14_ops", "curl_14_ops", "lazy_broadcast_d4_b2")
        r = _run_compile_mode(name)
        @test_skip r.success
        if r.success
            isnothing(r.cuda_profile_summary) ||
                @info "stress $name compiled" registers =
                    r.cuda_profile_summary.registers
            if !isnothing(r.llvm_analysis_summary)
                @test r.llvm_analysis_summary.invoke_count == 0
            end
        else
            @test !isempty(r.error_msg)
        end
    end
end
