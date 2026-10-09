EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
using CUDA
using Statistics

# ClimaComms defaults to CPU when CLIMACOMMS_DEVICE is unset, in which case
# every configuration below would time CPU stepping and report identical
# results.
get!(ENV, "CLIMACOMMS_DEVICE", "CUDA")
ENV["CI_PERF_SKIP_RUN"] = true
ENV["TEST_NAME"] = "sphere/baroclinic_wave_rhoe"
ENV["H_ELEM"] = "30"
ENV["Z_ELEM"] = "63"

println("=== Initializing baroclinic wave (helem=30, z_elem=63) on GPU ===")
t_init = time()
filename = joinpath(EXAMPLE_DIR, "hybrid", "driver.jl")
try
    include(filename)
catch err
    # The driver signals a completed setup-only run by throwing :exit_profile,
    # which include wraps in a LoadError.
    (err isa LoadError && err.error === :exit_profile) || rethrow()
end
println("Init done in ", round(time() - t_init, digits = 1), " s")

import ClimaTimeSteppers as CTS
import ClimaCore
CUDAExt = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)

println("Compiling kernels on initial step...")
CTS.step!(integrator)
CUDA.synchronize()
println("Compilation complete.")

function run_benchmark(label; n_warmup = 2, n_eval = 10)
    for _ in 1:n_warmup
        CTS.step!(integrator)
    end
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:n_eval
        CUDA.synchronize()
        t0 = time_ns()
        CTS.step!(integrator)
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end
    med = median(times)
    min_t = minimum(times)
    mean_t = mean(times)
    std_t = std(times)
    println(
        "Config: ",
        rpad(label, 40),
        " | Med: ",
        lpad(string(round(med, digits = 2)), 6),
        " ms | Min: ",
        lpad(string(round(min_t, digits = 2)), 6),
        " ms | Mean: ",
        lpad(string(round(mean_t, digits = 2)), 6),
        " ± ",
        round(std_t, digits = 2),
        " ms",
    )
    return (; med, min_t, mean_t)
end

results = Dict{String, Any}()

# The launch-configuration cache is flushed on every change so that no
# configuration is timed with block sizes computed for an earlier one.
function set_config!(; waves = 1, dss = 256)
    CUDAExt.MAX_WAVES[] = waves
    CUDAExt.DSS_MAX_THREADS[] = dss
    empty!(CUDAExt.LAUNCH_CONFIGURATION_CACHE)
    return nothing
end

println("\n=== 1. Baseline Configuration (MAX_WAVES=1, DSS_MAX=256) ===")
set_config!()
results["baseline"] = run_benchmark("Baseline (W=1, DSS=256)")

println("\n=== 2. Sweep MAX_WAVES (DSS=256) ===")
for w in (1, 2, 3, 4)
    set_config!(; waves = w)
    results["WAVES_$w"] = run_benchmark("MAX_WAVES=$w")
end

println("\n=== 3. Sweep DSS_MAX_THREADS (waves=1) ===")
for dss in (64, 128, 256, 512, 1024)
    set_config!(; dss)
    results["DSS_$dss"] = run_benchmark("DSS_MAX_THREADS=$dss")
end

println("\n=== 4. Combinations of Top Candidates ===")
for w in (1, 2)
    for dss in (256, 512)
        (w == 1 && dss == 256) && continue
        set_config!(; waves = w, dss)
        results["Combo_W$(w)_DSS$(dss)"] = run_benchmark("Combo (W=$w, DSS=$dss)")
    end
end

println("\n=== Summary of Top 5 Configurations ===")
sorted_results = sort(collect(results), by = x -> x.second.med)
for (i, (name, res)) in enumerate(sorted_results[1:min(5, length(sorted_results))])
    println(
        "  #$i: ",
        rpad(name, 35),
        " -> ",
        round(res.med, digits = 2),
        " ms (min: ",
        round(res.min_t, digits = 2),
        " ms)",
    )
end
