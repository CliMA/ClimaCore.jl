# including this file requires CUDA to be imported and available

# The first `CUDA.@profile` session in a process defines a method, which
# advances the world age. Kernels compiled afterwards from a function that
# started running in the older world then fail with
# `AssertionError: haskey(compiled, job.source)` in GPUCompiler, so run one
# profile at top level before benchmarking.
let x = CUDA.zeros(1)
    CUDA.@profile trace = true (x .+= 1; CUDA.synchronize())
end

"""
    device_time_ns(f, ncalls = 200; warmup_s = 0.1)

Mean device time (ns) per call of `f()`, over `ncalls` calls.

The time comes from `CUDA.@profile`: it is the sum of `stop - start` over the
kernels launched by the `ncalls` calls, divided by `ncalls`, so it excludes
host-side work such as broadcast setup, kernel launch latency, and
synchronization. Memory operations are ignored, since each CUDA profiling session
records a warmup host-to-device copy that is not issued by `f`. All calls run inside
one profiling session to amortize the profiler overhead, which assumes that
every call launches the same number of kernels.

Before profiling, `f` is run for `warmup_s` seconds: without a warm-up, a GPU
that has been idle runs the first several milliseconds of kernels at lower
clocks, which inflates the times by up to about 20%.
"""
function device_time_ns(f, ncalls = 200; warmup_s = 0.1)
    f() # compile outside of the timed warm-up and the profiled region
    CUDA.synchronize()
    t_start = time()
    while time() - t_start < warmup_s
        f()
    end
    CUDA.synchronize()
    prof = CUDA.@profile trace = true begin
        for _ in 1:ncalls
            f()
        end
        CUDA.synchronize()
    end
    (; start, stop, grid) = prof.device
    kernels = findall(!ismissing, grid) # memory operations have no grid
    durations = stop[kernels] .- start[kernels] # seconds
    kernels_per_call, rem = divrem(length(durations), ncalls)
    if rem != 0 || kernels_per_call == 0
        error("Expected the same number of kernels per call, but found \
               $(length(durations)) kernels for $ncalls calls")
    end
    return sum(durations) / ncalls * 1e9
end
