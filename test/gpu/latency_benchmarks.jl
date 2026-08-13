using ClimaCore
using ClimaCore.CommonSpaces
import ClimaComms
using CUDA
using BenchmarkTools
import LazyBroadcast: lazy


# Kernel-launch latency from ClimaCoreCUDAExt, timed on the central cluster.
# Reported rather than asserted: the numbers are stable to ~10-20% across
# builds, which is useful for spotting regressions by eye but too loose to
# gate on. Baselines below are the reference values to compare against.
let # kernel-launch latency, reported not asserted
    # test to catch regressions and improvement to kernel launch time from ClimaCoreCUDAExt
    # after the inital compilation
    ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
    @assert !isnothing(ext) # cuda must be loaded to test this extension
    space = ExtrudedCubedSphereSpace(Float32;
        z_elem = 10,
        z_min = 0,
        z_max = 1,
        radius = 10,
        h_elem = 10,
        n_quad_points = 4,
        staggering = CellCenter(),
    )

    scalar_field_1 = fill(1.0f0, space)
    scalar_field_2 = fill(1.0f0, space)
    # basic expression
    # intentionally benchmark without a sync between each trial
    CUDA.synchronize()
    latency = median(@benchmark $scalar_field_1 .= $scalar_field_1 .+ $scalar_field_2).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 12000
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"

    # repeated args expression
    CUDA.synchronize()
    latency =
        median(
            @benchmark $scalar_field_1 .=
                $scalar_field_1 .+ $scalar_field_2 .+ $scalar_field_1 .+ $scalar_field_2
        ).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 14000
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"

    # nested lazy broadcast
    lazy_sum_1 = @. lazy(scalar_field_1 + scalar_field_2)
    lazy_sum_2 = @. lazy(lazy_sum_1 + lazy_sum_1)
    lazy_sum_3 = @. lazy(lazy_sum_2 + lazy_sum_2)
    CUDA.synchronize()
    latency = median(@benchmark $scalar_field_1 .= $lazy_sum_3).time
    # update this value if the kernel launch time changes significantly and it is expected
    baseline_latency = 18500
    percent_change_latency =
        round(Int, (latency - baseline_latency) / baseline_latency * 100)
    @info "Latency: $latency ns, Percent change from baseline: $percent_change_latency%"
end
