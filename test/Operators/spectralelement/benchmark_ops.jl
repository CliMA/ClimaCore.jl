#=
To run this script:
julia --project=test

To run all tests:
```
using Revise; using ClimaCore; include(joinpath(pkgdir(ClimaCore), "test", "Operators", "spectralelement", "benchmark_ops.jl"))
```

For interactive experimentation:
```
using Revise; using ClimaCore
include(joinpath(pkgdir(ClimaCore), "test", "Operators", "spectralelement", "benchmark_utils.jl"))
include(joinpath(pkgdir(ClimaCore), "test", "Operators", "spectralelement", "benchmark_kernels.jl"))
kernel_args = setup_kernel_args(["--float-type", "Float64"]);
device = kernel_args.device
trial = benchmark_kernel!(kernel_args, kernel_spectral_div_grad!, device; silent=true);
trial = benchmark_kernel_array!(kernel_args.arr_args, kernel_spectral_wdiv_array!, device; silent=true);
show(stdout, MIME("text/plain"), trial);
```

Notes:
```
using CUDA
CUDA.@profile kernel_spectral_div_grad!(kernel_args)
```
=#
import ClimaCore as CC
include(
    joinpath(
        pkgdir(CC),
        "test",
        "Operators",
        "spectralelement",
        "benchmark_utils.jl",
    ),
)
include(
    joinpath(
        pkgdir(CC),
        "test",
        "Operators",
        "spectralelement",
        "benchmark_kernels.jl",
    ),
)
include(
    joinpath(
        pkgdir(CC),
        "test",
        "Operators",
        "spectralelement",
        "benchmark_times.jl",
    ),
)

#####
##### Compare timings
#####

function benchmark_all(kernel_args = setup_kernel_args(ARGS))

    device = kernel_args.device
    #=
    # Run benchmarks for a single kernel with:
    trial = benchmark_kernel!(kernel_args, kernel_spectral_div_grad!, device)
    trial = benchmark_kernel_array!(
        kernel_args.arr_args,
        kernel_spectral_div_grad_array!,
        kernel_args.device,
    )
    =#

    # TODO: use similar pattern as column benchmarks and extend kernels
    #! format: off
    kernels = [
        (kernel_spectral_wdiv!, kernel_spectral_wdiv_array!),
        (kernel_spectral_grad!, kernel_spectral_grad_array!),
        (kernel_spectral_grad_norm!, kernel_spectral_grad_norm_array!),
        (kernel_spectral_div_grad!, kernel_spectral_div_grad_array!),
        (kernel_spectral_wgrad_div!, kernel_spectral_wgrad_div_array!),
        (kernel_spectral_wcurl_curl!, kernel_spectral_wcurl_curl_array!),
        (kernel_spectral_u_cross_curl_u!, kernel_spectral_u_cross_curl_u_array!),
        (kernel_scalar_dss!, kernel_scalar_dss_array!),
        (kernel_vector_dss!, kernel_vector_dss_array!),
    ]
    #! format: on
    silent = true # see BenchmarkTools.@benchmark output with `silent = false`

    bm = OrderedCollections.OrderedDict()
    for (k, ka) in kernels
        # key = (Symbol(k), Symbol(ka))
        key = Symbol(k)
        @info "Benchmarking $key..."
        trial = benchmark_kernel!(kernel_args, k, device; silent)
        trial_arr =
            benchmark_kernel_array!(kernel_args.arr_args, ka, device; silent)
        bm[key] = get_summary(trial, trial_arr)
    end

    # Tabulate benchmarks:
    tabulate_summary(bm)

    # Print results for convenient copy-paste updates:
    for key in keys(bm)
        println("    best_times[:$key] = $(bm[key].t_mean_float)")
    end
    return bm
end

kernel_args = setup_kernel_args(ARGS);
bm = benchmark_all(kernel_args);
best_times = get_best_times(kernel_args);
test_against_best_times(bm, best_times);

using JET
@testset "DSS performance" begin
    kernel_scalar_dss!(kernel_args) # compile+test works
    kernel_vector_dss!(kernel_args) # compile+test works
    kernel_field_dss!(kernel_args) # compile+test works
    kernel_ntuple_field_dss!(kernel_args) # compile+test works
    # TODO: widen these tests to the GPU.
    if kernel_args.device isa ClimaComms.CPUDevice
        # Allocation tests
        p = @allocated kernel_scalar_dss!(kernel_args)
        @test p == 0
        p = @allocated kernel_vector_dss!(kernel_args)
        @test p == 0
        p = @allocated kernel_field_dss!(kernel_args)
        @test p == 0
        p = @allocated kernel_ntuple_field_dss!(kernel_args)
        @test p == 0
        # Inference tests
        JET.@test_opt kernel_scalar_dss!(kernel_args)
        JET.@test_opt kernel_vector_dss!(kernel_args)
        JET.@test_opt kernel_field_dss!(kernel_args)
        JET.@test_opt kernel_ntuple_field_dss!(kernel_args)
    end
end
