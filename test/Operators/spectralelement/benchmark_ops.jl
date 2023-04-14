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
args = setup_kernel_args(["--device", "CUDA"]);
device = args.device
trial = benchmark_kernel!(args, kernel_spectral_div_grad!, device; silent=true);
trial = benchmark_kernel_array!(args.arr_args, kernel_spectral_wdiv_array!, device; silent=true);
show(stdout, MIME("text/plain"), trial);
```

Notes:
```
using CUDA
CUDA.@profile kernel_spectral_div_grad!(args)
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

#####
##### Compare timings
#####

function benchmark_all(ARGS::Vector{String} = ARGS)
    kernel_args = setup_kernel_args(ARGS)
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

function get_best_times()
    # To prevent regressions, we test against best times (in nanoseconds).
    best_times = OrderedCollections.OrderedDict()
    #! format: off
    best_times[:kernel_spectral_wdiv!] = 35331.3846
    best_times[:kernel_spectral_grad!] = 22647.0877
    best_times[:kernel_spectral_grad_norm!] = 84885.4254
    best_times[:kernel_spectral_div_grad!] = 42131.9457
    best_times[:kernel_spectral_wgrad_div!] = 45716.0644
    best_times[:kernel_spectral_wcurl_curl!] = 119117.6098
    best_times[:kernel_spectral_u_cross_curl_u!] = 76965.9762
    best_times[:kernel_scalar_dss!] = 8000000 # TODO: update
    best_times[:kernel_vector_dss!] = 8000000 # TODO: update
    #! format: on
    return best_times
end

bm = benchmark_all(ARGS)
best_times = get_best_times()
test_against_best_times(bm, best_times)
