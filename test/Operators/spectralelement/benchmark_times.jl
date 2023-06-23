function get_best_times((; device, float_type))
    # To prevent regressions, we test against best times (in nanoseconds).
    best_times = Dict()
    best_times[(:CUDADevice, :Float64)] = get_best_times_gpu_float64()
    best_times[(:CPUSingleThreaded, :Float64)] = get_best_times_cpu_float64()
    best_times[(:CUDADevice, :Float32)] = get_best_times_gpu_float32()
    best_times[(:CPUSingleThreaded, :Float32)] = get_best_times_cpu_float32()
    return best_times[(nameof(typeof(device)), nameof(float_type))]
end

function get_best_times_cpu_float64()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 87042.2667
    best_times[:kernel_spectral_grad!] = 77839.3543
    best_times[:kernel_spectral_grad_norm!] = 105859.9405
    best_times[:kernel_spectral_div_grad!] = 197246.9832
    best_times[:kernel_spectral_wgrad_div!] = 217416.5156
    best_times[:kernel_spectral_wcurl_curl!] = 234982.54
    best_times[:kernel_spectral_u_cross_curl_u!] = 114439.7526
    best_times[:kernel_scalar_dss!] = 143490.7014
    best_times[:kernel_vector_dss!] = 653263.0927861996
    return best_times
end
function get_best_times_cpu_float32()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 79922.8909
    best_times[:kernel_spectral_grad!] = 74736.2011
    best_times[:kernel_spectral_grad_norm!] = 101959.6227
    best_times[:kernel_spectral_div_grad!] = 165802.621
    best_times[:kernel_spectral_wgrad_div!] = 167301.5922
    best_times[:kernel_spectral_wcurl_curl!] = 203505.1314
    best_times[:kernel_spectral_u_cross_curl_u!] = 102773.6134
    best_times[:kernel_scalar_dss!] = 135064.0959
    best_times[:kernel_vector_dss!] = 636712.7244936951
    return best_times
end
function get_best_times_gpu_float64()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 21750.2493
    best_times[:kernel_spectral_grad!] = 21046.113
    best_times[:kernel_spectral_grad_norm!] = 53723.9003
    best_times[:kernel_spectral_div_grad!] = 23895.6504
    best_times[:kernel_spectral_wgrad_div!] = 25609.3086
    best_times[:kernel_spectral_wcurl_curl!] = 75707.1739
    best_times[:kernel_spectral_u_cross_curl_u!] = 35292.1448
    best_times[:kernel_scalar_dss!] = 40036.6694
    best_times[:kernel_vector_dss!] = 50493.3335
    return best_times
end
function get_best_times_gpu_float32()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 26606.9483
    best_times[:kernel_spectral_grad!] = 20715.8317
    best_times[:kernel_spectral_grad_norm!] = 52673.6528
    best_times[:kernel_spectral_div_grad!] = 23968.5846
    best_times[:kernel_spectral_wgrad_div!] = 25486.985
    best_times[:kernel_spectral_wcurl_curl!] = 74688.6988
    best_times[:kernel_spectral_u_cross_curl_u!] = 31431.0998
    best_times[:kernel_scalar_dss!] = 39058.9365
    best_times[:kernel_vector_dss!] = 49039.1502
    return best_times
end
