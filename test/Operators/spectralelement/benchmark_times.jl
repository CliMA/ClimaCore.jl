function get_best_times((; device, float_type))
    # To prevent regressions, we test against best times (in nanoseconds).
    best_times = Dict()
    best_times[(:CUDA, :Float64)] = get_best_times_gpu_float64()
    best_times[(:CPU, :Float64)] = get_best_times_cpu_float64()
    best_times[(:CUDA, :Float32)] = get_best_times_gpu_float32()
    best_times[(:CPU, :Float32)] = get_best_times_cpu_float32()
    return best_times[(nameof(typeof(device)), nameof(float_type))]
end

function get_best_times_cpu_float64()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 21934.2528
    best_times[:kernel_spectral_grad!] = 20593.3485
    best_times[:kernel_spectral_grad_norm!] = 50244.6862
    best_times[:kernel_spectral_div_grad!] = 23443.0758
    best_times[:kernel_spectral_wgrad_div!] = 25229.8624
    best_times[:kernel_spectral_wcurl_curl!] = 74930.4376
    best_times[:kernel_spectral_u_cross_curl_u!] = 33769.296
    best_times[:kernel_scalar_dss!] = 39678.2349
    best_times[:kernel_vector_dss!] = 49917.9823
    return best_times
end
function get_best_times_cpu_float32()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 21934.2528
    best_times[:kernel_spectral_grad!] = 20593.3485
    best_times[:kernel_spectral_grad_norm!] = 50244.6862
    best_times[:kernel_spectral_div_grad!] = 23443.0758
    best_times[:kernel_spectral_wgrad_div!] = 25229.8624
    best_times[:kernel_spectral_wcurl_curl!] = 74930.4376
    best_times[:kernel_spectral_u_cross_curl_u!] = 33769.296
    best_times[:kernel_scalar_dss!] = 39678.2349
    best_times[:kernel_vector_dss!] = 49917.9823
    return best_times
end
function get_best_times_gpu_float64()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 21934.2528
    best_times[:kernel_spectral_grad!] = 20593.3485
    best_times[:kernel_spectral_grad_norm!] = 50244.6862
    best_times[:kernel_spectral_div_grad!] = 23443.0758
    best_times[:kernel_spectral_wgrad_div!] = 25229.8624
    best_times[:kernel_spectral_wcurl_curl!] = 74930.4376
    best_times[:kernel_spectral_u_cross_curl_u!] = 33769.296
    best_times[:kernel_scalar_dss!] = 39678.2349
    best_times[:kernel_vector_dss!] = 49917.9823
    return best_times
end
function get_best_times_gpu_float32()
    best_times = OrderedCollections.OrderedDict()
    best_times[:kernel_spectral_wdiv!] = 21934.2528
    best_times[:kernel_spectral_grad!] = 20593.3485
    best_times[:kernel_spectral_grad_norm!] = 50244.6862
    best_times[:kernel_spectral_div_grad!] = 23443.0758
    best_times[:kernel_spectral_wgrad_div!] = 25229.8624
    best_times[:kernel_spectral_wcurl_curl!] = 74930.4376
    best_times[:kernel_spectral_u_cross_curl_u!] = 33769.296
    best_times[:kernel_scalar_dss!] = 39678.2349
    best_times[:kernel_vector_dss!] = 49917.9823
    return best_times
end
