# Device module
module Device
using ClimaComms
using CUDA

Base.@deprecate device(; disablegpu = false) if disablegpu
    ClimaComms.CPUDevice()
else
    ClimaComms.device()
end false

Base.@deprecate device_array_type(ctx::ClimaComms.CPUDevice) ClimaComms.array_type(
    ctx,
) false
Base.@deprecate device_array_type(ctx::ClimaComms.CUDADevice) ClimaComms.array_type(
    ctx,
) false

Base.@deprecate device(ctx::ClimaComms.SingletonCommsContext) ClimaComms.device(
    ctx,
) false

Base.@deprecate device(ctx::Array) false
Base.@deprecate device(ctx::CUDA.CuArray) ClimaComms.device(ctx) false

end
