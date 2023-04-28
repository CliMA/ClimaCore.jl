module Device
using ClimaComms
using CUDA

device(; disablegpu = false) =
    CUDA.has_cuda_gpu() && !disablegpu ? ClimaComms.CUDADevice() :
    ClimaComms.CPUDevice()

device_array_type(::ClimaComms.CPUDevice) = Array
device_array_type(::ClimaComms.CUDADevice) = CUDA.CuArray

device(ctx::ClimaComms.SingletonCommsContext) = ctx.device

device(::Array) = ClimaComms.CPUDevice()
device(::CUDA.CuArray) = ClimaComms.CUDADevice()

end
