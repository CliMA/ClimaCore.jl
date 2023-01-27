module Device
using ClimaComms
using CUDA

device(; disablegpu = false) =
    CUDA.has_cuda_gpu() && !disablegpu ? ClimaComms.CUDA() : ClimaComms.CPU()

device_array_type(::ClimaComms.CPU) = Array
device_array_type(::ClimaComms.CUDA) = CUDA.CuArray

device(ctx::ClimaComms.SingletonCommsContext) = ctx.device

device(::Array) = ClimaComms.CPU()
device(::CUDA.CuArray) = ClimaComms.CUDA()

end
