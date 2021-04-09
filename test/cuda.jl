using Test
using CUDA

@test length(collect(CUDA.devices())) >= 1
