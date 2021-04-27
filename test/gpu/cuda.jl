using Test
using CUDA

@test CUDA.functional()
@test length(collect(CUDA.devices())) >= 1
