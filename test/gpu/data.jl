using Test
using KernelAbstractions
using ClimateMachineCore.DataLayouts
using CUDAKernels

@kernel function knl_copy!(dst, src)
  h = @index(Group, Linear)
  i,j,_ = @index(Local, NTuple)

  p_dst = pancake(dst, 1, 1, h)
  p_src = pancake(src, 1, 1, h)

  p_dst[i,j] = p_src[i,j]
end

S = Tuple{Complex{Float64}, Float64}
src = IJFH{S}(CuArray(rand(4,4,3,10)))
dst = IJFH{S}(CuArray(zeros(4,4,3,10)))

function test_copy!(device, dst, src)
  kernel! = knl_copy!(device, (4,4,1))
  event = kernel!(dst, src, ndrange=(4,4,10))
  wait(device, event)
end

test_copy!(CUDADevice(), dst, src)
@test getfield(dst,:array) == getfield(src,:array)