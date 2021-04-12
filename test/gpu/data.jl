using Test
using CUDA
using ClimateMachineCore.DataLayouts

function knl_copy!(dst, src)
    i = threadIdx().x
    j = threadIdx().y

    h = blockIdx().x

    p_dst = pancake(dst, 1, 1, h)
    p_src = pancake(src, 1, 1, h)

    @inbounds p_dst[i, j] = p_src[i, j]
    return nothing
end

S = Tuple{Complex{Float64}, Float64}
src = IJFH{S}(CuArray(rand(4, 4, 3, 10)))
dst = IJFH{S}(CuArray(zeros(4, 4, 3, 10)))

function test_copy!(dst, src)
    @cuda threads = (4, 4) blocks = (10,) knl_copy!(dst, src)
end

test_copy!(dst, src)

@test getfield(dst, :array) == getfield(src, :array)
