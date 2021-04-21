import CUDA

function knl_copyto!(dest, src)

    #=
    nij, nh = size(dest)

    thread_idx = CUDA.threadIdx().x
    block_idx = CUDA.blockIdx().x
    block_dim = CUDA.blockDim().x

    # mapping to global idx to make insensitive
    # to number of blocks / threads per device
    global_idx = thread_idx + (block_idx - 1) * block_dim

    nx, ny = nij, nij
    i = global_idx % nx == 0 ? nx : global_idx % nx
    j = cld(global_idx, nx)
    h = ((global_idx-1) % (nx*nx)) + 1
    =#

    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y

    h = CUDA.blockIdx().x


    p_dest = slab(dest, h)
    p_src = slab(src, h)

    @inbounds p_dest[i, j] = p_src[i, j]
    return nothing
end


function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
) where {S, Nij, A <: CUDA.CuArray}
    Nh = length(dest)
    #=
    nthreads = 512
    nblocks = cld(nij*nij*nh, nthreads)
    CUDA.@cuda threads = (nthreads,) blocks = (nblocks,) knl_copyto!(dest, bc)
    =#
    CUDA.@cuda threads = (Nij, Nij) blocks = (Nh,) knl_copyto!(dest, bc)
    return dest
end
