import Adapt
import CUDA

Adapt.adapt_structure(to, data::IJKFVH{S, Nij, Nk}) where {S, Nij, Nk} =
    IJKFVH{S, Nij, Nk}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IJFH{S, Nij}) where {S, Nij} =
    IJFH{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IFH{S, Ni}) where {S, Ni} =
    IFH{S, Ni}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IJF{S, Nij}) where {S, Nij} =
    IJF{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IF{S, Ni}) where {S, Ni} =
    IF{S, Ni}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::VF{S}) where {S} =
    VF{S}(Adapt.adapt(to, parent(data)))

parent_array_type(
    ::Type{A},
) where {A <: CUDA.GPUArrays.AbstractGPUArray{FT}} where {FT} = CUDA.CuArray{FT}

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
    bc::Union{IJFH{S, Nij, A}, Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, _, Nh = size(bc)
    CUDA.@cuda threads = (Nij, Nij) blocks = (Nh,) knl_copyto!(dest, bc)
    return dest
end
