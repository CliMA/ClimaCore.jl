import Adapt
import CUDA

_max_threads_cuda() = 512

function _configure_threadblock(nitems)
    nthreads = min(_max_threads_cuda(), nitems)
    nblocks = cld(nitems, nthreads)
    return (nthreads, nblocks)
end

Adapt.adapt_structure(to, data::IJKFVH{S, Nij, Nk}) where {S, Nij, Nk} =
    IJKFVH{S, Nij, Nk}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IJFH{S, Nij}) where {S, Nij} =
    IJFH{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::VIJFH{S, Nij}) where {S, Nij} =
    VIJFH{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IFH{S, Ni}) where {S, Ni} =
    IFH{S, Ni}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IJF{S, Nij}) where {S, Nij} =
    IJF{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IF{S, Ni}) where {S, Ni} =
    IF{S, Ni}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::VF{S}) where {S} =
    VF{S}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::DataF{S}) where {S} =
    DataF{S}(Adapt.adapt(to, parent(data)))

parent_array_type(::Type{<:CUDA.CuArray{T, N, B} where {N}}) where {T, B} =
    CUDA.CuArray{T, N, B} where {N}

# Ensure that both parent array types have the same memory buffer type.
promote_parent_array_type(
    ::Type{CUDA.CuArray{T1, N, B} where {N}},
    ::Type{CUDA.CuArray{T2, N, B} where {N}},
) where {T1, T2, B} = CUDA.CuArray{promote_type(T1, T2), N, B} where {N}

# Make `similar` accept our special `UnionAll` parent array type for CuArray.
Base.similar(
    ::Type{CUDA.CuArray{T, N′, B} where {N′}},
    dims::Dims{N},
) where {T, N, B} = similar(CUDA.CuArray{T, N, B}, dims)

# function knl_copyto!(dest, src)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for i = index:stride:length(src)
#         @inbounds dest[i] = src[i]
#     end
#     return
# end
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

    v = CUDA.blockIdx().x
    h = CUDA.blockIdx().y

    p_dest = slab(dest, v, h)
    p_src = slab(src, v, h)

    idx = CartesianIndex(i, j, 1, 1, 1)
    # dest[idx] = convert(S, bc[idx])
    # CUDA.@cuprint(i, ",", j, ",", v, ",", h, "\n")
    # @inbounds p_dest[i, j] = p_src[i, j]
    return nothing
end

function knl_copyto!(
    dest::IJFH{S, Nij},
    bc::Union{IJFH{S, Nij, A}, Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y
    v = CUDA.blockIdx().x
    h = CUDA.blockIdx().y
    p_dest = slab(dest, v, h)
    p_bc = slab(bc, v, h)
    idx = CartesianIndex(i, j, 1, v, h)
    # CUDA.@cuprint(i, ",", j, ",", v,    ",", h, "\n")
    # dest[idx] = convert(S, bc[idx])
    # @inbounds p_dest[idx] = convert(S, p_bc[idx])
    @inbounds p_dest[idx] = p_bc[idx]
    # @inbounds p_dest[idx] = convert(S, bc[idx])
    return nothing
end

#=
function Base.copyto!(
    dest::IJKFVH{S, Nij, Nk, A},
    bc::Union{IJKFVH{S, Nij, Nk, A}, Base.Broadcast.Broadcasted{IJKFVHStyle{Nij, Nk, A}}}
) where {S, Nij, Nk, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end
=#

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Union{IJFH{S, Nij, A}, Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, _, Nh = size(bc)
    CUDA.@cuda threads = (Nij, Nij) blocks = (1, Nh) knl_copyto!(dest, bc)
    return dest
end
#=
function Base.copyto!(
    dest::IFH{S, Ni, A},
    bc::Union{IFH{S, Ni, A}, Base.Broadcast.Broadcasted{IFHStyle{Ni, A}}}
) where {S, Ni, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::DataF{S, A},
    bc::Union{DataF{S, A}, Base.Broadcast.Broadcasted{DataFStyle{A}}}
) where {S, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::IJF{S, Nij, A},
    bc::Union{IJF{S, Nij, A}, Base.Broadcast.Broadcasted{IJFStyle{Nij, A}}}
) where {S, Nij, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::IF{S, Ni, A},
    bc::Union{IF{S, Ni, A}, Base.Broadcast.Broadcasted{IFStyle{Ni, A}}}
) where {S, Ni, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::VF{S, A},
    bc::Union{VF{S, A}, Base.Broadcast.Broadcasted{VFStyle{A}}}
) where {S, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end
=#

function knl_copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y
    v = CUDA.blockIdx().x
    h = CUDA.blockIdx().y
    p_dest = slab(dest, v, h)
    p_bc = slab(bc, v, h)
    idx = CartesianIndex(i, j, 1, v, h)
    # CUDA.@cuprint(i, ",", j, ",", v,    ",", h, "\n")
    # dest[idx] = convert(S, bc[idx])
    # @inbounds p_dest[idx] = convert(S, p_bc[idx])
    @inbounds p_dest[idx] = p_bc[idx]
    # @inbounds p_dest[idx] = convert(S, bc[idx])
    return nothing
end
function Base.copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(bc)
    @inbounds begin
        if Nv > 0 && Nh > 0
            CUDA.@cuda threads = (Nij, Nij) blocks = (Nv, Nh) knl_copyto!(dest, bc)
        end
    end
    return dest
end

#=
function Base.copyto!(
    dest::VIFH{S, Ni, A},
    bc::Union{VIFH{S, Ni, A}, Base.Broadcast.Broadcasted{VIFHStyle{Ni, A}}}
) where {S, Ni, A, A <: CUDA.CuArray}
    (Ni, _, _, Nv, Nh) = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::IH1JH2{S, Nij, A},
    bc::Union{IH1JH2{S, Nij, A}, Base.Broadcast.Broadcasted{IH1JH2Style{Nij, A}}}
) where {S, Nij, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end

function Base.copyto!(
    dest::IV1JH2{S, Ni, A},
    bc::Union{IV1JH2{S, Ni, A}, Base.Broadcast.Broadcasted{IV1JH2Style{Ni, A}}}
) where {S, Ni, A, A <: CUDA.CuArray}
     = size(bc)
    CUDA.@cuda threads = () blocks = () knl_copyto!(dest, bc)
end
=#


