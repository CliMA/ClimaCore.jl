import Adapt
import CUDA

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
    v = CUDA.blockIdx().y

    I = CartesianIndex((i, j, 1, v, h))

    @inbounds dest[I] = src[I]
    return nothing
end

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Union{IJFH{S, Nij, A}, Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, _, Nh = size(bc)
    if Nh > 0
        CUDA.@cuda threads = (Nij, Nij) blocks = (Nh, 1) knl_copyto!(dest, bc)
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(bc)
    if Nv > 0 && Nh > 0
        CUDA.@cuda threads = (Nij, Nij) blocks = (Nh, Nv) knl_copyto!(dest, bc)
    end
    return dest
end
