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

# handles the case where we want to do multiple h per block
function knl_copyto_multi_h!(dest, src)
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y
    # l is special: we use this to do multiple slabs per block
    l = CUDA.threadIdx().z
    Nl = CUDA.blockDim().z

    h = Nl * CUDA.blockIdx().x + (l - 1)
    v = CUDA.blockIdx().y

    if h > size(dest, 5)
        return nothing
    end

    I = CartesianIndex((i, j, 1, v, h))
    @inbounds dest[I] = src[I]
    return nothing
end
# handles the case where we want to do multiple v per block
function knl_copyto_multi_v!(dest, src)
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y
    # l is special: we use this to do multiple slabs per block
    l = CUDA.threadIdx().z
    Nl = CUDA.blockDim().z

    h = CUDA.blockIdx().x
    v = Nl * CUDA.blockIdx().y + (l - 1)

    if v > size(dest, 4)
        return nothing
    end

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
        Nl = fld(CUDA.warpsize(), Nij * Nij)
        CUDA.@cuda threads = (Nij, Nij, Nl) blocks = (cld(Nh, Nl), 1) knl_copyto_multi_h!(
            dest,
            bc,
        )
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(bc)
    if Nv > 0 && Nh > 0
        Nl = fld(CUDA.warpsize(), Nij * Nij)
        CUDA.@cuda threads = (Nij, Nij, Nl) blocks = (Nh, cld(Nv, Nl)) knl_copyto_multi_v!(
            dest,
            bc,
        )
    end
    return dest
end
