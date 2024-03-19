import Adapt
import CUDA

Adapt.adapt_structure(to, data::IJKFVH{S, Nij, Nk}) where {S, Nij, Nk} =
    IJKFVH{S, Nij, Nk}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::IJFH{S, Nij}) where {S, Nij} =
    IJFH{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::VIJFH{S, Nij}) where {S, Nij} =
    VIJFH{S, Nij}(Adapt.adapt(to, parent(data)))

Adapt.adapt_structure(to, data::VIFH{S, Ni, A}) where {S, Ni, A} =
    VIFH{S, Ni}(Adapt.adapt(to, parent(data)))

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

    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y

    h = CUDA.blockIdx().x
    v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z

    if v <= size(dest, 4)
        I = CartesianIndex((i, j, 1, v, h))
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function knl_fill!(dest, val)
    i = CUDA.threadIdx().x
    j = CUDA.threadIdx().y

    h = CUDA.blockIdx().x
    v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z

    if v <= size(dest, 4)
        I = CartesianIndex((i, j, 1, v, h))
        @inbounds dest[I] = val
    end
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
function Base.fill!(
    dest::IJFH{S, Nij, A},
    val,
) where {
    S,
    Nij,
    A <: Union{CUDA.CuArray, SubArray{<:Any, <:Any, <:CUDA.CuArray}},
}
    _, _, _, _, Nh = size(dest)
    if Nh > 0
        CUDA.@cuda threads = (Nij, Nij) blocks = (Nh, 1) knl_fill!(dest, val)
    end
    return dest
end



function Base.copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(bc)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        CUDA.@cuda always_inline = true threads = (Nij, Nij, Nv_per_block) blocks =
            (Nh, Nv_blocks) knl_copyto!(dest, bc)
    end
    return dest
end
function Base.fill!(
    dest::VIJFH{S, Nij, A},
    val,
) where {S, Nij, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(dest)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        CUDA.@cuda always_inline = true threads = (Nij, Nij, Nv_per_block) blocks =
            (Nh, Nv_blocks) knl_fill!(dest, val)
    end
    return dest
end


function Base.copyto!(
    dest::VF{S},
    bc::Union{VF{S, A}, Base.Broadcast.Broadcasted{VFStyle{A}}},
) where {S, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(bc)
    if Nv > 0 && Nh > 0
        CUDA.@cuda threads = (1, 1) blocks = (Nh, Nv) knl_copyto!(dest, bc)
    end
    return dest
end
function Base.fill!(dest::VF{S, A}, val) where {S, A <: CUDA.CuArray}
    _, _, _, Nv, Nh = size(dest)
    if Nv > 0 && Nh > 0
        CUDA.@cuda threads = (1, 1) blocks = (Nh, Nv) knl_fill!(dest, val)
    end
    return dest
end

function Base.copyto!(
    dest::DataF{S},
    bc::Union{DataF{S, A}, Base.Broadcast.Broadcasted{DataFStyle{A}}},
) where {S, A <: CUDA.CuArray}
    CUDA.@cuda threads = (1, 1) blocks = (1, 1) knl_copyto!(dest, bc)
    return dest
end
function Base.fill!(dest::DataF{S, A}, val) where {S, A <: CUDA.CuArray}
    CUDA.@cuda threads = (1, 1) blocks = (1, 1) knl_fill!(dest, val)
    return dest
end

Base.@propagate_inbounds function rcopyto_at!(
    pair::Pair{<:AbstractData, <:Any},
    I,
    v,
)
    dest, bc = pair.first, pair.second
    if v <= size(dest, 4)
        bcI = isascalar(bc) ? bc[] : bc[I]
        dest[I] = bcI
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, I, v)
    rcopyto_at!(first(pairs), I, v)
    rcopyto_at!(Base.tail(pairs), I, v)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, I, v) =
    rcopyto_at!(first(pairs), I, v)
@inline rcopyto_at!(pairs::Tuple{}, I, v) = nothing

function knl_fused_copyto!(fmbc::FusedMultiBroadcast)

    @inbounds begin
        i = CUDA.threadIdx().x
        j = CUDA.threadIdx().y

        h = CUDA.blockIdx().x
        v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z
        (; pairs) = fmbc
        I = CartesianIndex((i, j, 1, v, h))
        rcopyto_at!(pairs, I, v)
    end
    return nothing
end

function fused_copyto_cuda!(
    fmbc::FusedMultiBroadcast,
    dest1::VIJFH{S, Nij},
) where {S, Nij}
    _, _, _, Nv, Nh = size(dest1)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        CUDA.@cuda always_inline = true threads = (Nij, Nij, Nv_per_block) blocks =
            (Nh, Nv_blocks) knl_fused_copyto!(fmbc)
    end
    return nothing
end

adapt_f(to, f::F) where {F} = Adapt.adapt(to, f)
adapt_f(to, ::Type{F}) where {F} = (x...) -> F(x...)

function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    fmbc::FusedMultiBroadcast,
)
    FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            dest = pair.first
            bc = pair.second
            Pair(
                Adapt.adapt(to, dest),
                Base.Broadcast.Broadcasted(
                    bc.style,
                    adapt_f(to, bc.f),
                    Adapt.adapt(to, bc.args),
                    Adapt.adapt(to, bc.axes),
                ),
            )
        end,
    )
end
