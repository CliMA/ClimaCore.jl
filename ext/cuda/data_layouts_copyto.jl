DataLayouts._device_dispatch(x::CUDA.CuArray) = ToCUDA()

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

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::DataLayouts.BroadcastedUnionIJFH{S, Nij},
    ::ToCUDA,
) where {S, Nij}
    _, _, _, _, Nh = size(bc)
    if Nh > 0
        auto_launch!(
            knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (Nij, Nij),
            blocks_s = (Nh, 1),
        )
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij},
    bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij},
    ::ToCUDA,
) where {S, Nv, Nij}
    _, _, _, _, Nh = size(bc)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        auto_launch!(
            knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (Nij, Nij, Nv_per_block),
            blocks_s = (Nh, Nv_blocks),
        )
    end
    return dest
end

function Base.copyto!(
    dest::VF{S, Nv},
    bc::DataLayouts.BroadcastedUnionVF{S, Nv},
    ::ToCUDA,
) where {S, Nv}
    _, _, _, _, Nh = size(dest)
    if Nv > 0 && Nh > 0
        auto_launch!(
            knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (1, 1),
            blocks_s = (Nh, Nv),
        )
    end
    return dest
end

function Base.copyto!(
    dest::DataF{S},
    bc::DataLayouts.BroadcastedUnionDataF{S},
    ::ToCUDA,
) where {S}
    auto_launch!(
        knl_copyto!,
        (dest, bc),
        dest;
        threads_s = (1, 1),
        blocks_s = (1, 1),
    )
    return dest
end

import ClimaCore.DataLayouts: isascalar
function knl_copyto_flat!(dest::AbstractData, bc)
    @inbounds begin
        n = size(dest)
        tidx = thread_index()
        if valid_range(tidx, prod(n))
            I = kernel_indexes(tidx, n)
            dest[I] = bc[I]
        end
    end
    return nothing
end

function cuda_copyto!(dest::AbstractData, bc)
    (_, _, Nf, Nv, Nh) = DataLayouts.universal_size(dest)
    if Nv > 0 && Nh > 0 && Nf > 0
        auto_launch!(knl_copyto_flat!, (dest, bc), dest; auto = true)
    end
    return dest
end

# TODO: can we use CUDA's luanch configuration for all data layouts?
# Currently, it seems to have a slight performance degredation.
#! format: off
# Base.copyto!(dest::IJFH{S, Nij},      bc::DataLayouts.BroadcastedUnionIJFH{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IFH{S, Ni},        bc::DataLayouts.BroadcastedUnionIFH{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IJF{S, Nij},       bc::DataLayouts.BroadcastedUnionIJF{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IF{S, Ni},         bc::DataLayouts.BroadcastedUnionIF{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIFH{S, Nv, Ni},   bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni}, ::ToCUDA) where {S, Nv, Ni} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::VIJFH{S, Nv, Nij}, bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij}, ::ToCUDA) where {S, Nv, Nij} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::VF{S, Nv},         bc::DataLayouts.BroadcastedUnionVF{S, Nv}, ::ToCUDA) where {S, Nv} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::DataF{S},          bc::DataLayouts.BroadcastedUnionDataF{S}, ::ToCUDA) where {S} = cuda_copyto!(dest, bc)
#! format: on
