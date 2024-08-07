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
    dest::IJFH{S, Nij, Nh},
    bc::DataLayouts.BroadcastedUnionIJFH{S, Nij, Nh},
    ::ToCUDA,
) where {S, Nij, Nh}
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
    dest::VIJFH{S, Nv, Nij, Nh},
    bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij, Nh},
    ::ToCUDA,
) where {S, Nv, Nij, Nh}
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
    if Nv > 0
        auto_launch!(
            knl_copyto!,
            (dest, bc),
            dest;
            threads_s = (1, 1),
            blocks_s = (1, Nv),
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
function knl_copyto_flat!(dest::AbstractData, bc, us)
    @inbounds begin
        tidx = thread_index()
        if tidx ≤ get_N(us)
            n = size(dest)
            I = kernel_indexes(tidx, n)
            dest[I] = bc[I]
        end
    end
    return nothing
end

function cuda_copyto!(dest::AbstractData, bc)
    (_, _, Nv, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        auto_launch!(knl_copyto_flat!, (dest, bc, us), dest; auto = true)
    end
    return dest
end

# TODO: can we use CUDA's luanch configuration for all data layouts?
# Currently, it seems to have a slight performance degradation.
#! format: off
# Base.copyto!(dest::IJFH{S, Nij},          bc::DataLayouts.BroadcastedUnionIJFH{S, Nij, Nh}, ::ToCUDA) where {S, Nij, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IFH{S, Ni, Nh},        bc::DataLayouts.BroadcastedUnionIFH{S, Ni, Nh}, ::ToCUDA) where {S, Ni, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IJF{S, Nij},           bc::DataLayouts.BroadcastedUnionIJF{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IF{S, Ni},             bc::DataLayouts.BroadcastedUnionIF{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIFH{S, Nv, Ni, Nh},   bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni, Nh}, ::ToCUDA) where {S, Nv, Ni, Nh} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::VIJFH{S, Nv, Nij, Nh}, bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij, Nh}, ::ToCUDA) where {S, Nv, Nij, Nh} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::VF{S, Nv},             bc::DataLayouts.BroadcastedUnionVF{S, Nv}, ::ToCUDA) where {S, Nv} = cuda_copyto!(dest, bc)
# Base.copyto!(dest::DataF{S},              bc::DataLayouts.BroadcastedUnionDataF{S}, ::ToCUDA) where {S} = cuda_copyto!(dest, bc)
#! format: on
