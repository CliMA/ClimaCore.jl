DataLayouts._device_dispatch(x::CUDA.CuArray) = ToCUDA()

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

function knl_copyto_flat_specialized!(dest::AbstractData, bc, us)
    @inbounds begin
        tidx = thread_index()
        if tidx ≤ get_N(us)
            n = array_size(dest)
            CIS = CartesianIndices(map(x -> Base.OneTo(x), n))
            I = DataSpecificCartesianIndex(CIS[tidx])
            dest[I] = bc[I]
        end
    end
    return nothing
end

function cuda_copyto!(dest::AbstractData, bc)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        us = DataLayouts.UniversalSize(dest)
        if has_uniform_datalayouts(bc)
            auto_launch!(
                knl_copyto_flat_specialized!,
                (dest, bc, us),
                dest;
                auto = true,
            )
        else
            auto_launch!(knl_copyto_flat!, (dest, bc, us), dest; auto = true)
        end
    end
    return dest
end

# TODO: can we use CUDA's luanch configuration for all data layouts?
# Currently, it seems to have a slight performance degradation.
#! format: off
Base.copyto!(dest::IJFH{S, Nij},          bc::DataLayouts.BroadcastedUnionIJFH{S, Nij, Nh}, ::ToCUDA) where {S, Nij, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IFH{S, Ni, Nh},        bc::DataLayouts.BroadcastedUnionIFH{S, Ni, Nh}, ::ToCUDA) where {S, Ni, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IJF{S, Nij},           bc::DataLayouts.BroadcastedUnionIJF{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IF{S, Ni},             bc::DataLayouts.BroadcastedUnionIF{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIFH{S, Nv, Ni, Nh},   bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni, Nh}, ::ToCUDA) where {S, Nv, Ni, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIJFH{S, Nv, Nij, Nh}, bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij, Nh}, ::ToCUDA) where {S, Nv, Nij, Nh} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VF{S, Nv},             bc::DataLayouts.BroadcastedUnionVF{S, Nv}, ::ToCUDA) where {S, Nv} = cuda_copyto!(dest, bc)
Base.copyto!(dest::DataF{S},              bc::DataLayouts.BroadcastedUnionDataF{S}, ::ToCUDA) where {S} = cuda_copyto!(dest, bc)
#! format: on
