DataLayouts._device_dispatch(x::CUDA.CuArray) = ToCUDA()

function knl_copyto!(dest, src, us)
    I = universal_index(dest)
    if is_valid_index(dest, I, us)
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function cuda_copyto!(dest::AbstractData, bc)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        args = (dest, bc, us)
        threads = threads_via_occupancy(knl_copyto!, args)
        n_max_threads = min(threads, get_N(us))
        p = partition(dest, n_max_threads)
        auto_launch!(
            knl_copyto!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
    return dest
end

#! format: off
Base.copyto!(dest::IJFH{S, Nij},          bc::DataLayouts.BroadcastedUnionIJFH{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IFH{S, Ni},            bc::DataLayouts.BroadcastedUnionIFH{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IJF{S, Nij},           bc::DataLayouts.BroadcastedUnionIJF{S, Nij}, ::ToCUDA) where {S, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::IF{S, Ni},             bc::DataLayouts.BroadcastedUnionIF{S, Ni}, ::ToCUDA) where {S, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIFH{S, Nv, Ni},       bc::DataLayouts.BroadcastedUnionVIFH{S, Nv, Ni}, ::ToCUDA) where {S, Nv, Ni} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VIJFH{S, Nv, Nij},     bc::DataLayouts.BroadcastedUnionVIJFH{S, Nv, Nij}, ::ToCUDA) where {S, Nv, Nij} = cuda_copyto!(dest, bc)
Base.copyto!(dest::VF{S, Nv},             bc::DataLayouts.BroadcastedUnionVF{S, Nv}, ::ToCUDA) where {S, Nv} = cuda_copyto!(dest, bc)
Base.copyto!(dest::DataF{S},              bc::DataLayouts.BroadcastedUnionDataF{S}, ::ToCUDA) where {S} = cuda_copyto!(dest, bc)
#! format: on
