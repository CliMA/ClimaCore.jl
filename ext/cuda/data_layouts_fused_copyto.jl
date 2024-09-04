Base.@propagate_inbounds function rcopyto_at!(
    pair::Pair{<:AbstractData, <:Any},
    I,
    us,
)
    dest, bc = pair.first, pair.second
    if is_valid_index(dest, I, us)
        dest[I] = isascalar(bc) ? bc[] : bc[I]
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pair::Pair{<:DataF, <:Any}, I, us)
    dest, bc = pair.first, pair.second
    if is_valid_index(dest, I, us)
        bcI = isascalar(bc) ? bc[] : bc[I]
        dest[] = bcI
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, I, us)
    rcopyto_at!(first(pairs), I, us)
    rcopyto_at!(Base.tail(pairs), I, us)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, I, us) =
    rcopyto_at!(first(pairs), I, us)
@inline rcopyto_at!(pairs::Tuple{}, I, us) = nothing

function knl_fused_copyto!(fmbc::FusedMultiBroadcast, dest1, us)
    @inbounds begin
        I = universal_index(dest1)
        if is_valid_index(dest1, I, us)
            (; pairs) = fmbc
            rcopyto_at!(pairs, I, us)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::DataLayouts.AbstractData,
    ::ToCUDA,
)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest1)
    if Nv > 0 && Nh > 0
        us = DataLayouts.UniversalSize(dest1)
        args = (fmbc, dest1, us)
        threads = threads_via_occupancy(knl_fused_copyto!, args)
        n_max_threads = min(threads, get_N(us))
        p = partition(dest1, n_max_threads)
        auto_launch!(
            knl_fused_copyto!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
    return nothing
end
