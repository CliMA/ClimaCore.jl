function knl_fill!(dest, val, us)
    I = universal_index(dest)
    if is_valid_index(dest, I, us)
        @inbounds dest[I] = val
    end
    return nothing
end

function cuda_fill!(dest::AbstractData, bc)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        args = (dest, bc, us)
        threads = threads_via_occupancy(knl_fill!, args)
        n_max_threads = min(threads, get_N(us))
        p = partition(dest, n_max_threads)
        auto_launch!(
            knl_fill!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    end
    return dest
end

Base.fill!(dest::AbstractData, val, ::ToCUDA) = cuda_fill!(dest, val)
