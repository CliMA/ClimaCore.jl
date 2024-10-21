function knl_fill!(dest, val, us)
    I = universal_index(dest)
    if is_valid_index(dest, I, us)
        @inbounds dest[I] = val
    end
    return nothing
end

function knl_fill_linear!(dest, val, us)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if linear_is_valid_index(i, us)
        @inbounds dest[i] = val
    end
    return nothing
end

function Base.fill!(dest::AbstractData, bc, ::ToCUDA)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    args = (dest, bc, us)
    if Nv > 0 && Nh > 0
        if !(VERSION ≥ v"1.11.0-beta") && dest isa DataLayouts.EndsWithField
            threads = threads_via_occupancy(knl_fill_linear!, args)
            n_max_threads = min(threads, get_N(us))
            p = linear_partition(prod(size(dest)), n_max_threads)
            auto_launch!(
                knl_fill_linear!,
                args;
                threads_s = p.threads,
                blocks_s = p.blocks,
            )
        else
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
    end
    return dest
end
