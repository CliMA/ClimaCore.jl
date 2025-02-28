function knl_fill!(dest, val, us, mask)
    I = if mask isa NoMask
        universal_index(dest)
    else
        masked_universal_index(mask)
    end
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

function Base.fill!(dest::AbstractData, bc, to::ToCUDA, mask = NoMask())
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        if !(VERSION ≥ v"1.11.0-beta") &&
           dest isa DataLayouts.EndsWithField &&
           mask isa NoMask
            args = (dest, bc, us)
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
            args = (dest, bc, us, mask)
            threads = threads_via_occupancy(knl_fill!, args)
            n_max_threads = min(threads, get_N(us))
            p = if mask isa NoMask
                partition(dest, n_max_threads)
            else
                masked_partition(us, n_max_threads, mask)
            end
            auto_launch!(
                knl_fill!,
                args;
                threads_s = p.threads,
                blocks_s = p.blocks,
            )
        end
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc, to)
    return dest
end
