function knl_fill!(dest, val, us, mask, cart_inds)
    tidx = linear_thread_idx()
    if linear_is_valid_index(tidx, us) && tidx ≤ length(unval(cart_inds))
        I = if mask isa NoMask
            unval(cart_inds)[tidx]
        else
            masked_universal_index(mask, cart_inds)
        end
        @inbounds dest[I] = val
    end
    return nothing
end

function knl_fill_linear!(dest, val, us)
    i = linear_thread_idx()
    if linear_is_valid_index(i, us)
        @inbounds dest[i] = val
    end
    return nothing
end

function Base.fill!(dest::AbstractData, bc, to::ToCUDA, mask = NoMask())
    (Ni, Nj, Nv, _, Nh) = DataLayouts.universal_size(dest)
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
            cart_inds = if mask isa NoMask
                cartesian_indices(us)
            else
                cartesian_indicies_mask(us, mask)
            end
            args = (dest, bc, us, mask, cart_inds)
            threads = threads_via_occupancy(knl_fill!, args)
            n_max_threads = min(threads, get_N(us))
            p = if mask isa NoMask
                linear_partition(prod(size(dest)), n_max_threads)
            else
                masked_partition(mask, n_max_threads, us)
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
