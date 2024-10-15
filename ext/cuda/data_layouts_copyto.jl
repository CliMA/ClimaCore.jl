DataLayouts.device_dispatch(x::CUDA.CuArray) = ToCUDA()

function knl_copyto!(dest, src, us)
    I = universal_index(dest)
    if is_valid_index(dest, I, us)
        @inbounds dest[I] = src[I]
    end
    return nothing
end

function Base.copyto!(dest::AbstractData, bc, ::ToCUDA)
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

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
# And this is valid for the CPU or GPU, since the broadcasted object
# is a scalar type.
function Base.copyto!(
    dest::AbstractData,
    bc::Base.Broadcast.Broadcasted{Style},
    ::ToCUDA,
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, ()),
    )
    @inbounds bc0 = bc[]
    fill!(dest, bc0)
end
