Base.@propagate_inbounds function rcopyto_at!(
    pair::Pair{<:AbstractData, <:Any},
    I,
    v,
)
    dest, bc = pair.first, pair.second
    if 1 ≤ v <= size(dest, 4)
        dest[I] = isascalar(bc) ? bc[] : bc[I]
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pair::Pair{<:DataF, <:Any}, I, v)
    dest, bc = pair.first, pair.second
    if 1 ≤ v <= size(dest, 4)
        bcI = isascalar(bc) ? bc[] : bc[I]
        dest[] = bcI
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, I, v)
    rcopyto_at!(first(pairs), I, v)
    rcopyto_at!(Base.tail(pairs), I, v)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, I, v) =
    rcopyto_at!(first(pairs), I, v)
@inline rcopyto_at!(pairs::Tuple{}, I, v) = nothing

function knl_fused_copyto!(fmbc::FusedMultiBroadcast)

    @inbounds begin
        i = CUDA.threadIdx().x
        j = CUDA.threadIdx().y

        h = CUDA.blockIdx().x
        v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z
        (; pairs) = fmbc
        I = CartesianIndex((i, j, 1, v, h))
        rcopyto_at!(pairs, I, v)
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VIJFH{S, Nv, Nij, Nh},
    ::ToCUDA,
) where {S, Nv, Nij, Nh}
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (Nij, Nij, Nv_per_block),
            blocks_s = (Nh, Nv_blocks),
        )
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::IJFH{S, Nij},
    ::ToCUDA,
) where {S, Nij}
    _, _, _, _, Nh = size(dest1)
    if Nh > 0
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (Nij, Nij),
            blocks_s = (Nh, 1),
        )
    end
    return nothing
end
function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VF{S, Nv},
    ::ToCUDA,
) where {S, Nv}
    _, _, _, _, Nh = size(dest1)
    if Nv > 0 && Nh > 0
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (1, 1),
            blocks_s = (Nh, Nv),
        )
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::DataF{S},
    ::ToCUDA,
) where {S}
    auto_launch!(
        knl_fused_copyto!,
        (fmbc,),
        dest1;
        threads_s = (1, 1),
        blocks_s = (1, 1),
    )
    return nothing
end
