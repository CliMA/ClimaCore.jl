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

Base.@propagate_inbounds function rcopyto_at_linear!(
    pair::Pair{<:AbstractData, <:DataLayouts.NonExtrudedBroadcasted},
    I,
)
    (dest, bc) = pair.first, pair.second
    bcI = isascalar(bc) ? bc[] : bc[I]
    dest[I] = bcI
    return nothing
end
Base.@propagate_inbounds function rcopyto_at_linear!(
    pair::Pair{<:DataF, <:DataLayouts.NonExtrudedBroadcasted},
    I,
)
    (dest, bc) = pair.first, pair.second
    bcI = isascalar(bc) ? bc[] : bc[I]
    dest[] = bcI
    return nothing
end
Base.@propagate_inbounds function rcopyto_at_linear!(pairs::Tuple, I)
    rcopyto_at_linear!(first(pairs), I)
    rcopyto_at_linear!(Base.tail(pairs), I)
end
Base.@propagate_inbounds rcopyto_at_linear!(pairs::Tuple{<:Any}, I) =
    rcopyto_at_linear!(first(pairs), I)
@inline rcopyto_at_linear!(pairs::Tuple{}, I) = nothing

function knl_fused_copyto_linear!(fmbc::FusedMultiBroadcast, us)
    @inbounds begin
        I = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
        if linear_is_valid_index(I, us)
            (; pairs) = fmbc
            rcopyto_at_linear!(pairs, I)
        end
    end
    return nothing
end
import MultiBroadcastFusion
const MBFCUDA =
    Base.get_extension(MultiBroadcastFusion, :MultiBroadcastFusionCUDAExt)
# https://github.com/JuliaLang/julia/issues/56295
# Julia 1.11's Base.Broadcast currently requires
# multiple integer indexing, wheras Julia 1.10 did not.
# This means that we cannot reserve linear indexing to
# special-case fixes for https://github.com/JuliaLang/julia/issues/28126
# (including the GPU-variant related issue resolution efforts:
# JuliaGPU/GPUArrays.jl#454, JuliaGPU/GPUArrays.jl#464).

function fused_multibroadcast_args(fmb::FusedMultiBroadcast)
    dest = first(fmb.pairs).first
    us = DataLayouts.UniversalSize(dest)
    return (fmb, us)
end

import MultiBroadcastFusion
function fused_copyto!(
    fmb::FusedMultiBroadcast,
    dest1::DataLayouts.AbstractData,
    ::ToCUDA,
)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest1)
    (Nv > 0 && Nh > 0) || return nothing # short circuit

    if pkgversion(MultiBroadcastFusion) >= v"0.3.3"
        # Automatically split kernels by available parameter memory space:
        fmbs = MBFCUDA.partition_kernels(
            fmb,
            FusedMultiBroadcast,
            fused_multibroadcast_args,
        )
        for fmb in fmbs
            launch_fused_copyto!(fmb)
        end
    else
        launch_fused_copyto!(fmb)
    end
    return nothing
end

function launch_fused_copyto!(fmb::FusedMultiBroadcast)
    dest1 = first(fmb.pairs).first
    us = DataLayouts.UniversalSize(dest1)
    destinations = map(p -> p.first, fmb.pairs)
    bcs = map(p -> p.second, fmb.pairs)
    if all(bc -> DataLayouts.has_uniform_datalayouts(bc), bcs) &&
       all(d -> d isa DataLayouts.EndsWithField, destinations)
        pairs′ = map(fmb.pairs) do p
            bc′ = DataLayouts.to_non_extruded_broadcasted(p.second)
            Pair(p.first, Base.Broadcast.instantiate(bc′))
        end
        fmb′ = FusedMultiBroadcast(pairs′)
        args = (fmb′, us)
        threads = threads_via_occupancy(knl_fused_copyto_linear!, args)
        n_max_threads = min(threads, get_N(us))
        p = linear_partition(prod(size(dest1)), n_max_threads)
        auto_launch!(
            knl_fused_copyto_linear!,
            args;
            threads_s = p.threads,
            blocks_s = p.blocks,
        )
    else
        args = (fmb, dest1, us)
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
