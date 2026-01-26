
Base.@propagate_inbounds function rcopyto_at_linear!(
    pair::Pair{<:AbstractData, <:Any},
    I,
)
    dest, bc = pair.first, pair.second
    bcI = isascalar(bc) ? bc[] : bc[I]
    dest[I] = bcI
    return nothing
end
Base.@propagate_inbounds function rcopyto_at_linear!(
    pair::Pair{<:DataF, <:Any},
    I,
)
    dest, bc = pair.first, pair.second
    bcI = isascalar(bc) ? bc[] : bc[I]
    dest[] = bcI
    return nothing
end
Base.@propagate_inbounds function rcopyto_at_linear!(pairs::Tuple, I)
    unrolled_foreach(Base.Fix2(rcopyto_at_linear!, I), pairs)
end

# Fused multi-broadcast entry point for DataLayouts
function Base.copyto!(
    fmbc::FusedMultiBroadcast{T},
) where {N, T <: NTuple{N, Pair{<:AbstractData, <:Any}}}
    dest1 = first(fmbc.pairs).first
    fmb_inst = FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            bc = pair.second
            bc′ = if isascalar(bc)
                Base.Broadcast.instantiate(
                    Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, ()),
                )
            else
                bc
            end
            Pair(pair.first, bc′)
        end,
    )
    # check_fused_broadcast_axes(fmbc) # we should already have checked the axes

    bcs = map(p -> p.second, fmb_inst.pairs)
    destinations = map(p -> p.first, fmb_inst.pairs)
    dest1 = first(destinations)
    us = DataLayouts.UniversalSize(dest1)
    dev = device_dispatch(parent(dest1))
    if dev isa ClimaComms.AbstractCPUDevice &&
       all(bc -> has_uniform_datalayouts(bc), bcs) &&
       all(d -> d isa EndsWithField, destinations) &&
       !(VERSION ≥ v"1.11.0-beta")
        pairs′ = map(fmb_inst.pairs) do p
            bc′ = to_non_extruded_broadcasted(p.second)
            Pair(p.first, bc′)
        end
        fmbc′ = FusedMultiBroadcast(pairs′)
        @inbounds for I in 1:get_N(us)
            rcopyto_at_linear!(fmbc′.pairs, I)
        end
    else
        fused_copyto!(fmb_inst, dest1, dev)
    end
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::Union{VIJFH{S1, Nv1, Nij}, VIJHF{S1, Nv1, Nij}},
    ::ToCPU,
) where {S1, Nv1, Nij}
    for (dest, bc) in fmbc.pairs
        (_, _, _, _, Nh) = size(dest1)
        # Base.copyto!(dest, bc) # we can just fall back like this
        @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij, v in 1:Nv1
            I = CartesianIndex(i, j, 1, v, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::Union{IJFH{S, Nij}, IJHF{S, Nij}},
    ::ToCPU,
) where {S, Nij}
    # copy contiguous columns
    _, _, _, Nv, _ = size(dest1)
    for (dest, bc) in fmbc.pairs
        (_, _, _, _, Nh) = size(dest1)
        @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
            I = CartesianIndex(i, j, 1, 1, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::Union{VIFH{S, Nv1, Ni}, VIHF{S, Nv1, Ni}},
    ::ToCPU,
) where {S, Nv1, Ni}
    # copy contiguous columns
    for (dest, bc) in fmbc.pairs
        (_, _, _, _, Nh) = size(dest1)
        @inbounds for h in 1:Nh, i in 1:Ni, v in 1:Nv1
            I = CartesianIndex(i, 1, 1, v, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VF{S1, Nv1},
    ::ToCPU,
) where {S1, Nv1}
    for (dest, bc) in fmbc.pairs
        @inbounds for v in 1:Nv1
            I = CartesianIndex(1, 1, 1, v, 1)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest::DataF{S},
    ::ToCPU,
) where {S}
    for (dest, bc) in fmbc.pairs
        @inbounds dest[] = convert(S, bc[])
    end
    return dest
end

# we've already diagonalized dest, so we only need to make
# sure that all the broadcast axes are compatible.
# Logic here is similar to Base.Broadcast.instantiate
@inline function _check_fused_broadcast_axes(bc1, bc2)
    axes = Base.Broadcast.combine_axes(bc1.args..., bc2.args...)
    if !(axes isa Nothing)
        Base.Broadcast.check_broadcast_axes(axes, bc1.args...)
        Base.Broadcast.check_broadcast_axes(axes, bc2.args...)
    end
end

@inline check_fused_broadcast_axes(fmbc::FusedMultiBroadcast) =
    check_fused_broadcast_axes(
        map(x -> x.second, fmbc.pairs),
        first(fmbc.pairs).second,
    )
@inline check_fused_broadcast_axes(bcs::Tuple{<:Any}, bc1) =
    _check_fused_broadcast_axes(first(bcs), bc1)
@inline check_fused_broadcast_axes(bcs::Tuple{}, bc1) = nothing
@inline function check_fused_broadcast_axes(bcs::Tuple, bc1)
    _check_fused_broadcast_axes(first(bcs), bc1)
    check_fused_broadcast_axes(Base.tail(bcs), bc1)
end
