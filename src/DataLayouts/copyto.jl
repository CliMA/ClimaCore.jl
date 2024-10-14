#####
##### Dispatching and edge cases
#####

Base.copyto!(
    dest::AbstractData,
    bc::Union{AbstractData, Base.Broadcast.Broadcasted},
) = Base.copyto!(dest, bc, device_dispatch(dest))

# Specialize on non-Broadcasted objects
function Base.copyto!(dest::D, src::D) where {D <: AbstractData}
    copyto!(parent(dest), parent(src))
    return dest
end

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
# And this is valid for the CPU or GPU, since the broadcasted object
# is a scalar type.
function Base.copyto!(
    dest::AbstractData,
    bc::Base.Broadcast.Broadcasted{Style},
    ::AbstractDispatchToDevice,
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

#####
##### DataLayouts
#####

function Base.copyto!(
    dest::DataF{S},
    bc::BroadcastedUnionDataF{S, A},
    ::ToCPU,
) where {S, A}
    @inbounds dest[] = convert(S, bc[])
    return dest
end

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::BroadcastedUnionIJFH{S, Nij},
    ::ToCPU,
) where {S, Nij}
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

function Base.copyto!(
    dest::IFH{S, Ni},
    bc::BroadcastedUnionIFH{S, Ni},
    ::ToCPU,
) where {S, Ni}
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

# inline inner slab(::DataSlab2D) copy
function Base.copyto!(
    dest::IJF{S, Nij},
    bc::BroadcastedUnionIJF{S, Nij, A},
    ::ToCPU,
) where {S, Nij, A}
    @inbounds for j in 1:Nij, i in 1:Nij
        idx = CartesianIndex(i, j, 1, 1, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function Base.copyto!(
    dest::IF{S, Ni},
    bc::BroadcastedUnionIF{S, Ni, A},
    ::ToCPU,
) where {S, Ni, A}
    @inbounds for i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

# inline inner slab(::DataSlab1D) copy
function Base.copyto!(
    dest::IF{S, Ni},
    bc::Base.Broadcast.Broadcasted{IFStyle{Ni, A}},
    ::ToCPU,
) where {S, Ni, A}
    @inbounds for i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

# inline inner column(::DataColumn) copy
function Base.copyto!(
    dest::VF{S, Nv},
    bc::BroadcastedUnionVF{S, Nv, A},
    ::ToCPU,
) where {S, Nv, A}
    @inbounds for v in 1:Nv
        idx = CartesianIndex(1, 1, 1, v, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function Base.copyto!(
    dest::VIFH{S, Nv, Ni},
    bc::BroadcastedUnionVIFH{S, Nv, Ni},
    ::ToCPU,
) where {S, Nv, Ni}
    # copy contiguous columns
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, i in 1:Ni
        col_dest = column(dest, i, h)
        col_bc = column(bc, i, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij},
    bc::BroadcastedUnionVIJFH{S, Nv, Nij},
    ::ToCPU,
) where {S, Nv, Nij}
    # copy contiguous columns
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
        col_dest = column(dest, i, j, h)
        col_bc = column(bc, i, j, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end
