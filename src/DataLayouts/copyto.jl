#####
##### Dispatching and edge cases
#####

############################ new/old kernels

function Base.copyto!(
    dest::AbstractData{S},
    bc::Union{AbstractData, Base.Broadcast.Broadcasted},
) where {S}
    ncomponents(dest) > 0 || return dest
    dev = device_dispatch(dest)
    if dev isa ToCPU && has_uniform_datalayouts(bc) && !(dest isa DataF)
        # Specialize on linear indexing case:
        bc′ = Base.Broadcast.instantiate(to_non_extruded_broadcasted(bc))
        @inbounds @simd for I in 1:get_N(UniversalSize(dest))
            dest[I] = convert(S, bc′[I])
        end
    else
        Base.copyto!(dest, bc, device_dispatch(dest))
    end
    return dest
end

# function Base.copyto!(
#     dest::AbstractData,
#     bc::Union{AbstractData, Base.Broadcast.Broadcasted},
# )
#     ncomponents(dest) > 0 || return dest
#     Base.copyto!(dest, bc, device_dispatch(dest))
# end

############################

# Specialize on non-Broadcasted objects
function Base.copyto!(dest::D, src::D) where {D <: AbstractData}
    copyto!(parent(dest), parent(src))
    return dest
end

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
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
    bc::BroadcastedUnionIJFH{S, Nij, Nh},
    ::ToCPU,
) where {S, Nij, Nh}
    @inbounds for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

function Base.copyto!(
    dest::IFH{S, Ni},
    bc::BroadcastedUnionIFH{S, Ni, Nh},
    ::ToCPU,
) where {S, Ni, Nh}
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
    dest::VIFH{S, Nv, Ni, Nh},
    bc::BroadcastedUnionVIFH{S, Nv, Ni, Nh},
    ::ToCPU,
) where {S, Nv, Ni, Nh}
    # copy contiguous columns
    @inbounds for h in 1:Nh, i in 1:Ni
        col_dest = column(dest, i, h)
        col_bc = column(bc, i, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij, Nh},
    bc::BroadcastedUnionVIJFH{S, Nv, Nij, Nh},
    ::ToCPU,
) where {S, Nv, Nij, Nh}
    # copy contiguous columns
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
        col_dest = column(dest, i, j, h)
        col_bc = column(bc, i, j, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end
