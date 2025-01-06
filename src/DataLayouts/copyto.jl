#####
##### Dispatching and edge cases
#####
function Base.copyto!(
    dest::AbstractData{S},
    bc::Union{AbstractData, Base.Broadcast.Broadcasted},
) where {S}
    dev = device_dispatch(parent(dest))
    if dev isa ToCPU &&
       has_uniform_datalayouts(bc) &&
       dest isa EndsWithField &&
       !(dest isa DataF)
        # Specialize on linear indexing when possible:
        bc′ = Base.Broadcast.instantiate(to_non_extruded_broadcasted(bc))
        @inbounds @simd for I in 1:get_N(UniversalSize(dest))
            dest[I] = convert(S, bc′[I])
        end
    else
        Base.copyto!(dest, bc, device_dispatch(parent(dest)))
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

# Specialize on non-Broadcasted objects
function Base.copyto!(dest::D, src::D) where {D <: AbstractData}
    copyto!(parent(dest), parent(src))
    call_post_op_callback() && post_op_callback(dest, dest, src)
    return dest
end

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
function Base.copyto!(
    dest::AbstractData,
    bc::Base.Broadcast.Broadcasted{Style},
    to::AbstractDispatchToDevice,
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, ()),
    )
    @inbounds bc0 = bc[]
    fill!(dest, bc0)
    call_post_op_callback() && post_op_callback(dest, dest, bc, to)
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
    dest::Union{IJFH{S, Nij}, IJHF{S, Nij}},
    bc::Union{BroadcastedUnionIJFH{S, Nij}, BroadcastedUnionIJHF{S, Nij}},
    ::ToCPU,
) where {S, Nij}
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
        idx = CartesianIndex(i, j, 1, 1, h)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function Base.copyto!(
    dest::Union{IFH{S, Ni}, IHF{S, Ni}},
    bc::Union{BroadcastedUnionIFH{S, Ni}, BroadcastedUnionIHF{S, Ni}},
    ::ToCPU,
) where {S, Ni}
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, h)
        dest[idx] = convert(S, bc[idx])
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
    dest::Union{VIFH{S, Nv, Ni}, VIHF{S, Nv, Ni}},
    bc::Union{BroadcastedUnionVIFH{S, Nv, Ni}, BroadcastedUnionVIHF{S, Nv, Ni}},
    ::ToCPU,
) where {S, Nv, Ni}
    # copy contiguous columns
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, i in 1:Ni, v in 1:Nv
        idx = CartesianIndex(i, 1, 1, v, h)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function Base.copyto!(
    dest::Union{VIJFH{S, Nv, Nij}, VIJHF{S, Nv, Nij}},
    bc::Union{
        BroadcastedUnionVIJFH{S, Nv, Nij},
        BroadcastedUnionVIJHF{S, Nv, Nij},
    },
    ::ToCPU,
) where {S, Nv, Nij}
    # copy contiguous columns
    (_, _, _, _, Nh) = size(dest)
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij, v in 1:Nv
        idx = CartesianIndex(i, j, 1, v, h)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function copyto_per_field!(
    array::AbstractArray,
    bc::Union{AbstractArray, Base.Broadcast.Broadcasted},
    ::ToCPU,
)
    bc′ = to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    N = prod(size(array))
    @inbounds @simd for I in 1:N
        array[I] = bc′[I]
    end
    return array
end

# Need 2 methods here to avoid unbound arguments:
function copyto_per_field_scalar!(array::AbstractArray, bc::Real, ::ToCPU)
    bc′ = to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    N = prod(size(array))
    @inbounds @simd for I in 1:N
        array[I] = bc′[]
    end
    return array
end

function copyto_per_field_scalar!(
    array::AbstractArray,
    bc::Base.Broadcast.Broadcasted{Style},
    ::ToCPU,
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc′ = to_non_extruded_broadcasted(bc)
    # All field variables are treated separately, so
    # we can parallelize across the field index, and
    # leverage linear indexing:
    N = prod(size(array))
    @inbounds @simd for I in 1:N
        array[I] = bc′[]
    end
    return array
end
