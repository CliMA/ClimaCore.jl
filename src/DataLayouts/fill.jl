function Base.fill!(dest::AbstractData, val, mask = NoMask())
    dev = device_dispatch(parent(dest))
    if !(VERSION ≥ v"1.11.0-beta") &&
       dest isa EndsWithField &&
       dev isa ClimaComms.AbstractCPUDevice &&
       mask isa NoMask
        @inbounds @simd for I in 1:get_N(UniversalSize(dest))
            dest[I] = val
        end
    else
        Base.fill!(dest, val, dev, mask)
    end
    call_post_op_callback() && post_op_callback(dest, dest, val, mask)
    dest
end

function Base.fill!(data::Union{IJFH, IJHF}, val, ::ToCPU, mask = NoMask())
    (Ni, Nj, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh, i in 1:Ni, j in 1:Nj
        idx = CartesianIndex(i, j, 1, 1, h)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end
function Base.fill!(data::Union{IFH, IHF}, val, ::ToCPU, mask = NoMask())
    (Ni, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh, i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, h)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end
function Base.fill!(data::DataF, val, ::ToCPU, mask = NoMask())
    @inbounds data[] = val
    return data
end

function Base.fill!(
    data::IJF{S, Nij},
    val,
    ::ToCPU,
    mask = NoMask(),
) where {S, Nij}
    @inbounds for j in 1:Nij, i in 1:Nij
        idx = CartesianIndex(i, j, 1, 1, 1)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end

function Base.fill!(
    data::IF{S, Ni},
    val,
    ::ToCPU,
    mask = NoMask(),
) where {S, Ni}
    @inbounds for i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, 1)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end

function Base.fill!(data::VF, val, ::ToCPU, mask::NoMask = NoMask())
    Nv = nlevels(data)
    # we don't need a mask here, since this is for a column
    @inbounds for v in 1:Nv
        data[CartesianIndex(1, 1, 1, v, 1)] = val
    end
    return data
end

function Base.fill!(data::Union{VIJFH, VIJHF}, val, ::ToCPU, mask = NoMask())
    (Ni, Nj, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, i in 1:Ni, j in 1:Nj, v in 1:Nv
        idx = CartesianIndex(i, j, 1, v, h)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end
function Base.fill!(data::Union{VIFH, VIHF}, val, ::ToCPU, mask = NoMask())
    (Ni, _, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, i in 1:Ni, v in 1:Nv
        idx = CartesianIndex(i, 1, 1, v, h)
        should_compute(mask, idx) || continue
        data[idx] = val
    end
    return data
end
