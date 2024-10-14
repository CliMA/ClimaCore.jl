function Base.fill!(data::IJFH, val, ::ToCPU)
    (_, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh
        fill!(slab(data, h), val)
    end
    return data
end

function Base.fill!(data::IFH, val, ::ToCPU)
    (_, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh
        fill!(slab(data, h), val)
    end
    return data
end

function Base.fill!(data::DataF, val, ::ToCPU)
    @inbounds data[] = val
    return data
end

function Base.fill!(data::IJF{S, Nij}, val, ::ToCPU) where {S, Nij}
    @inbounds for j in 1:Nij, i in 1:Nij
        data[CartesianIndex(i, j, 1, 1, 1)] = val
    end
    return data
end

function Base.fill!(data::IF{S, Ni}, val, ::ToCPU) where {S, Ni}
    @inbounds for i in 1:Ni
        data[CartesianIndex(i, 1, 1, 1, 1)] = val
    end
    return data
end

function Base.fill!(data::VF, val, ::ToCPU)
    Nv = nlevels(data)
    @inbounds for v in 1:Nv
        data[CartesianIndex(1, 1, 1, v, 1)] = val
    end
    return data
end

function Base.fill!(data::VIJFH, val, ::ToCPU)
    (Ni, Nj, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end

function Base.fill!(data::VIFH, val, ::ToCPU)
    (Ni, _, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end

Base.fill!(dest::AbstractData, val) =
    Base.fill!(dest, val, device_dispatch(dest))
