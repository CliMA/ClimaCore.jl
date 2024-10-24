function Base.fill!(dest::AbstractData, val)
    dev = device_dispatch(parent(dest))
    if !(VERSION ≥ v"1.11.0-beta") &&
       dest isa EndsWithField &&
       dev isa ClimaComms.AbstractCPUDevice
        @inbounds @simd for I in 1:get_N(UniversalSize(dest))
            dest[I] = val
        end
    else
        Base.fill!(dest, val, dev)
    end
end

function Base.fill!(data::Union{IJFH, IJHF}, val, ::ToCPU)
    (_, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh
        fill!(slab(data, h), val)
    end
    return data
end
function Base.fill!(data::Union{IFH, IHF}, val, ::ToCPU)
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

function Base.fill!(data::Union{VIJFH, VIJHF}, val, ::ToCPU)
    (Ni, Nj, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end
function Base.fill!(data::Union{VIFH, VIHF}, val, ::ToCPU)
    (Ni, _, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end
