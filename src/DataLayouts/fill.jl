
############################ new/old kernels
function Base.fill!(dest::AbstractData, val)
    ncomponents(dest) > 0 || return dest
    if device_dispatch(dest) isa ToCPU && !(dest isa DataF)
        @inbounds @simd for I in 1:get_N(UniversalSize(dest))
            dest[I] = val
        end
    else
        Base.fill!(dest, val, device_dispatch(dest))
    end
end

# function Base.fill!(dest::AbstractData, val)
#     ncomponents(dest) > 0 || return dest
#     Base.fill!(dest, val, device_dispatch(dest))
# end
############################

function Base.fill!(data::AbstractData, val, ::ToCPU)
    @inbounds for i in 1:get_N(UniversalSize(data))
        data[i] = val
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
