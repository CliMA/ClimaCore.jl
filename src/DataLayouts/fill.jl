function Base.fill!(dest::AbstractData, val, ::ToCPU)
    @inbounds @simd for I in 1:get_N(UniversalSize(dest))
        dest[I] = val
    end
    return dest
end

function Base.fill!(dest::DataF, val, ::ToCPU)
    @inbounds dest[] = val
    return dest
end

Base.fill!(dest::AbstractData, val) =
    Base.fill!(dest, val, device_dispatch(dest))
