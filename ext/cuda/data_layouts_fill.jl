function knl_fill_flat!(dest::AbstractData, val, us)
    @inbounds begin
        tidx = thread_index()
        if tidx ≤ get_N(us)
            n = array_size(dest)
            CIS = CartesianIndices(map(x -> Base.OneTo(x), n))
            I = DataSpecificCartesianIndex(CIS[tidx])
            @inbounds dest[I] = val
        end
    end
    return nothing
end

function cuda_fill!(dest::AbstractData, val)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        auto_launch!(knl_fill_flat!, (dest, val, us), dest; auto = true)
    end
    return dest
end

#! format: off
Base.fill!(dest::IJFH{<:Any, <:Any},         val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::IFH{<:Any, <:Any},          val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::IJF{<:Any, <:Any},          val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::IF{<:Any, <:Any},           val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::VIFH{<:Any, <:Any, <:Any},  val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::VIJFH{<:Any, <:Any, <:Any}, val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::VF{<:Any, <:Any},           val, ::ToCUDA) = cuda_fill!(dest, val)
Base.fill!(dest::DataF{<:Any},               val, ::ToCUDA) = cuda_fill!(dest, val)
#! format: on
