function knl_fill_flat!(dest::AbstractData, val, us)
    # @inbounds begin
    #     tidx = thread_index()
    #     if tidx ≤ get_N(us)
    #         n = size(dest)
    #         I = kernel_indexes(tidx, n)
    #         @inbounds dest[I] = val
    #     end
    # end
    return nothing
end

function cuda_fill!(dest::AbstractData, val)
    (_, _, Nv, _, Nh) = DataLayouts.universal_size(dest)
    us = DataLayouts.UniversalSize(dest)
    if Nv > 0 && Nh > 0
        nitems = prod(DataLayouts.universal_size(dest))
        auto_launch!(knl_fill_flat!, (dest, val, us), nitems; auto = true)
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
