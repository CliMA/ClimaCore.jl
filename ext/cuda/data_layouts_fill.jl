function knl_fill_flat!(dest::AbstractData, val)
    @inbounds begin
        tidx = thread_index()
        n = size(dest)
        if valid_range(tidx, prod(n))
            I = kernel_indexes(tidx, n)
            @inbounds dest[I] = val
        end
    end
    return nothing
end

function cuda_fill!(dest::AbstractData, val)
    (_, _, Nf, Nv, Nh) = DataLayouts.universal_size(dest)
    if Nv > 0 && Nh > 0 && Nf > 0
        auto_launch!(knl_fill_flat!, (dest, val), dest; auto = true)
    end
    return dest
end

#! format: off
Base.fill!(dest::IJFH{<:Any, <:Any, <:CuArrayBackedTypes},         val) = cuda_fill!(dest, val)
Base.fill!(dest::IFH{<:Any, <:Any, <:CuArrayBackedTypes},          val) = cuda_fill!(dest, val)
Base.fill!(dest::IJF{<:Any, <:Any, <:CuArrayBackedTypes},          val) = cuda_fill!(dest, val)
Base.fill!(dest::IF{<:Any, <:Any, <:CuArrayBackedTypes},           val) = cuda_fill!(dest, val)
Base.fill!(dest::VIFH{<:Any, <:Any, <:Any, <:CuArrayBackedTypes},  val) = cuda_fill!(dest, val)
Base.fill!(dest::VIJFH{<:Any, <:Any, <:Any, <:CuArrayBackedTypes}, val) = cuda_fill!(dest, val)
Base.fill!(dest::VF{<:Any, <:Any, <:CuArrayBackedTypes},           val) = cuda_fill!(dest, val)
Base.fill!(dest::DataF{<:Any, <:CuArrayBackedTypes},               val) = cuda_fill!(dest, val)
#! format: on
