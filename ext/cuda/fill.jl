cartesian_index(::AbstractData, inds) = CartesianIndex(inds)

function knl_fill_flat!(dest::AbstractData, val)
    n = DataLayouts.universal_size(dest)
    inds = kernel_indexes(n)
    if valid_range(inds, n)
        I = cartesian_index(dest, inds)
        @inbounds dest[I] = val
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
