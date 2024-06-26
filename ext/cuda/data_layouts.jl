
import ClimaCore.DataLayouts: AbstractData
import ClimaCore.DataLayouts: FusedMultiBroadcast
import ClimaCore.DataLayouts: IJKFVH, IJFH, VIJFH, VIFH, IFH, IJF, IF, VF, DataF
import ClimaCore.DataLayouts: IJFHStyle, VIJFHStyle, VFStyle, DataFStyle
import ClimaCore.DataLayouts: promote_parent_array_type
import ClimaCore.DataLayouts: parent_array_type
import ClimaCore.DataLayouts: isascalar
import ClimaCore.DataLayouts: fused_copyto!
import Adapt
import CUDA

parent_array_type(::Type{<:CUDA.CuArray{T, N, B} where {N}}) where {T, B} =
    CUDA.CuArray{T, N, B} where {N}

# Ensure that both parent array types have the same memory buffer type.
promote_parent_array_type(
    ::Type{CUDA.CuArray{T1, N, B} where {N}},
    ::Type{CUDA.CuArray{T2, N, B} where {N}},
) where {T1, T2, B} = CUDA.CuArray{promote_type(T1, T2), N, B} where {N}

# Make `similar` accept our special `UnionAll` parent array type for CuArray.
Base.similar(
    ::Type{CUDA.CuArray{T, N′, B} where {N′}},
    dims::Dims{N},
) where {T, N, B} = similar(CUDA.CuArray{T, N, B}, dims)

include("data_layouts_fill.jl")
include("data_layouts_copyto.jl")
include("data_layouts_mapreduce.jl")

Base.@propagate_inbounds function rcopyto_at!(
    pair::Pair{<:AbstractData, <:Any},
    I,
    v,
)
    dest, bc = pair.first, pair.second
    if 1 ≤ v <= size(dest, 4)
        dest[I] = isascalar(bc) ? bc[] : bc[I]
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pair::Pair{<:DataF, <:Any}, I, v)
    dest, bc = pair.first, pair.second
    if 1 ≤ v <= size(dest, 4)
        bcI = isascalar(bc) ? bc[] : bc[I]
        dest[] = bcI
    end
    return nothing
end
Base.@propagate_inbounds function rcopyto_at!(pairs::Tuple, I, v)
    rcopyto_at!(first(pairs), I, v)
    rcopyto_at!(Base.tail(pairs), I, v)
end
Base.@propagate_inbounds rcopyto_at!(pairs::Tuple{<:Any}, I, v) =
    rcopyto_at!(first(pairs), I, v)
@inline rcopyto_at!(pairs::Tuple{}, I, v) = nothing

function knl_fused_copyto!(fmbc::FusedMultiBroadcast)

    @inbounds begin
        i = CUDA.threadIdx().x
        j = CUDA.threadIdx().y

        h = CUDA.blockIdx().x
        v = CUDA.blockDim().z * (CUDA.blockIdx().y - 1) + CUDA.threadIdx().z
        (; pairs) = fmbc
        I = CartesianIndex((i, j, 1, v, h))
        rcopyto_at!(pairs, I, v)
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VIJFH{S, Nv, Nij},
    ::ToCUDA,
) where {S, Nv, Nij}
    _, _, _, _, Nh = size(dest1)
    if Nv > 0 && Nh > 0
        Nv_per_block = min(Nv, fld(256, Nij * Nij))
        Nv_blocks = cld(Nv, Nv_per_block)
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (Nij, Nij, Nv_per_block),
            blocks_s = (Nh, Nv_blocks),
        )
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::IJFH{S, Nij},
    ::ToCUDA,
) where {S, Nij}
    _, _, _, _, Nh = size(dest1)
    if Nh > 0
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (Nij, Nij),
            blocks_s = (Nh, 1),
        )
    end
    return nothing
end
function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VF{S, Nv},
    ::ToCUDA,
) where {S, Nv}
    _, _, _, _, Nh = size(dest1)
    if Nv > 0 && Nh > 0
        auto_launch!(
            knl_fused_copyto!,
            (fmbc,),
            dest1;
            threads_s = (1, 1),
            blocks_s = (Nh, Nv),
        )
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::DataF{S},
    ::ToCUDA,
) where {S}
    auto_launch!(
        knl_fused_copyto!,
        (fmbc,),
        dest1;
        threads_s = (1, 1),
        blocks_s = (1, 1),
    )
    return nothing
end


adapt_f(to, f::F) where {F} = Adapt.adapt(to, f)
adapt_f(to, ::Type{F}) where {F} = (x...) -> F(x...)

function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    fmbc::FusedMultiBroadcast,
)
    FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            dest = pair.first
            bc = pair.second
            Pair(
                Adapt.adapt(to, dest),
                Base.Broadcast.Broadcasted(
                    bc.style,
                    adapt_f(to, bc.f),
                    Adapt.adapt(to, bc.args),
                    Adapt.adapt(to, bc.axes),
                ),
            )
        end,
    )
end
