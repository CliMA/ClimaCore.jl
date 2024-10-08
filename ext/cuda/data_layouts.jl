
import ClimaCore.DataLayouts: AbstractData
import ClimaCore.DataLayouts: FusedMultiBroadcast
import ClimaCore.DataLayouts: IJKFVH, IJHF, VIJHF, VIHF, IHF, IJF, IF, VF, DataF
import ClimaCore.DataLayouts: IJHFStyle, VIJHFStyle, VFStyle, DataFStyle
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
include("data_layouts_fused_copyto.jl")
include("data_layouts_mapreduce.jl")
include("data_layouts_threadblock.jl")

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
