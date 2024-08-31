
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

# Can we remove this?
# parent_array_type(
#     ::Type{<:CUDA.CuArray{T, N, B} where {N}},
#     ::Val{ND},
# ) where {T, B, ND} = CUDA.CuArray{T, ND, B}

parent_array_type(
    ::Type{<:CUDA.CuArray{T, N, B} where {N}},
    as::ArraySize,
) where {T, B} = CUDA.CuArray{T, ndims(as), B}

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

import Adapt
import CUDA
function Adapt.adapt_structure(
    to::CUDA.KernelAdaptor,
    bc::DataLayouts.NonExtrudedBroadcasted{Style},
) where {Style}
    DataLayouts.NonExtrudedBroadcasted{Style}(
        adapt_f(to, bc.f),
        Adapt.adapt(to, bc.args),
        Adapt.adapt(to, bc.axes),
    )
end

import ClimaCore.DataLayouts as DL
import CUDA
function CUDA.CuArray(fa::DL.FieldArray{FD}) where {FD}
    arrays = ntuple(Val(DL.ncomponents(fa))) do f
        CUDA.CuArray(fa.arrays[f])
    end
    return DL.FieldArray{FD}(arrays)
end

DL.field_array(array::CUDA.CuArray, as::ArraySize) =
    CUDA.CuArray(DL.field_array(Array(array), as))


# TODO: this could be improved, but it's not typically used at runtime
function copyto_field_array_knl!(x::DL.FieldArray{FD}, y) where {FD}
    gidx =
        CUDA.threadIdx().x + (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x
    I = cart_ind(size(y), gidx)
    x[I] = y[I]
    return nothing
end

@inline function Base.copyto!(
    x::DL.FieldArray{FD, NT},
    y::CUDA.CuArray,
) where {FD, NT <: NTuple}
    if ndims(eltype(NT)) == ndims(y)
        @inbounds for i in 1:DL.tuple_length(NT)
            Base.copyto!(x.arrays[i], y)
        end
    elseif ndims(eltype(NT)) + 1 == ndims(y)
        n = prod(size(y))
        kernel =
            CUDA.@cuda always_inline = true launch = false copyto_field_array_knl!(
                x,
                y,
            )
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(n, config.threads)
        blocks = cld(n, threads)
        kernel(x, y; threads, blocks)
    end
    x
end
