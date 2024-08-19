import MultiBroadcastFusion as MBF
import MultiBroadcastFusion: fused_direct

# Make a MultiBroadcastFusion type, `FusedMultiBroadcast`, and macro, `@fused`:
# via https://github.com/CliMA/MultiBroadcastFusion.jl
MBF.@make_type FusedMultiBroadcast
MBF.@make_fused fused_direct FusedMultiBroadcast fused_direct

# Broadcasting of AbstractData objects
# https://docs.julialang.org/en/v1/manual/interfaces/#Broadcast-Styles

abstract type DataStyle <: Base.BroadcastStyle end

abstract type Data0DStyle <: DataStyle end
struct DataFStyle{A} <: Data0DStyle end
DataStyle(::Type{DataF{S, A}}) where {S, A} = DataFStyle{parent_array_type(A)}()
Data0DStyle(::Type{DataFStyle{A}}) where {A} = DataFStyle{A}

abstract type DataColumnStyle <: DataStyle end
struct VFStyle{Nv, A} <: DataColumnStyle end
DataStyle(::Type{VF{S, Nv, A}}) where {S, Nv, A} =
    VFStyle{Nv, parent_array_type(A)}()
DataColumnStyle(::Type{VFStyle{Nv, A}}) where {Nv, A} = VFStyle{Nv, A}

abstract type Data1DStyle{Ni} <: DataStyle end
struct IFHStyle{Ni, Nh, A} <: Data1DStyle{Ni} end
DataStyle(::Type{IFH{S, Ni, Nh, A}}) where {S, Ni, Nh, A} =
    IFHStyle{Ni, Nh, parent_array_type(A)}()

abstract type DataSlab1DStyle{Ni} <: DataStyle end
DataSlab1DStyle(::Type{IFHStyle{Ni, Nh, A}}) where {Ni, Nh, A} = IFStyle{Ni, A}

struct IFStyle{Ni, A} <: DataSlab1DStyle{Ni} end
DataStyle(::Type{IF{S, Ni, A}}) where {S, Ni, A} =
    IFStyle{Ni, parent_array_type(A)}()

abstract type DataSlab2DStyle{Nij} <: DataStyle end
struct IJFStyle{Nij, A} <: DataSlab2DStyle{Nij} end
DataStyle(::Type{IJF{S, Nij, A}}) where {S, Nij, A} =
    IJFStyle{Nij, parent_array_type(A)}()

abstract type Data2DStyle{Nij} <: DataStyle end
struct IJFHStyle{Nij, Nh, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJFH{S, Nij, Nh, A}}) where {S, Nij, Nh, A} =
    IJFHStyle{Nij, Nh, parent_array_type(A)}()
DataSlab2DStyle(::Type{IJFHStyle{Nij, Nh, A}}) where {Nij, Nh, A} =
    IJFStyle{Nij, A}

abstract type Data1DXStyle{Nv, Ni} <: DataStyle end
struct VIFHStyle{Nv, Ni, Nh, A} <: Data1DXStyle{Nv, Ni} end
DataStyle(::Type{VIFH{S, Nv, Ni, Nh, A}}) where {S, Nv, Ni, Nh, A} =
    VIFHStyle{Nv, Ni, Nh, parent_array_type(A)}()
Data1DXStyle(::Type{VIFHStyle{Nv, Ni, Nh, A}}) where {Ni, Nv, Nh, A} =
    VIFHStyle{Nv, Ni, Nh, A}
DataColumnStyle(::Type{VIFHStyle{Nv, Ni, Nh, A}}) where {Ni, Nv, Nh, A} =
    VFStyle{Nv, A}
DataSlab1DStyle(::Type{VIFHStyle{Nv, Ni, Nh, A}}) where {Ni, Nv, Nh, A} =
    IFStyle{Ni, A}

abstract type Data2DXStyle{Nv, Nij} <: DataStyle end
struct VIJFHStyle{Nv, Nij, Nh, A} <: Data2DXStyle{Nv, Nij} end
DataStyle(::Type{VIJFH{S, Nv, Nij, Nh, A}}) where {S, Nv, Nij, Nh, A} =
    VIJFHStyle{Nv, Nij, Nh, parent_array_type(A)}()
Data2DXStyle(::Type{VIJFHStyle{Nv, Nij, Nh, A}}) where {Nv, Nij, Nh, A} =
    VIJFHStyle{Nv, Nij, Nh, A}
DataColumnStyle(::Type{VIJFHStyle{Nv, Nij, Nh, A}}) where {Nv, Nij, Nh, A} =
    VFStyle{Nv, A}
DataSlab2DStyle(::Type{VIJFHStyle{Nv, Nij, Nh, A}}) where {Nv, Nij, Nh, A} =
    IJFStyle{Nij, A}

#####
##### Union styles
#####

#! format: off
const BroadcastedUnionIJFH{S, Nij, Nh, A}      = Union{Base.Broadcast.Broadcasted{IJFHStyle{Nij, Nh, A}}, IJFH{S, Nij, Nh, A}}
const BroadcastedUnionIFH{S, Ni, Nh, A}        = Union{Base.Broadcast.Broadcasted{IFHStyle{Ni, Nh, A}}, IFH{S, Ni, Nh, A}}
const BroadcastedUnionIJF{S, Nij, A}           = Union{Base.Broadcast.Broadcasted{IJFStyle{Nij, A}}, IJF{S, Nij, A}}
const BroadcastedUnionIF{S, Ni, A}             = Union{Base.Broadcast.Broadcasted{IFStyle{Ni, A}}, IF{S, Ni, A}}
const BroadcastedUnionVIFH{S, Nv, Ni, Nh, A}   = Union{Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, Nh, A}}, VIFH{S, Nv, Ni, Nh, A}}
const BroadcastedUnionVIJFH{S, Nv, Nij, Nh, A} = Union{Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, Nh, A}}, VIJFH{S, Nv, Nij, Nh, A}}
const BroadcastedUnionVF{S, Nv, A}             = Union{Base.Broadcast.Broadcasted{VFStyle{Nv, A}}, VF{S, Nv, A}}
const BroadcastedUnionDataF{S, A}              = Union{Base.Broadcast.Broadcasted{DataFStyle{A}}, DataF{S, A}}
#! format: on

abstract type Data3DStyle <: DataStyle end

Base.Broadcast.BroadcastStyle(::Type{D}) where {D <: AbstractData} =
    DataStyle(D)

# precedence rules

# scalars are broadcast over the data object
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    ds::DataStyle,
) = ds

Base.Broadcast.BroadcastStyle(::Base.Broadcast.Style{Tuple}, ds::DataStyle) = ds

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::DataFStyle{A2},
) where {A1, A2} = DataFStyle{promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VFStyle{Nv, A2},
) where {Nv, A1, A2} = VFStyle{Nv, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IFStyle{Ni, A1},
    ::IFStyle{Ni, A2},
) where {Ni, A1, A2} = IFStyle{Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, Nh, A1},
    ::IFHStyle{Ni, Nh, A2},
) where {Ni, Nh, A1, A2} = IFHStyle{Ni, Nh, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIFHStyle{Nv, Ni, Nh, A1},
    ::VIFHStyle{Nv, Ni, Nh, A2},
) where {Nv, Ni, Nh, A1, A2} =
    VIFHStyle{Nv, Ni, Nh, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFStyle{Nij, A1},
    ::IJFStyle{Nij, A2},
) where {Nij, A1, A2} = IJFStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, Nh, A1},
    ::IJFHStyle{Nij, Nh, A2},
) where {Nij, Nh, A1, A2} =
    IJFHStyle{Nij, Nh, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIJFHStyle{Nv, Nij, Nh, A1},
    ::VIJFHStyle{Nv, Nij, Nh, A2},
) where {Nv, Nij, Nh, A1, A2} =
    VIJFHStyle{Nv, Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IFStyle{Ni, A2},
) where {Ni, A1, A2} = IFStyle{Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IJFStyle{Nij, A2},
) where {Nij, A1, A2} = IJFStyle{Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VFStyle{Nv, A2},
) where {A1, Nv, A2} = VFStyle{Nv, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IFHStyle{Ni, Nh, A2},
) where {Ni, Nh, A1, A2} = IFHStyle{Ni, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IJFHStyle{Nij, Nh, A2},
) where {Nij, Nh, A1, A2} =
    IJFHStyle{Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIFHStyle{Nv, Ni, Nh, A2},
) where {Nv, Ni, Nh, A1, A2} =
    VIFHStyle{Nv, Ni, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIJFHStyle{Nv, Nij, Nh, A2},
) where {Nv, Nij, Nh, A1, A2} =
    VIJFHStyle{Nv, Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IFHStyle{Ni, Nh, A2},
) where {Nv, Ni, Nh, A1, A2} =
    VIFHStyle{Nv, Ni, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IJFHStyle{Nij, Nh, A2},
) where {Nv, Nij, Nh, A1, A2} =
    VIJFHStyle{Nv, Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIFHStyle{Nv, Ni, Nh, A2},
) where {Nv, Ni, Nh, A1, A2} =
    VIFHStyle{Nv, Ni, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIJFHStyle{Nv, Nij, Nh, A2},
) where {Nv, Nij, Nh, A1, A2} =
    VIJFHStyle{Nv, Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, Nh, A1},
    ::VIFHStyle{Nv, Ni, Nh, A2},
) where {Nv, Ni, Nh, A1, A2} =
    VIFHStyle{Nv, Ni, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, Nh, A1},
    ::VIJFHStyle{Nv, Nij, Nh, A2},
) where {Nv, Nij, Nh, A1, A2} =
    VIJFHStyle{Nv, Nij, Nh, promote_parent_array_type(A1, A2)}()

Base.Broadcast.broadcastable(data::AbstractData) = data

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Ni, DS <: Data1DStyle{Ni}}
    _args = slab_args(bc.args, inds...)
    _axes = (SOneTo(Ni),)
    Base.Broadcast.Broadcasted{DataSlab1DStyle(DS)}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nv, Ni, DS <: Data1DXStyle{Nv, Ni}}
    _args = slab_args(bc.args, inds...)
    _axes = (SOneTo(Ni),)
    Base.Broadcast.Broadcasted{DataSlab1DStyle(DS)}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nij, DS <: Data2DStyle{Nij}}
    _args = slab_args(bc.args, inds...)
    _axes = (SOneTo(Nij), SOneTo(Nij))
    Base.Broadcast.Broadcasted{DataSlab2DStyle(DS)}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nv, Nij, DS <: Data2DXStyle{Nv, Nij}}
    _args = slab_args(bc.args, inds...)
    _axes = (SOneTo(Nij), SOneTo(Nij))
    Base.Broadcast.Broadcasted{DataSlab2DStyle(DS)}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function column(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nv, N, DS <: Union{Data1DXStyle{Nv, N}, Data2DXStyle{Nv, N}}}
    _args = column_args(bc.args, inds...)
    _axes = nothing
    Base.Broadcast.Broadcasted{DataColumnStyle(DS)}(bc.f, _args, _axes)
end

@inline function column(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {DS <: DataColumnStyle}
    bc
end

Base.@propagate_inbounds function column(
    bc::Union{Data1D, Base.Broadcast.Broadcasted{<:Data1D}},
    i,
    h,
)
    slab(bc, h)[i]
end
Base.@propagate_inbounds function column(
    bc::Union{Data1D, Base.Broadcast.Broadcasted{<:Data1D}},
    i,
    j,
    h,
)
    slab(bc, h)[i]
end

Base.@propagate_inbounds function column(
    bc::Union{Data2D, Base.Broadcast.Broadcasted{<:Data2D}},
    i,
    j,
    h,
)
    slab(bc, h)[i, j]
end

function Base.similar(
    bc::BroadcastedUnionDataF{<:Any, A},
    ::Type{Eltype},
) where {A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (typesize(eltype(A), Eltype)))
    return DataF{Eltype}(array)
end

function Base.similar(
    bc::BroadcastedUnionIJFH{<:Any, Nij, Nh, A},
    ::Type{Eltype},
) where {Nij, Nh, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return IJFH{Eltype, Nij, Nh}(array)
end

function Base.similar(
    bc::BroadcastedUnionIFH{<:Any, Ni, Nh, A},
    ::Type{Eltype},
) where {Ni, Nh, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Ni, typesize(eltype(A), Eltype), Nh))
    return IFH{Eltype, Ni, Nh}(array)
end

function Base.similar(
    ::BroadcastedUnionIJF{<:Any, Nij, A},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    array = MArray{Tuple{Nij, Nij, Nf}, eltype(A), 3, Nij * Nij * Nf}(undef)
    return IJF{Eltype, Nij}(array)
end

function Base.similar(
    ::BroadcastedUnionIF{<:Any, Ni, A},
    ::Type{Eltype},
) where {Ni, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    array = MArray{Tuple{Ni, Nf}, eltype(A), 2, Ni * Nf}(undef)
    return IF{Eltype, Ni}(array)
end

Base.similar(
    bc::BroadcastedUnionVF{<:Any, Nv},
    ::Type{Eltype},
) where {Nv, Eltype} = Base.similar(bc, Eltype, Val(Nv))

function Base.similar(
    bc::BroadcastedUnionVF{<:Any, Nv, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, A, Eltype, newNv}
    PA = parent_array_type(A)
    array = similar(PA, (newNv, typesize(eltype(A), Eltype)))
    return VF{Eltype, newNv}(array)
end

Base.similar(
    bc::BroadcastedUnionVIFH{<:Any, Nv},
    ::Type{Eltype},
) where {Nv, Eltype} = Base.similar(bc, Eltype, Val(Nv))

function Base.similar(
    bc::BroadcastedUnionVIFH{<:Any, Nv, Ni, Nh, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Ni, Nh, A, Eltype, newNv}
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Ni, typesize(eltype(A), Eltype), Nh))
    return VIFH{Eltype, newNv, Ni, Nh}(array)
end

Base.similar(
    bc::BroadcastedUnionVIJFH{<:Any, Nv, Nij, Nh, A},
    ::Type{Eltype},
) where {Nv, Nij, Nh, A, Eltype} = similar(bc, Eltype, Val(Nv))

function Base.similar(
    bc::BroadcastedUnionVIJFH{<:Any, Nv, Nij, Nh, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Nij, Nh, A, Eltype, newNv}
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return VIJFH{Eltype, newNv, Nij, Nh}(array)
end

# ============= FusedMultiBroadcast

isascalar(
    bc::Base.Broadcast.Broadcasted{Style},
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
} = true
isascalar(bc) = false
