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
Data0DStyle(::Type{VFStyle{Nv, A}}) where {Nv, A} = DataFStyle{A}

abstract type DataLevelStyle <: DataStyle end
abstract type Data1DStyle{Ni} <: DataLevelStyle end
struct IFHStyle{Ni, A} <: Data1DStyle{Ni} end
DataStyle(::Type{IFH{S, Ni, A}}) where {S, Ni, A} =
    IFHStyle{Ni, parent_array_type(A)}()
Data0DStyle(::Type{IFHStyle{Ni, A}}) where {Ni, A} = DataFStyle{A}
struct IHFStyle{Ni, A} <: Data1DStyle{Ni} end
DataStyle(::Type{IHF{S, Ni, A}}) where {S, Ni, A} =
    IHFStyle{Ni, parent_array_type(A)}()
Data0DStyle(::Type{IHFStyle{Ni, A}}) where {Ni, A} = DataFStyle{A}

abstract type DataSlab1DStyle{Ni} <: DataLevelStyle end
DataSlab1DStyle(::Type{IFHStyle{Ni, A}}) where {Ni, A} = IFStyle{Ni, A}
DataSlab1DStyle(::Type{IHFStyle{Ni, A}}) where {Ni, A} = IFStyle{Ni, A}

struct IFStyle{Ni, A} <: DataSlab1DStyle{Ni} end
DataStyle(::Type{IF{S, Ni, A}}) where {S, Ni, A} =
    IFStyle{Ni, parent_array_type(A)}()
Data0DStyle(::Type{IFStyle{Ni, A}}) where {Ni, A} = DataFStyle{A}

abstract type DataSlab2DStyle{Nij} <: DataLevelStyle end
struct IJFStyle{Nij, A} <: DataSlab2DStyle{Nij} end
DataStyle(::Type{IJF{S, Nij, A}}) where {S, Nij, A} =
    IJFStyle{Nij, parent_array_type(A)}()
Data0DStyle(::Type{IJFStyle{Nij, A}}) where {Nij, A} = DataFStyle{A}

abstract type Data2DStyle{Nij} <: DataLevelStyle end
struct IJFHStyle{Nij, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} =
    IJFHStyle{Nij, parent_array_type(A)}()
DataSlab2DStyle(::Type{IJFHStyle{Nij, A}}) where {Nij, A} = IJFStyle{Nij, A}
Data0DStyle(::Type{IJFHStyle{Nij, A}}) where {Nij, A} = DataFStyle{A}

struct IJHFStyle{Nij, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJHF{S, Nij, A}}) where {S, Nij, A} =
    IJHFStyle{Nij, parent_array_type(A)}()
DataSlab2DStyle(::Type{IJHFStyle{Nij, A}}) where {Nij, A} = IJFStyle{Nij, A}
Data0DStyle(::Type{IJHFStyle{Nij, A}}) where {Nij, A} = DataFStyle{A}

abstract type Data1DXStyle{Nv, Ni} <: DataStyle end
struct VIFHStyle{Nv, Ni, A} <: Data1DXStyle{Nv, Ni} end
DataStyle(::Type{VIFH{S, Nv, Ni, A}}) where {S, Nv, Ni, A} =
    VIFHStyle{Nv, Ni, parent_array_type(A)}()
Data1DXStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} =
    VIFHStyle{Nv, Ni, A}
DataLevelStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} = IFHStyle{Ni, A}
DataColumnStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} = VFStyle{Nv, A}
DataSlab1DStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} = IFStyle{Ni, A}
Data0DStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Nv, Ni, A} = DataFStyle{A}

struct VIHFStyle{Nv, Ni, A} <: Data1DXStyle{Nv, Ni} end
DataStyle(::Type{VIHF{S, Nv, Ni, A}}) where {S, Nv, Ni, A} =
    VIHFStyle{Nv, Ni, parent_array_type(A)}()
Data1DXStyle(::Type{VIHFStyle{Nv, Ni, A}}) where {Ni, Nv, A} =
    VIHFStyle{Nv, Ni, A}
DataLevelStyle(::Type{VIHFStyle{Nv, Ni, A}}) where {Ni, Nv, A} = IHFStyle{Ni, A}
DataColumnStyle(::Type{VIHFStyle{Nv, Ni, A}}) where {Ni, Nv, A} = VFStyle{Nv, A}
DataSlab1DStyle(::Type{VIHFStyle{Nv, Ni, A}}) where {Ni, Nv, A} = IFStyle{Ni, A}
Data0DStyle(::Type{VIHFStyle{Nv, Ni, A}}) where {Nv, Ni, A} = DataFStyle{A}

abstract type Data2DXStyle{Nv, Nij} <: DataStyle end
struct VIJFHStyle{Nv, Nij, A} <: Data2DXStyle{Nv, Nij} end
DataStyle(::Type{VIJFH{S, Nv, Nij, A}}) where {S, Nv, Nij, A} =
    VIJFHStyle{Nv, Nij, parent_array_type(A)}()
Data2DXStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VIJFHStyle{Nv, Nij, A}
DataLevelStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    IJFHStyle{Nij, A}
DataColumnStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VFStyle{Nv, A}
DataSlab2DStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    IJFStyle{Nij, A}
Data0DStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} = DataFStyle{A}

struct VIJHFStyle{Nv, Nij, A} <: Data2DXStyle{Nv, Nij} end
DataStyle(::Type{VIJHF{S, Nv, Nij, A}}) where {S, Nv, Nij, A} =
    VIJHFStyle{Nv, Nij, parent_array_type(A)}()
Data2DXStyle(::Type{VIJHFStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VIJHFStyle{Nv, Nij, A}
DataLevelStyle(::Type{VIJHFStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    IJHFStyle{Nij, A}
DataColumnStyle(::Type{VIJHFStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VFStyle{Nv, A}
DataSlab2DStyle(::Type{VIJHFStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    IJFStyle{Nij, A}
Data0DStyle(::Type{VIJHFStyle{Nv, Nij, A}}) where {Nv, Nij, A} = DataFStyle{A}

DataLevelStyle(::Type{Style}) where {Style <: DataLevelStyle} = Style
DataLevelStyle(::Type{Style}) where {Style <: DataColumnStyle} =
    Data0DStyle(Style)
DataColumnStyle(::Type{Style}) where {Style <: DataLevelStyle} =
    Data0DStyle(Style)
DataSlabStyle(::Type{Style}) where {Style <: Union{Data1DStyle, Data1DXStyle}} =
    DataSlab1DStyle(Style)
DataSlabStyle(::Type{Style}) where {Style <: Union{Data2DStyle, Data2DXStyle}} =
    DataSlab2DStyle(Style)

#####
##### Union styles
#####

#! format: off
const BroadcastedUnionIJFH{S, Nij, A}      = Union{Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}, IJFH{S, Nij, A}}
const BroadcastedUnionIJHF{S, Nij, A}      = Union{Base.Broadcast.Broadcasted{IJHFStyle{Nij, A}}, IJHF{S, Nij, A}}
const BroadcastedUnionIFH{S, Ni, A}        = Union{Base.Broadcast.Broadcasted{IFHStyle{Ni, A}}, IFH{S, Ni, A}}
const BroadcastedUnionIHF{S, Ni, A}        = Union{Base.Broadcast.Broadcasted{IHFStyle{Ni, A}}, IHF{S, Ni, A}}
const BroadcastedUnionIJF{S, Nij, A}       = Union{Base.Broadcast.Broadcasted{IJFStyle{Nij, A}}, IJF{S, Nij, A}}
const BroadcastedUnionIF{S, Ni, A}         = Union{Base.Broadcast.Broadcasted{IFStyle{Ni, A}}, IF{S, Ni, A}}
const BroadcastedUnionVIFH{S, Nv, Ni, A}   = Union{Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}}, VIFH{S, Nv, Ni, A}}
const BroadcastedUnionVIHF{S, Nv, Ni, A}   = Union{Base.Broadcast.Broadcasted{VIHFStyle{Nv, Ni, A}}, VIHF{S, Nv, Ni, A}}
const BroadcastedUnionVIJFH{S, Nv, Nij, A} = Union{Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}}, VIJFH{S, Nv, Nij, A}}
const BroadcastedUnionVIJHF{S, Nv, Nij, A} = Union{Base.Broadcast.Broadcasted{VIJHFStyle{Nv, Nij, A}}, VIJHF{S, Nv, Nij, A}}
const BroadcastedUnionVF{S, Nv, A}         = Union{Base.Broadcast.Broadcasted{VFStyle{Nv, A}}, VF{S, Nv, A}}
const BroadcastedUnionDataF{S, A}          = Union{Base.Broadcast.Broadcasted{DataFStyle{A}}, DataF{S, A}}
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
    ::IFHStyle{Ni, A1},
    ::IFHStyle{Ni, A2},
) where {Ni, A1, A2} = IFHStyle{Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IHFStyle{Ni, A1},
    ::IHFStyle{Ni, A2},
) where {Ni, A1, A2} = IHFStyle{Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIFHStyle{Nv, Ni, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIHFStyle{Nv, Ni, A1},
    ::VIHFStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIHFStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFStyle{Nij, A1},
    ::IJFStyle{Nij, A2},
) where {Nij, A1, A2} = IJFStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, A1},
    ::IJFHStyle{Nij, A2},
) where {Nij, A1, A2} = IJFHStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJHFStyle{Nij, A1},
    ::IJHFStyle{Nij, A2},
) where {Nij, A1, A2} = IJHFStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIJFHStyle{Nv, Nij, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIJHFStyle{Nv, Nij, A1},
    ::VIJHFStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJHFStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

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
    ::IFHStyle{Ni, A2},
) where {Ni, A1, A2} = IFHStyle{Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IHFStyle{Ni, A2},
) where {Ni, A1, A2} = IHFStyle{Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IJFHStyle{Nij, A2},
) where {Nij, A1, A2} = IJFHStyle{Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::IJHFStyle{Nij, A2},
) where {Nij, A1, A2} = IJHFStyle{Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIHFStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIHFStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIJHFStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJHFStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IFHStyle{Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IHFStyle{Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIHFStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IJFHStyle{Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IJHFStyle{Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJHFStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIHFStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIHFStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIJHFStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJHFStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, A1},
    ::VIHFStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIHFStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IJHFStyle{Nij, A1},
    ::VIJHFStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJHFStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

broadcast_arg(arg) = arg
broadcast_arg(arg::Ref) = Ref(enable_auto_broadcasting(arg[]))
broadcast_arg(arg::Tuple{<:Any}) = (enable_auto_broadcasting(arg[1]),)

# Enable automatic broadcasting over supported types of iterators, in addition
# to the default broadcasting over array indices.
Base.Broadcast.broadcastable(data::AbstractData) =
    is_auto_broadcastable(eltype(data)) ?
    reinterpret(inferred_type(enable_auto_broadcasting, eltype(data)), data) :
    data
Base.Broadcast.broadcastable(bc::Base.Broadcast.Broadcasted{<:DataStyle}) =
    is_auto_broadcastable(eltype(bc)) ?
    Base.Broadcast.broadcasted(bc.style, enable_auto_broadcasting, bc) : bc

Base.Broadcast.broadcasted(ds::DataStyle, f::F, args...) where {F} =
    Base.Broadcast.Broadcasted(ds, f, unrolled_map(broadcast_arg, args))

Base.eltype(bc::Base.Broadcast.Broadcasted{<:DataStyle}) = broadcast_eltype(bc)

Base.copy(bc::Base.Broadcast.Broadcasted{<:DataStyle}) =
    copyto!(similar(bc), bc)

# Remove all AutoBroadcaster wrappers before allocating a new AbstractData.
Base.similar(bc::Base.Broadcast.Broadcasted{<:DataStyle}) =
    similar(bc, inferred_type(disable_auto_broadcasting, eltype(bc)))

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

Base.@propagate_inbounds function level(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {DS <: DataStyle}
    _args = level_args(bc.args, inds...)
    _axes = nothing
    bcc = Base.Broadcast.Broadcasted{DataLevelStyle(DS)}(bc.f, _args, _axes)
    Base.Broadcast.instantiate(bcc)
end

@inline function level(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {DS <: DataLevelStyle}
    bc
end

Base.@propagate_inbounds function column(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nv, N, DS <: Union{Data1DXStyle{Nv, N}, Data2DXStyle{Nv, N}}}
    _args = column_args(bc.args, inds...)
    _axes = nothing
    bcc = Base.Broadcast.Broadcasted{DataColumnStyle(DS)}(bc.f, _args, _axes)
    Base.Broadcast.instantiate(bcc)
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
    bc::BroadcastedUnionIJFH{<:Any, Nij, A},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Nij, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return IJFH{Eltype, Nij}(array)
end

function Base.similar(
    bc::BroadcastedUnionIJHF{<:Any, Nij, A},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Nij, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Nij, Nij, Nh, typesize(eltype(A), Eltype)))
    return IJHF{Eltype, Nij}(array)
end

function Base.similar(
    bc::BroadcastedUnionIFH{<:Any, Ni, A},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Ni, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Ni, typesize(eltype(A), Eltype), Nh))
    return IFH{Eltype, Ni}(array)
end

function Base.similar(
    bc::BroadcastedUnionIHF{<:Any, Ni, A},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Ni, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Ni, Nh, typesize(eltype(A), Eltype)))
    return IHF{Eltype, Ni}(array)
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
    bc::Union{BroadcastedUnionVIFH{<:Any, Nv}, BroadcastedUnionVIHF{<:Any, Nv}},
    ::Type{Eltype},
) where {Nv, Eltype} = Base.similar(bc, Eltype, Val(Nv))

function Base.similar(
    bc::BroadcastedUnionVIFH{<:Any, Nv, Ni, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Ni, A, Eltype, newNv}
    (_, _, _, _, Nh) = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Ni, typesize(eltype(A), Eltype), Nh))
    return VIFH{Eltype, newNv, Ni}(array)
end

function Base.similar(
    bc::BroadcastedUnionVIHF{<:Any, Nv, Ni, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Ni, A, Eltype, newNv}
    (_, _, _, _, Nh) = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Ni, Nh, typesize(eltype(A), Eltype)))
    return VIHF{Eltype, newNv, Ni}(array)
end

Base.similar(
    bc::BroadcastedUnionVIJFH{<:Any, Nv, Nij, A},
    ::Type{Eltype},
) where {Nv, Nij, A, Eltype} = similar(bc, Eltype, Val(Nv))

Base.similar(
    bc::BroadcastedUnionVIJHF{<:Any, Nv, Nij, A},
    ::Type{Eltype},
) where {Nv, Nij, A, Eltype} = similar(bc, Eltype, Val(Nv))

function Base.similar(
    bc::BroadcastedUnionVIJFH{<:Any, Nv, Nij, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Nij, A, Eltype, newNv}
    (_, _, _, _, Nh) = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return VIJFH{Eltype, newNv, Nij}(array)
end

function Base.similar(
    bc::BroadcastedUnionVIJHF{<:Any, Nv, Nij, A},
    ::Type{Eltype},
    ::Val{newNv},
) where {Nv, Nij, A, Eltype, newNv}
    (_, _, _, _, Nh) = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Nij, Nij, Nh, typesize(eltype(A), Eltype)))
    return VIJHF{Eltype, newNv, Nij}(array)
end

# ============= FusedMultiBroadcast

isascalar(
    bc::Base.Broadcast.Broadcasted{Style},
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
} = true
isascalar(
    bc::NonExtrudedBroadcasted{Style},
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
} = true
isascalar(bc) = false
