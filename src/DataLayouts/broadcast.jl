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
struct IFHStyle{Ni, A} <: Data1DStyle{Ni} end
DataStyle(::Type{IFH{S, Ni, A}}) where {S, Ni, A} =
    IFHStyle{Ni, parent_array_type(A)}()

abstract type DataSlab1DStyle{Ni} <: DataStyle end
DataSlab1DStyle(::Type{IFHStyle{Ni, A}}) where {Ni, A} = IFStyle{Ni, A}

struct IFStyle{Ni, A} <: DataSlab1DStyle{Ni} end
DataStyle(::Type{IF{S, Ni, A}}) where {S, Ni, A} =
    IFStyle{Ni, parent_array_type(A)}()

abstract type DataSlab2DStyle{Nij} <: DataStyle end
struct IJFStyle{Nij, A} <: DataSlab2DStyle{Nij} end
DataStyle(::Type{IJF{S, Nij, A}}) where {S, Nij, A} =
    IJFStyle{Nij, parent_array_type(A)}()

abstract type Data2DStyle{Nij} <: DataStyle end
struct IJFHStyle{Nij, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} =
    IJFHStyle{Nij, parent_array_type(A)}()
DataSlab2DStyle(::Type{IJFHStyle{Nij, A}}) where {Nij, A} = IJFStyle{Nij, A}

abstract type Data1DXStyle{Nv, Ni} <: DataStyle end
struct VIFHStyle{Nv, Ni, A} <: Data1DXStyle{Nv, Ni} end
DataStyle(::Type{VIFH{S, Nv, Ni, A}}) where {S, Nv, Ni, A} =
    VIFHStyle{Nv, Ni, parent_array_type(A)}()
Data1DXStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} =
    VIFHStyle{Nv, Ni, A}
DataColumnStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} = VFStyle{Nv, A}
DataSlab1DStyle(::Type{VIFHStyle{Nv, Ni, A}}) where {Ni, Nv, A} = IFStyle{Ni, A}

abstract type Data2DXStyle{Nv, Nij} <: DataStyle end
struct VIJFHStyle{Nv, Nij, A} <: Data2DXStyle{Nv, Nij} end
DataStyle(::Type{VIJFH{S, Nv, Nij, A}}) where {S, Nv, Nij, A} =
    VIJFHStyle{Nv, Nij, parent_array_type(A)}()
Data2DXStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VIJFHStyle{Nv, Nij, A}
DataColumnStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    VFStyle{Nv, A}
DataSlab2DStyle(::Type{VIJFHStyle{Nv, Nij, A}}) where {Nv, Nij, A} =
    IJFStyle{Nij, A}

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
    ::VIFHStyle{Nv, Ni, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFStyle{Nij, A1},
    ::IJFStyle{Nij, A2},
) where {Nij, A1, A2} = IJFStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, A1},
    ::IJFHStyle{Nij, A2},
) where {Nij, A1, A2} = IJFHStyle{Nij, promote_parent_array_type(A1, A2)}()
Base.Broadcast.BroadcastStyle(
    ::VIJFHStyle{Nv, Nij, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

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
    ::IJFHStyle{Nij, A2},
) where {Nij, A1, A2} = IJFHStyle{Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::DataFStyle{A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IFHStyle{Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::IJFHStyle{Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{Nv, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, A1},
    ::VIFHStyle{Nv, Ni, A2},
) where {Nv, Ni, A1, A2} =
    VIFHStyle{Nv, Ni, promote_parent_array_type(A1, A2)}()

Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, A1},
    ::VIJFHStyle{Nv, Nij, A2},
) where {Nv, Nij, A1, A2} =
    VIJFHStyle{Nv, Nij, promote_parent_array_type(A1, A2)}()

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
    bc::Union{DataF{<:Any, A}, Broadcast.Broadcasted{DataFStyle{A}}},
    ::Type{Eltype},
) where {A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (typesize(eltype(A), Eltype)))
    return DataF{Eltype}(array)
end

function Base.similar(
    bc::Union{IJFH{<:Any, Nij, A}, Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Nij, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return IJFH{Eltype, Nij}(array)
end

function Base.similar(
    bc::Union{IFH{<:Any, Ni, A}, Broadcast.Broadcasted{IFHStyle{Ni, A}}},
    ::Type{Eltype},
    (_, _, _, _, Nh) = size(bc),
) where {Ni, A, Eltype}
    PA = parent_array_type(A)
    array = similar(PA, (Ni, typesize(eltype(A), Eltype), Nh))
    return IFH{Eltype, Ni}(array)
end

function Base.similar(
    ::Union{IJF{<:Any, Nij, A}, Broadcast.Broadcasted{IJFStyle{Nij, A}}},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    array = MArray{Tuple{Nij, Nij, Nf}, eltype(A), 3, Nij * Nij * Nf}(undef)
    return IJF{Eltype, Nij}(array)
end

function Base.similar(
    ::Union{IF{<:Any, Ni, A}, Broadcast.Broadcasted{IFStyle{Ni, A}}},
    ::Type{Eltype},
) where {Ni, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    array = MArray{Tuple{Ni, Nf}, eltype(A), 2, Ni * Nf}(undef)
    return IF{Eltype, Ni}(array)
end

function Base.similar(
    bc::Union{VF{<:Any, Nv, A}, Broadcast.Broadcasted{VFStyle{Nv, A}}},
    ::Type{Eltype},
    dims = nothing,
) where {Nv, A, Eltype}
    (_, _, _, newNv, Nh) = isnothing(dims) ? size(bc) : dims
    PA = parent_array_type(A)
    array = similar(PA, (newNv, typesize(eltype(A), Eltype)))
    return VF{Eltype, newNv}(array)
end

function Base.similar(
    bc::Union{
        VIFH{<:Any, Nv, Ni, A},
        Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}},
    },
    ::Type{Eltype},
    dims = nothing,
) where {Nv, Ni, A, Eltype}
    (_, _, _, newNv, Nh) = isnothing(dims) ? size(bc) : dims
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Ni, typesize(eltype(A), Eltype), Nh))
    return VIFH{Eltype, newNv, Ni}(array)
end

function Base.similar(
    bc::Union{
        VIJFH{<:Any, Nv, Nij, A},
        Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}},
    },
    ::Type{Eltype},
    dims = nothing,
) where {Nv, Nij, A, Eltype}
    (_, _, _, newNv, Nh) = isnothing(dims) ? size(bc) : dims
    PA = parent_array_type(A)
    array = similar(PA, (newNv, Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return VIJFH{Eltype, newNv, Nij}(array)
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{DataF{<:Any, A}, Base.Broadcast.Broadcasted{DataFStyle{A}}},
) where {F, Op, A}
    mapreduce(op, 1) do v
        Base.@_inline_meta
        @inbounds fn(bc[])
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{
        IJFH{<:Any, Nij, A},
        Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
    },
) where {F, Op, Nij, A}
    # mapreduce across DataSlab2D
    _, _, _, _, Nh = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce(fn, op, slabview)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{IFH{<:Any, Ni, A}, Base.Broadcast.Broadcasted{IFHStyle{Ni, A}}},
) where {F, Op, Ni, A}
    # mapreduce across DataSlab1D
    _, _, _, _, Nh = size(bc)
    mapreduce(op, 1:Nh) do h
        Base.@_inline_meta
        slabview = @inbounds slab(bc, h)
        mapreduce(fn, op, slabview)
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IJF{S, Nij}) where {F, Op, S, Nij}
    # mapreduce across DataSlab2D nodes
    mapreduce(op, Iterators.product(1:Nij, 1:Nij)) do (i, j)
        Base.@_inline_meta
        idx = CartesianIndex(i, j, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IF{S, Ni}) where {F, Op, S, Ni}
    # mapreduce across DataSlab1D nodes
    mapreduce(op, 1:Ni) do i
        Base.@_inline_meta
        idx = CartesianIndex(i, 1, 1, 1, 1)
        node = @inbounds bc[idx]
        fn(node)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{VF{<:Any, Nv, A}, Base.Broadcast.Broadcasted{VFStyle{Nv, A}}},
) where {F, Op, Nv, A}
    # mapreduce across DataColumn levels
    mapreduce(op, 1:Nv) do v
        Base.@_inline_meta
        idx = CartesianIndex(1, 1, 1, v, 1)
        level = @inbounds bc[idx]
        fn(level)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{
        VIFH{<:Any, Nv, Ni, A},
        Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}},
    },
) where {F, Op, Nv, Ni, A}
    # mapreduce across columns
    _, _, _, _, Nh = size(bc)
    mapreduce(op, Iterators.product(1:Ni, 1:Nh)) do (i, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, h)
        mapreduce(fn, op, columnview)
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{
        VIJFH{<:Any, Nv, Nij, A},
        Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}},
    },
) where {F, Op, Nv, Nij, A}
    # mapreduce across columns
    _, _, _, _, Nh = size(bc)
    mapreduce(op, Iterators.product(1:Nij, 1:Nij, 1:Nh)) do (i, j, h)
        Base.@_inline_meta
        columnview = @inbounds column(bc, i, j, h)
        mapreduce(fn, op, columnview)
    end
end

# broadcasting scalar assignment
# Performance optimization for the common identity scalar case: dest .= val
function Base.copyto!(
    dest::AbstractData,
    bc::Base.Broadcast.Broadcasted{Style},
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
}
    bc = Base.Broadcast.instantiate(
        Base.Broadcast.Broadcasted{Style}(bc.f, bc.args, ()),
    )
    @inbounds bc0 = bc[]
    fill!(dest, bc0)
end

function Base.copyto!(
    dest::DataF{S},
    bc::Union{DataF{S, A}, Base.Broadcast.Broadcasted{DataFStyle{A}}},
) where {S, A}
    @inbounds dest[] = convert(S, bc[])
    return dest
end

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Union{IJFH{S, Nij}, Base.Broadcast.Broadcasted{<:IJFHStyle{Nij}}},
) where {S, Nij}
    _, _, _, _, Nh = size(bc)
    @inbounds for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

function Base.copyto!(
    dest::IFH{S, Ni},
    bc::Union{IFH{S, Ni}, Base.Broadcast.Broadcasted{<:IFHStyle{Ni}}},
) where {S, Ni}
    _, _, _, _, Nh = size(bc)
    @inbounds for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

# inline inner slab(::DataSlab2D) copy
function Base.copyto!(
    dest::IJF{S, Nij},
    bc::Union{IJF{S, Nij, A}, Base.Broadcast.Broadcasted{IJFStyle{Nij, A}}},
) where {S, Nij, A}
    @inbounds for j in 1:Nij, i in 1:Nij
        idx = CartesianIndex(i, j, 1, 1, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

# inline inner slab(::DataSlab1D) copy
function Base.copyto!(
    dest::IF{S, Ni},
    bc::Base.Broadcast.Broadcasted{IFStyle{Ni, A}},
) where {S, Ni, A}
    @inbounds for i in 1:Ni
        idx = CartesianIndex(i, 1, 1, 1, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

# inline inner column(::DataColumn) copy
function Base.copyto!(
    dest::VF{S, Nv},
    bc::Union{VF{S, Nv, A}, Base.Broadcast.Broadcasted{VFStyle{Nv, A}}},
) where {S, Nv, A}
    @inbounds for v in 1:Nv
        idx = CartesianIndex(1, 1, 1, v, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function _serial_copyto!(
    dest::VIFH{S, Nv, Ni},
    bc::Union{
        VIFH{S, Nv, Ni, A},
        Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}},
    },
) where {S, Nv, Ni, A}
    (_, _, _, _, Nh) = size(bc)
    # copy contiguous columns
    @inbounds for h in 1:Nh, i in 1:Ni
        col_dest = column(dest, i, h)
        col_bc = column(bc, i, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

function _threaded_copyto!(
    dest::VIFH{S, Nv, Ni},
    bc::Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}},
) where {S, Nv, Ni, A}
    _, _, _, _, Nh = size(dest)
    # parallelize over elements
    @inbounds begin
        Threads.@threads for h in 1:Nh
            # copy contiguous columns
            for i in 1:Ni
                col_dest = column(dest, i, h)
                col_bc = column(bc, i, h)
                copyto!(col_dest, col_bc)
            end
        end
    end
    return dest
end

function Base.copyto!(
    dest::VIFH{S, Nv, Ni},
    source::VIFH{S, Nv, Ni, A},
) where {S, Nv, Ni, A}
    return _serial_copyto!(dest, source)
end

function Base.copyto!(
    dest::VIFH{S, Nv, Ni},
    bc::Base.Broadcast.Broadcasted{VIFHStyle{Nv, Ni, A}},
) where {S, Nv, Ni, A}
    return _serial_copyto!(dest, bc)
end

function _serial_copyto!(
    dest::VIJFH{S, Nv, Nij},
    bc::Union{
        VIJFH{S, Nv, Nij, A},
        Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}},
    },
) where {S, Nv, Nij, A}
    # copy contiguous columns
    _, _, _, _, Nh = size(dest)
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
        col_dest = column(dest, i, j, h)
        col_bc = column(bc, i, j, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

function _threaded_copyto!(
    dest::VIJFH{S, Nv, Nij},
    bc::Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}},
) where {S, Nv, Nij, A}
    _, _, _, _, Nh = size(dest)
    # parallelize over elements
    @inbounds begin
        Threads.@threads for h in 1:Nh
            # copy contiguous columns
            for j in 1:Nij, i in 1:Nij
                col_dest = column(dest, i, j, h)
                col_bc = column(bc, i, j, h)
                copyto!(col_dest, col_bc)
            end
        end
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij},
    source::VIJFH{S, Nv, Nij, A},
) where {S, Nv, Nij, A}
    return _serial_copyto!(dest, source)
end

function Base.copyto!(
    dest::VIJFH{S, Nv, Nij},
    bc::Base.Broadcast.Broadcasted{VIJFHStyle{Nv, Nij, A}},
) where {S, Nv, Nij, A}
    return _serial_copyto!(dest, bc)
end

# ============= FusedMultiBroadcast

isascalar(
    bc::Base.Broadcast.Broadcasted{Style},
) where {
    Style <:
    Union{Base.Broadcast.AbstractArrayStyle{0}, Base.Broadcast.Style{Tuple}},
} = true
isascalar(bc) = false

# Fused multi-broadcast entry point for DataLayouts
function Base.copyto!(
    fmbc::FusedMultiBroadcast{T},
) where {N, T <: NTuple{N, Pair{<:AbstractData, <:Any}}}
    dest1 = first(fmbc.pairs).first
    fmb_inst = FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            bc = pair.second
            bc′ = if isascalar(bc)
                Base.Broadcast.instantiate(
                    Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, ()),
                )
            else
                bc
            end
            Pair(pair.first, bc′)
        end,
    )
    # check_fused_broadcast_axes(fmbc) # we should already have checked the axes
    fused_copyto!(fmb_inst, dest1)
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VIJFH{S1, Nv1, Nij},
) where {S1, Nv1, Nij}
    _, _, _, _, Nh = size(dest1)
    for (dest, bc) in fmbc.pairs
        # Base.copyto!(dest, bc) # we can just fall back like this
        @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij, v in 1:Nv1
            I = CartesianIndex(i, j, 1, v, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::IJFH{S, Nij},
) where {S, Nij}
    # copy contiguous columns
    _, _, _, Nv, Nh = size(dest1)
    for (dest, bc) in fmbc.pairs
        @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
            I = CartesianIndex(i, j, 1, 1, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VIFH{S, Nv1, Ni},
) where {S, Nv1, Ni}
    # copy contiguous columns
    _, _, _, _, Nh = size(dest1)
    for (dest, bc) in fmbc.pairs
        @inbounds for h in 1:Nh, i in 1:Ni, v in 1:Nv1
            I = CartesianIndex(i, 1, 1, v, h)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(
    fmbc::FusedMultiBroadcast,
    dest1::VF{S1, Nv1},
) where {S1, Nv1}
    for (dest, bc) in fmbc.pairs
        @inbounds for v in 1:Nv1
            I = CartesianIndex(1, 1, 1, v, 1)
            bcI = isascalar(bc) ? bc[] : bc[I]
            dest[I] = convert(eltype(dest), bcI)
        end
    end
    return nothing
end

function fused_copyto!(fmbc::FusedMultiBroadcast, dest::DataF{S}) where {S}
    for (dest, bc) in fmbc.pairs
        @inbounds dest[] = convert(S, bc[])
    end
    return dest
end

# we've already diagonalized dest, so we only need to make
# sure that all the broadcast axes are compatible.
# Logic here is similar to Base.Broadcast.instantiate
@inline function _check_fused_broadcast_axes(bc1, bc2)
    axes = Base.Broadcast.combine_axes(bc1.args..., bc2.args...)
    if !(axes isa Nothing)
        Base.Broadcast.check_broadcast_axes(axes, bc1.args...)
        Base.Broadcast.check_broadcast_axes(axes, bc2.args...)
    end
end

@inline check_fused_broadcast_axes(fmbc::FusedMultiBroadcast) =
    check_fused_broadcast_axes(
        map(x -> x.second, fmbc.pairs),
        first(fmbc.pairs).second,
    )
@inline check_fused_broadcast_axes(bcs::Tuple{<:Any}, bc1) =
    _check_fused_broadcast_axes(first(bcs), bc1)
@inline check_fused_broadcast_axes(bcs::Tuple{}, bc1) = nothing
@inline function check_fused_broadcast_axes(bcs::Tuple, bc1)
    _check_fused_broadcast_axes(first(bcs), bc1)
    check_fused_broadcast_axes(Base.tail(bcs), bc1)
end
