# Broadcasting of AbstractData objects
# https://docs.julialang.org/en/v1/manual/interfaces/#Broadcast-Styles

abstract type DataStyle <: Base.BroadcastStyle end

abstract type DataColumnStyle <: DataStyle end
struct VFStyle{A} <: DataColumnStyle end
DataStyle(::Type{VF{S, A}}) where {S, A} = VFStyle{parent_array_type(A)}()
DataColumnStyle(::Type{VFStyle{A}}) where {A} = VFStyle{A}

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

abstract type Data1DXStyle{Ni} <: DataStyle end
struct VIFHStyle{Ni, A} <: Data1DXStyle{Ni} end
DataStyle(::Type{VIFH{S, Ni, A}}) where {S, Ni, A} =
    VIFHStyle{Ni, parent_array_type(A)}()
Data1DXStyle(::Type{VIFHStyle{Ni, A}}) where {Ni, A} = VIFHStyle{Ni, A}
DataColumnStyle(::Type{VIFHStyle{Ni, A}}) where {Ni, A} = VFStyle{A}
DataSlab1DStyle(::Type{VIFHStyle{Ni, A}}) where {Ni, A} = IFStyle{Ni, A}

abstract type Data2DXStyle{Nij} <: DataStyle end
struct VIJFHStyle{Nij, A} <: Data2DXStyle{Nij} end
DataStyle(::Type{VIJFH{S, Nij, A}}) where {S, Nij, A} =
    VIJFHStyle{Nij, parent_array_type(A)}()
Data2DXStyle(::Type{VIJFHStyle{Nij, A}}) where {Nij, A} = VIJFHStyle{Nij, A}
DataColumnStyle(::Type{VIJFHStyle{Nij, A}}) where {Nij, A} = VFStyle{A}
DataSlab2DStyle(::Type{VIJFHStyle{Nij, A}}) where {Nij, A} = IJFStyle{Nij, A}

abstract type Data3DStyle <: DataStyle end

Base.Broadcast.BroadcastStyle(::Type{D}) where {D <: AbstractData} =
    DataStyle(D)

# precedence rules
# scalars are broadcast over the data object
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    ds::DataStyle,
) = ds

Base.Broadcast.BroadcastStyle(::VFStyle{A}, ::IFHStyle{Ni, A}) where {Ni, A} =
    VIFHStyle{Ni, A}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{A},
    ::IJFHStyle{Nij, A},
) where {Nij, A} = VIJFHStyle{Nij, A}()

Base.Broadcast.BroadcastStyle(::VFStyle{A}, ::VIFHStyle{Ni, A}) where {Ni, A} =
    VIFHStyle{Ni, A}()

Base.Broadcast.BroadcastStyle(
    ::VFStyle{A},
    ::VIJFHStyle{Nij, A},
) where {Nij, A} = VIJFHStyle{Nij, A}()

Base.Broadcast.BroadcastStyle(
    ::IFHStyle{Ni, A},
    ::VIFHStyle{Ni, A},
) where {Ni, A} = VIFHStyle{Ni, A}()

Base.Broadcast.BroadcastStyle(
    ::IJFHStyle{Nij, A},
    ::VIJFHStyle{Nij, A},
) where {Nij, A} = VIJFHStyle{Nij, A}()

Base.Broadcast.broadcastable(data::AbstractData) = data

function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Ni, DS <: Union{Data1DStyle{Ni}, Data1DXStyle{Ni}}}
    args = map(arg -> slab(arg, inds...), bc.args)
    axes = (SOneTo(Ni),)
    Base.Broadcast.Broadcasted{DataSlab1DStyle(DS)}(bc.f, args, axes)
end

function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nij, DS <: Union{Data2DStyle{Nij}, Data2DXStyle{Nij}}}
    args = map(arg -> slab(arg, inds...), bc.args)
    axes = (SOneTo(Nij), SOneTo(Nij))
    Base.Broadcast.Broadcasted{DataSlab2DStyle(DS)}(bc.f, args, axes)
end

function column(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {N, DS <: Union{Data1DXStyle{N}, Data2DXStyle{N}}}
    args = map(arg -> column(arg, inds...), bc.args)
    axes = nothing
    Base.Broadcast.Broadcasted{DataColumnStyle(DS)}(bc.f, args, axes)
end

function column(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {DS <: DataColumnStyle}
    bc
end

function column(bc::Union{Data1D, Base.Broadcast.Broadcasted{<:Data1D}}, i, h)
    Ref(slab(bc, h)[i])
end

function Base.similar(
    bc::Union{IJFH{<:Any, Nij, A}, Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    _, _, _, _, Nh = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return IJFH{Eltype, Nij}(array)
end

function Base.similar(
    bc::Union{IFH{<:Any, Ni, A}, Broadcast.Broadcasted{IFHStyle{Ni, A}}},
    ::Type{Eltype},
) where {Ni, A, Eltype}
    _, _, _, _, Nh = size(bc)
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
) where {S, Ni, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    array = MArray{Tuple{Ni, Nf}, eltype(A), 2, Ni * Nf}(undef)
    return IF{Eltype, Ni}(array)
end

function Base.similar(
    bc::Union{VF{<:Any, A}, Broadcast.Broadcasted{VFStyle{A}}},
    ::Type{Eltype},
) where {A, Eltype}
    _, _, _, Nv, _ = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (Nv, typesize(eltype(A), Eltype)))
    return VF{Eltype}(array)
end

function Base.similar(
    bc::Union{VIFH{<:Any, Ni, A}, Broadcast.Broadcasted{VIFHStyle{Ni, A}}},
    ::Type{Eltype},
) where {Ni, A, Eltype}
    _, _, _, Nv, Nh = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (Nv, Ni, typesize(eltype(A), Eltype), Nh))
    return VIFH{Eltype, Ni}(array)
end

function Base.similar(
    bc::Union{VIJFH{<:Any, Nij, A}, Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    _, _, _, Nv, Nh = size(bc)
    PA = parent_array_type(A)
    array = similar(PA, (Nv, Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return VIJFH{Eltype, Nij}(array)
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{
        IJFH{<:Any, Nij, A},
        Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
    },
) where {F, Op, Nij, A}
    # mapreduce across slabs
    _, _, _, _, Nh = size(bc)
    mapreduce(op, 1:Nh) do h
        mapreduce(fn, op, slab(bc, h))
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{IFH{<:Any, Ni, A}, Base.Broadcast.Broadcasted{IFHStyle{Ni, A}}},
) where {F, Op, Ni, A}
    _, _, _, _, Nh = size(bc)
    mapreduce(op, 1:Nh) do h
        mapreduce(fn, op, slab(bc, h))
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IJF{S, Nij}) where {F, Op, S, Nij}
    mapreduce(op, Iterators.product(1:Nij, 1:Nij)) do (i, j)
        idx = CartesianIndex(i, j, 1, 1, 1)
        fn(bc[idx])
    end
end

function Base.mapreduce(fn::F, op::Op, bc::IF{S, Ni}) where {F, Op, S, Ni}
    mapreduce(op, 1:Ni) do i
        idx = CartesianIndex(i, 1, 1, 1, 1)
        fn(bc[idx])
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{VF{<:Any, A}, Base.Broadcast.Broadcasted{VFStyle{A}}},
) where {F, Op, A}
    _, _, _, Nv, _ = size(bc)
    mapreduce(op, 1:Nv) do v
        idx = CartesianIndex(1, 1, 1, v, 1)
        fn(bc[idx])
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{VIFH{<:Any, Ni, A}, Base.Broadcast.Broadcasted{VIFHStyle{Ni, A}}},
) where {F, Op, Ni, A}
    # mapreduce across columns
    _, _, _, _, Nh = size(bc)
    mapreduce(op, Iterators.product(1:Ni, 1:Nh)) do (i, h)
        mapreduce(fn, op, column(bc, i, h))
    end
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Union{
        VIJFH{<:Any, Nij, A},
        Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}},
    },
) where {F, Op, Nij, A}
    # mapreduce across columns
    _, _, _, _, Nh = size(bc)
    mapreduce(op, Iterators.product(1:Nij, 1:Nij, 1:Nh)) do (i, j, h)
        mapreduce(fn, op, column(bc, i, j, h))
    end
end

function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Union{IJFH{S, Nij}, Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
) where {S, Nij, A}
    _, _, _, _, Nh = size(bc)
    for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

function Base.copyto!(
    dest::IFH{S, Ni},
    bc::Union{IFH{S, Ni}, Base.Broadcast.Broadcasted{IFHStyle{Ni, A}}},
) where {S, Ni, A}
    _, _, _, _, Nh = size(bc)
    for h in 1:Nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

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

function Base.copyto!(
    dest::VF{S},
    bc::Union{VF{S, A}, Base.Broadcast.Broadcasted{VFStyle{A}}},
) where {S, A}
    _, _, _, Nv, _ = size(bc)
    @inbounds for v in 1:Nv
        idx = CartesianIndex(1, 1, 1, v, 1)
        dest[idx] = convert(S, bc[idx])
    end
    return dest
end

function Base.copyto!(
    dest::VIFH{S, Ni},
    bc::Union{VIFH{S, Ni, A}, Base.Broadcast.Broadcasted{VIFHStyle{Ni, A}}},
) where {S, Ni, A}
    # copy contiguous columns
    _, _, _, _, Nh = size(dest)
    @inbounds for h in 1:Nh, i in 1:Ni
        col_dest = column(dest, i, h)
        col_bc = column(bc, i, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

function Base.copyto!(
    dest::VIJFH{S, Nij},
    bc::Union{VIJFH{S, Nij, A}, Base.Broadcast.Broadcasted{VIJFHStyle{Nij, A}}},
) where {S, Nij, A}
    # copy contiguous columns
    _, _, _, _, Nh = size(dest)
    @inbounds for h in 1:Nh, j in 1:Nij, i in 1:Nij
        col_dest = column(dest, i, j, h)
        col_bc = column(bc, i, j, h)
        copyto!(col_dest, col_bc)
    end
    return dest
end

# broadcasting scalar assignment
function Base.Broadcast.materialize!(
    ::DS,
    dest,
    bc::Base.Broadcast.Broadcasted{Style},
) where {DS <: DataStyle, Style}
    return copyto!(
        dest,
        Base.Broadcast.instantiate(
            Base.Broadcast.Broadcasted{DS}(bc.f, bc.args, axes(dest)),
        ),
    )
end
