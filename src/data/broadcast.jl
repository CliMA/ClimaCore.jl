# Broadcasting of AbstractData objects
# https://docs.julialang.org/en/v1/manual/interfaces/#Broadcast-Styles

abstract type DataStyle <: Base.BroadcastStyle end

abstract type DataColumnStyle <: DataStyle end


abstract type DataSlabStyle{Nij} <: DataStyle end

# determine the parent type underlying any SubArrays
parent_array_type(::Type{A}) where {A <: AbstractArray} = A
parent_array_type(::Type{S}) where {S <: SubArray{T, N, A}} where {T, N, A} =
    parent_array_type(A)

struct IJFStyle{Nij, A} <: DataSlabStyle{Nij} end
DataStyle(::Type{IJF{S, Nij, A}}) where {S, Nij, A} =
    IJFStyle{Nij, parent_array_type(A)}()

abstract type Data2DStyle{Nij} <: DataStyle end
struct IJFHStyle{Nij, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} =
    IJFHStyle{Nij, parent_array_type(A)}()
DataSlabStyle(::Type{IJFHStyle{Nij, A}}) where {Nij, A} = IJFStyle{Nij, A}


abstract type Data3DStyle <: DataStyle end


Base.Broadcast.BroadcastStyle(::Type{D}) where {D <: AbstractData} =
    DataStyle(D)

# precedence rules
# scalars are broadcast over the data object
Base.Broadcast.BroadcastStyle(
    a::Base.Broadcast.AbstractArrayStyle{0},
    b::DataStyle,
) = b


Base.Broadcast.broadcastable(data::AbstractData) = data


function slab(
    bc::Base.Broadcast.Broadcasted{DS},
    inds...,
) where {Nij, DS <: Data2DStyle{Nij}}
    args = map(arg -> slab(arg, inds...), bc.args)
    axes = (SOneTo(Nij), SOneTo(Nij))
    Base.Broadcast.Broadcasted{DataSlabStyle(DS)}(bc.f, args, axes)
end

function Base.similar(
    bc::Union{IJFH{<:Any, Nij, A}, Broadcast.Broadcasted{IJFHStyle{Nij, A}}},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    Nh = length(bc)
    array = similar(A, (Nij, Nij, typesize(eltype(A), Eltype), Nh))
    return IJFH{Eltype, Nij}(array)
end
function Base.similar(
    data::Union{IJF{<:Any, Nij, A}, Broadcast.Broadcasted{IJFStyle{Nij, A}}},
    ::Type{Eltype},
) where {S, Nij, A, Eltype}
    Nf = typesize(eltype(A), Eltype)
    #array = similar(A, (Nij, Nij, typesize(eltype(A), Eltype)))
    array = MArray{Tuple{Nij, Nij, Nf}, eltype(A), 3, Nij * Nij * Nf}(undef)
    return IJF{Eltype, Nij}(array)
end

function Base.mapreduce(
    fn::F,
    op::Op,
    bc::Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
) where {F, Op, Nij, A}
    mapreduce(op, 1:length(bc)) do h
        mapreduce(fn, op, slab(bc, h))
    end
end
function Base.mapreduce(fn::F, op::Op, bc::IJFH) where {F, Op}
    mapreduce(op, 1:length(bc)) do h
        mapreduce(fn, op, slab(bc, h))
    end
end
function Base.mapreduce(
    fn::F,
    op::Op,
    slab_bc::IJF{S, Nij},
) where {F, Op, S, Nij}
    mapreduce(op, Iterators.product(1:Nij, 1:Nij)) do (i, j)
        fn(slab_bc[i, j])
    end
end



function Base.copyto!(
    dest::IJFH{S, Nij},
    bc::Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
) where {S, Nij, A}
    nh = length(dest)
    for h in 1:nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        copyto!(slab_dest, slab_bc)
    end
    return dest
end

function Base.copyto!(
    dest::IJF{S, Nij},
    bc::Base.Broadcast.Broadcasted{IJFStyle{Nij, A}},
) where {S, Nij, A}
    @inbounds for j in 1:Nij, i in 1:Nij
        dest[i, j] = convert(S, bc[i, j])
    end
    return dest
end



# broadcasting scalar assignment
@inline function Base.Broadcast.materialize!(
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
