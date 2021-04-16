# Broadcasting of AbstractData objects
# https://docs.julialang.org/en/v1/manual/interfaces/#Broadcast-Styles

abstract type DataStyle <: Base.BroadcastStyle end

abstract type DataColumnStyle <: DataStyle end


abstract type DataSlabStyle{Nij} <: DataStyle end
struct IJFStyle{Nij, A} <: DataSlabStyle{Nij} end
DataStyle(::Type{IJF{S, Nij, A}}) where {S, Nij, A} = IJFStyle{Nij, A}()

abstract type Data2DStyle{Nij} <: DataStyle end
struct IJFHStyle{Nij, A} <: Data2DStyle{Nij} end
DataStyle(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} = IJFHStyle{Nij, A}()

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
    args = map(arg -> arg isa AbstractData ? slab(arg, inds...) : arg, bc.args)
    axes = (SOneTo(Nij), SOneTo(Nij))
    Base.Broadcast.Broadcasted{DataSlabStyle{DS}}(bc.f, args, axes)
end


# What should axes return?
# The Broadcast / Base machinery specializes on Tuple types and doesn't support NamedTuple
Base.axes(data::IJFH{S}) where {S} =
    (axes(parent(data), 1), axes(parent(data), 4))

function Base.similar(
    bc::Broadcast.Broadcasted{IJFHStyle{Nij, A}},
    ::Type{Eltype},
) where {Nij, A, Eltype}
    axes_ij, axes_h = axes(bc)
    array = similar(
        A,
        (axes_ij, axes_ij, Base.OneTo(typesize(eltype(A), Eltype)), axes_h),
    )
    return IJFH{Eltype, Nij}(array)
end

function Base.copyto!(
    dest::IJFH,
    bc::Base.Broadcast.Broadcasted{IJFHStyle{Nij, A}},
) where {Nij, A}
    _, nh = size(dest)
    for h in 1:nh
        slab_dest = slab(dest, h)
        slab_bc = slab(bc, h)
        @inbounds for j in 1:Nij, i in 1:Nij
            slab_dest[i, j] = slab_bc[i, j]
        end
    end
    return dest
end
