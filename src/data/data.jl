"""
    ClimateMachineCore.DataLayouts

Notation:
- `i,j` are horizontal node indices within an element
- `k` is the vertical node index within an element
- `v` is the vertical element index in a stack
- `h` is the element stack index
- `f` is the field index

Data layout is specified by the order in which they appear, e.g. `IJKFVH`
indexes the underlying array as `[i,j,k,f,v,h]`

"""
module DataLayouts

import Adapt
import StaticArrays: SOneTo

# TODO:
#  - doc strings for each type
#  - printing
#  - should some of these be subtypes of AbstractArray?

import ..slab, ..column
export slab, column, IJFH, IJF

include("struct.jl")

abstract type AbstractData{S} end

Base.eltype(::AbstractData{S}) where {S} = S

"""
    DataColumn{S}

Abstract type for data storage for a column. Objects `data` should define a
`data[k,v]`, returning a value of type `S`.
"""
abstract type DataColumn{S} <: AbstractData{S} end

"""
    DataSlab{S,Nij}

Abstract type for data storage for a slab of `Nij × Nij` values of type `S`.
Objects `data` should define a `data[i,j]`, returning a value of type `S`.
"""
abstract type DataSlab{S, Nij} <: AbstractData{S} end

"""
    Data2D{S,Nij}

Abstract type for data storage for a 2D field made up of `Nij × Nij` values of type `S`.

Objects `data` should define `slab(data, h)` to return a `DataSlab{S,Nij}` object.
"""
abstract type Data2D{S, Nij} <: AbstractData{S} end


"""
    Data3D{S,Nij,Nk}

Abstract type for data storage for a 3D field made up of `Nij × Nij × Nk` values of type `S`.
"""
abstract type Data3D{S, Nij, Nk} <: AbstractData{S} end


Base.propertynames(data::AbstractData{S}) where {S} = fieldnames(S)
Base.parent(data::AbstractData) = getfield(data, :array)

# TODO: if this gets used inside kernels, move to a generated function?
function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
    i = findfirst(isequal(name), fieldnames(S))
    i === nothing && error("Invalid field name")
    return getproperty(data, i)
end


struct IJKFVH{S, Nij, Nk, A} <: Data3D{S, Nij, Nk}
    array::A
end
function IJKFVH{S, Nij, Nk}(array::AbstractArray{T, 6}) where {S, Nij, Nk, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == Nk
    IJKFVH{S, Nij, Nk, typeof(array)}(array)
end

Adapt.adapt_structure(to, data::IJKFVH{S, Nij, Nk}) where {S, Nij, Nk} =
    IJKFVH{S, Nij, Nk}(Adapt.adapt(to, getfield(data, :array)))

function Base.getproperty(
    data::IJKFVH{S, Nij, Nk},
    i::Integer,
) where {S, Nij, Nk}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJKFVH{SS, Nij, Nk}(view(array, :, :, :, (offset + 1):(offset + len), :, :))
end


struct IJFH{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IJFH{S, Nij}(array::AbstractArray{T, 4}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    IJFH{S, Nij, typeof(array)}(array)
end


"""
    IJFH{S,Nij}(ArrayType, nelements)

Construct an IJFH structure given the backing `ArrayType`,
quadrature degrees of freedom `Nij`,
and the number of mesh elements `nelements`.
"""
function IJFH{S, Nij}(ArrayType, nelements) where {S, Nij}
    FT = eltype(ArrayType)
    IJFH{S, Nij}(ArrayType(undef, Nij, Nij, typesize(FT, S), nelements))
end

Adapt.adapt_structure(to, data::IJFH{S, Nij}) where {S, Nij} =
    IJFH{S, Nij}(Adapt.adapt(to, parent(data)))


function Base.size(data::IJFH)
    array = parent(data)
    size(array, 1), size(array, 4)
end

function Base.getproperty(data::IJFH{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJFH{SS, Nij}(view(array, :, :, (offset + 1):(offset + len), :))
end



#=
struct KFV{S,A} <: DataColumn{S}
  array::A
end
function KFV{S}(array::AbstractArray{T,3}) where {S,T}
  KFV{S,typeof(array)}(array)
end
=#

struct IJF{S, Nij, A} <: DataSlab{S, Nij}
    array::A
end
function IJF{S, Nij}(array::AbstractArray{T, 3}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    IJF{S, Nij, typeof(array)}(array)
end

Adapt.adapt_structure(to, data::IJF{S, Nij}) where {S, Nij} =
    IJF{S, Nij}(Adapt.adapt(to, parent(data)))

function Base.size(data::IJF{S, Nij}) where {S, Nij}
    return (Nij, Nij)
end

function Base.getproperty(data::IJF{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJF{SS, Nij}(view(array, :, :, (offset + 1):(offset + len)))
end


# TODO: should this return a S or a 0-d box containing S?
#  - perhaps the latter, as then it is mutable?

function column(ijfh::IJFH{S}, i::Integer, j::Integer, h) where {S}
    get_struct(view(parent(ijfh), i, j, :, h), S)
end

@inline function slab(ijfh::IJFH{S, Nij}, h) where {S, Nij} # k,v are unused
    @boundscheck (1 <= h <= size(ijfh)[2]) || throw(BoundsError(ijfh, (h,)))
    IJF{S, Nij}(view(parent(ijfh), :, :, :, h))
end

@inline function Base.getindex(
    ijf::IJF{S, Nij},
    i::Integer,
    j::Integer,
) where {S, Nij}
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(ijf, (i, j)))
    @inbounds get_struct(view(parent(ijf), i, j, :), S)
end

@inline function Base.setindex!(
    ijf::IJF{S, Nij},
    val,
    i::Integer,
    j::Integer,
) where {S, Nij}
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(ijf, (i, j)))
    set_struct!(view(parent(ijf), i, j, :), val)
end

@propagate_inbounds function Base.getindex(
    slab::DataSlab{S},
    I::CartesianIndex{2},
) where {S}
    slab[I[1], I[2]]
end
Base.size(slab::DataSlab{S, Nij}) where {S, Nij} = (Nij, Nij)
Base.axes(slab::DataSlab{S, Nij}) where {S, Nij} = (SOneTo(Nij), SOneTo(Nij))



include("broadcast.jl")
include("cuda.jl")

end # module
