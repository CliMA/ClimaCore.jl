"""
    ClimaCore.DataLayouts

Notation:
- `i,j` are horizontal node indices within an element
- `k` is the vertical node index within an element
- `f` is the field index
- `v` is the vertical element index in a stack
- `h` is the element stack index

Data layout is specified by the order in which they appear, e.g. `IJKFVH`
indexes the underlying array as `[i,j,k,f,v,h]`

"""
module DataLayouts

import Requires
import StaticArrays: SOneTo, MArray

# TODO:
#  - doc strings for each type
#  - printing
#  - should some of these be subtypes of AbstractArray?

import ..slab, ..column
export slab, column, IJFH, IJF, VF

include("struct.jl")

abstract type AbstractData{S} end

"""
    DataColumn{S}

Abstract type for data storage for a column. Objects `data` should define a
`data[k,v]`, returning a value of type `S`.
"""
abstract type DataColumn{S} <: AbstractData{S} end

"""
    DataSlab1D{S,Ni}

Abstract type for data storage for a slab of `Ni` values of type `S`.
Objects `data` should define a `data[i]`, returning a value of type `S`.
"""
abstract type DataSlab1D{S, Nij} <: AbstractData{S} end

"""
    DataSlab2D{S,Nij}

Abstract type for data storage for a slab of `Nij × Nij` values of type `S`.
Objects `data` should define a `data[i,j]`, returning a value of type `S`.
"""
abstract type DataSlab2D{S, Nij} <: AbstractData{S} end

"""
    Data1D{S,Ni}

Abstract type for data storage for a 1D field made up of `Ni` values of type `S`.

Objects `data` should define `slab(data, h)` to return a `DataSlab2D{S,Nij}` object.
"""
abstract type Data1D{S, Ni} <: AbstractData{S} end

"""
    Data2D{S,Nij}

Abstract type for data storage for a 2D field made up of `Nij × Nij` values of type `S`.

Objects `data` should define `slab(data, h)` to return a `DataSlab2D{S,Nij}` object.
"""
abstract type Data2D{S, Nij} <: AbstractData{S} end


"""
    Data3D{S,Nij,Nk}

Abstract type for data storage for a 3D field made up of `Nij × Nij × Nk` values of type `S`.
"""
abstract type Data3D{S, Nij, Nk} <: AbstractData{S} end

Base.eltype(::AbstractData{S}) where {S} = S
Base.propertynames(::AbstractData{S}) where {S} = fieldnames(S)
Base.parent(data::AbstractData) = getfield(data, :array)

Base.similar(data::AbstractData{S}) where {S} = similar(data, S)

function Base.copyto!(dest::D, src::D) where {D <: AbstractData}
    copyto!(parent(dest), parent(src))
    return dest
end

# TODO: if this gets used inside kernels, move to a generated function?

@generated function _getproperty(data::AbstractData{S}, name::Val{Name}) where {S,Name}
    errorstring = "Invalid foo field name $(Name)"
    i = findfirst(isequal(Name), fieldnames(S))
    if i == nothing
        return :(error($errorstring))
    end
    return :(Base.@_inline_meta; getproperty(data, $i))
end

@inline function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
    #=
    if @generated 
    	errorstring = "Invalid field name $(name)"
    	i = findfirst(isequal(name), fieldnames(S))
	if i == nothing
	    return :(error($errorstring))
	else
	    return :(Base.@_inline_meta; getproperty(data, $i)) 
    	end
    else
    =#
    	i = findfirst(isequal(name), fieldnames(S))
    	i === nothing && error("Invalid field name $(name)")
    	return getproperty(data, i)
    #end
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


@inline function Base.getproperty(
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

rebuild(data::IJFH{S, Nij}, array) where {S, Nij} = IJFH{S, Nij}(array)
Base.copy(data::IJFH{S, Nij}) where {S, Nij} = IJFH{S, Nij}(copy(parent(data)))


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

Base.length(data::IJFH) = size(parent(data), 4)
Base.size(data::Data2D) = (length(data),)

@inline function Base.getproperty(data::IJFH{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJFH{SS, Nij}(view(array, :, :, (offset + 1):(offset + len), :))
end

struct IFH{S, Ni, A} <: Data1D{S, Ni}
    array::A
end

function IFH{S, Ni}(array::AbstractArray{T, 3}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    IFH{S, Ni, typeof(array)}(array)
end

rebuild(data::IFH{S, Ni}, array::AbstractArray{T, 3}) where {S, Ni, T} =
    IFH{S, Ni}(array)

Base.copy(data::IFH{S, Ni}) where {S, Ni} = IFH{S, Ni}(copy(parent(data)))

function IFH{S, Ni}(ArrayType, nelements) where {S, Ni}
    FT = eltype(ArrayType)
    IFH{S, Ni}(ArrayType(undef, Ni, typesize(FT, S), nelements))
end

Base.length(data::IFH) = size(parent(data), 3)

@inline function slab(data::IFH{S, Ni}, h::Integer) where {S, Ni}
    @boundscheck (1 <= h <= length(data)) || throw(BoundsError(data, (h,)))
    IF{S, Ni}(view(parent(data), :, :, h))
end

@inline function Base.getproperty(data::IFH{S, Ni}, f::Integer) where {S, Ni}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, f)
    offset = fieldtypeoffset(T, S, f)
    len = typesize(T, SS)
    IFH{SS, Ni}(view(array, :, (offset + 1):(offset + len), :))
end


"""
    IH1JH2{S, Nij}(data::AbstractMatrix{S})

Stores a 2D field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid.
"""
struct IH1JH2{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IH1JH2{S, Nij}(array::AbstractMatrix{S}) where {S, Nij}
    @assert size(array, 1) % Nij == 0
    @assert size(array, 2) % Nij == 0
    IH1JH2{S, Nij, typeof(array)}(array)
end

Base.length(data::IH1JH2{S, Nij}) where {S, Nij} =
    div(length(parent(data)), Nij * Nij)

function Base.similar(
    data::IH1JH2{S, Nij, A},
    ::Type{Eltype},
) where {S, Nij, A, Eltype}
    array = similar(A, Eltype)
    return IH1JH2{Eltype, Nij}(array)
end

Base.copy(data::IH1JH2{S, Nij}) where {S, Nij} =
    IH1JH2{S, Nij}(copy(parent(data)))

@inline function slab(data::IH1JH2{S, Nij}, h::Integer) where {S, Nij}
    N1, N2 = size(parent(data))
    n1 = div(N1, Nij)
    n2 = div(N2, Nij)
    z2, z1 = fldmod(h - 1, n1)

    @boundscheck (1 <= h <= n1 * n2) || throw(BoundsError(data, (h,)))

    return view(parent(data), Nij * z1 .+ (1:Nij), Nij * z2 .+ (1:Nij))
end



#=
struct KFV{S,A} <: DataColumn{S}
  array::A
end
function KFV{S}(array::AbstractArray{T,3}) where {S,T}
  KFV{S,typeof(array)}(array)
end
=#

struct IJF{S, Nij, A} <: DataSlab2D{S, Nij}
    array::A
end

function IJF{S, Nij}(array::AbstractArray{T, 3}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    IJF{S, Nij, typeof(array)}(array)
end

function Base.size(data::IJF{S, Nij}) where {S, Nij}
    return (Nij, Nij)
end

@inline function Base.getproperty(data::IJF{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJF{SS, Nij}(view(array, :, :, (offset + 1):(offset + len)))
end


# TODO: should this return a S or a 0-d box containing S?
#  - perhaps the latter, as then it is mutable?

# function column(ijfh::IJFH{S}, i::Integer, j::Integer, h) where {S}
#     get_struct(view(parent(ijfh), i, j, :, h), S)
# end

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

struct IF{S, Ni, A} <: DataSlab1D{S, Ni}
    array::A
end

function IF{S, Ni}(array::AbstractArray{T, 2}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    IF{S, Ni, typeof(array)}(array)
end

function Base.size(::IF{S, Ni}) where {S, Ni}
    return (Ni,)
end

@inline function Base.getproperty(data::IF{S, Ni}, f::Integer) where {S, Ni}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, f)
    offset = fieldtypeoffset(T, S, f)
    len = typesize(T, SS)
    IF{SS, Ni}(view(array, :, (offset + 1):(offset + len)))
end

@inline function Base.getindex(data::IF{S, Ni}, i::Integer) where {S, Ni}
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    @inbounds get_struct(view(parent(data), i, :), S)
end

@inline function Base.setindex!(data::IF{S, Ni}, val, i::Integer) where {S, Ni}
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    set_struct!(view(parent(data), i, :), val)
end

# TODO: should this return a S or a 0-d box containing S?
#  - perhaps the latter, as then it is mutable?

function column(ijfh::IJFH{S}, i::Integer, j::Integer, h) where {S}
    get_struct(view(parent(ijfh), i, j, :, h), S)
end

@inline function slab(ijfh::IJFH{S, Nij}, h::Integer) where {S, Nij} # k,v are unused
    @boundscheck (1 <= h <= length(ijfh)) || throw(BoundsError(ijfh, (h,)))
    IJF{S, Nij}(view(parent(ijfh), :, :, :, h))
end

@propagate_inbounds function Base.getindex(
    slab::DataSlab2D{S},
    I::CartesianIndex{2},
) where {S}
    slab[I[1], I[2]]
end

Base.size(::DataSlab2D{S, Nij}) where {S, Nij} = (Nij, Nij)
Base.axes(::DataSlab2D{S, Nij}) where {S, Nij} = (SOneTo(Nij), SOneTo(Nij))

# Data column
struct VF{S, A} <: DataColumn{S}
    array::A
end

function VF{S}(array::AbstractArray{T, 2}) where {S, T}
    VF{S, typeof(array)}(array)
end

function VF{S}(array::AbstractVector{T}) where {S, T}
    @assert typesize(T, S) == 1
    VF{S}(reshape(array, (:, 1)))
end

function VF{S}(ArrayType, nelements) where {S}
    FT = eltype(ArrayType)
    VF{S}(ArrayType(undef, nelements, typesize(FT, S)))
end

Base.length(data::VF) = size(parent(data), 1)
Base.size(data::VF) = (length(data),)
Base.copy(data::VF{S}) where {S} = VF{S}(copy(parent(data)))
Base.lastindex(data::VF) = length(data)

@inline function Base.getproperty(data::VF{S}, i::Integer) where {S}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    VF{SS}(view(array, :, (offset + 1):(offset + len)))
end

@propagate_inbounds function Base.getindex(data::VF{S}, i::Integer) where {S}
    get_struct(view(parent(data), i, :), S)
end

@propagate_inbounds function Base.getindex(data::VF{S}, I::CartesianIndex{1}) where {S}
    getindex(data, I[1])
end

@inline function Base.setindex!(data::VF{S}, val, v::Integer) where {S}
    set_struct!(view(parent(data), v, :), val)
end

# broadcast machinery
include("broadcast.jl")

# GPU method specializations
include("cuda.jl")

end # module
