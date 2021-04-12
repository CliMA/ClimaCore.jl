"""
    ClimateMachineCore.DataLayouts

Notation:
- `i,j` are horizontal node indices within an element
- `k` is the vertical node index within an element
- `v` is the vertical element index in a stack
- `h` is the element stack index
- `f` is the field index

Data layout is specified by the order in which they appear, e.g. `IJKFVH` indexes the underlying array as `[i,j,k,f,v,h]`

"""
module DataLayouts

import Adapt

# TODO:
#  - doc strings for each type
#  - printing
#  - should some of these be subtypes of AbstractArray?


export pancake, column, IJFH, IJF

include("struct.jl")

abstract type AbstractData{S} end

"""
    DataColumn{S}

Abstract type for data storage for a column. Objects `data` should define a
`data[k,v]`, returning a value of type `S`.
"""
abstract type DataColumn{S} <: AbstractData{S} end

"""
    DataPancake{S}

Abstract type for data storage for a pancake. Objects `data` should define a
`data[i,j]`, returning a value of type `S`.
"""
abstract type DataPancake{S} <: AbstractData{S} end
abstract type Data2D{S} <: AbstractData{S} end
abstract type Data3D{S} <: AbstractData{S} end

# TODO: if this gets used inside kernels, move to a generated function?
function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
    i = findfirst(fieldnames(S), name)
    i === nothing && error("Invalid field name")
    return getproperty(data, i)
end




struct IJKFVH{S, A} <: Data3D{S}
    array::A
end
Adapt.adapt_structure(to, obj::IJKFVH{S}) where {S} = IJKFVH{S}(getfield(obj, :array))

function IJKFVH{S}(array::AbstractArray{T, 6}) where {S, T}
    IJKFVH{S, typeof(array)}(array)
end
function Base.getproperty(data::IJKFVH{S}, i::Integer) where {S}
    array = getfield(data, :array)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJKFVH{SS}(view(array, :, :, :, (offset + 1):(offset + len), :, :))
end


struct IJFH{S, A} <: Data2D{S}
    array::A
end
Adapt.adapt_structure(to, obj::IJFH{S}) where {S} = IJFH{S}(getfield(obj, :array))

function IJFH{S}(array::AbstractArray{T, 4}) where {S, T}
    IJFH{S, typeof(array)}(array)
end
function Base.getproperty(data::IJFH{S}, i::Integer) where {S}
    array = getfield(data, :array)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJFH{SS}(view(array, :, :, (offset + 1):(offset + len), :))
end



#=
struct KFV{S,A} <: DataColumn{S}
  array::A
end
function KFV{S}(array::AbstractArray{T,3}) where {S,T}
  KFV{S,typeof(array)}(array)
end
=#

struct IJF{S, A} <: DataPancake{S}
    array::A
end
Adapt.adapt_structure(to, obj::IJF{S}) where {S} = IJF{S}(getfield(obj, :array))

function IJF{S}(array::AbstractArray{T, 3}) where {S, T}
    IJF{S, typeof(array)}(array)
end

function Base.getproperty(data::IJF{S}, i::Integer) where {S}
    array = getfield(data, :array)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    len = typesize(T, SS)
    IJF{SS}(view(array, :, :, (offset + 1):(offset + len)))
end


# TODO: should this return a S or a 0-d box containing S?
#  - perhaps the latter, as then it is mutable?
function column(ijfh::IJFH{S}, i, j, h) where {S}
    get_struct(view(getfield(ijfh, :array), i, j, :, h), S)
end

function pancake(ijfh::IJFH{S}, k, v, h) where {S} # k,v are unused
    IJF{S}(view(getfield(ijfh, :array), :, :, :, h))
end


function Base.getindex(ijf::IJF{S}, i, j) where {S}
    get_struct(view(getfield(ijf, :array), i, j, :), S)
end
function Base.setindex!(ijf::IJF{S}, val, i, j) where {S}
    set_struct!(view(getfield(ijf, :array), i, j, :), val)
end

end # module
