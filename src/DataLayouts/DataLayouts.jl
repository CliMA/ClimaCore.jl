"""
    ClimaCore.DataLayouts

Notation:
- `i,j` are horizontal node indices within an element
- `k` is the vertical node index within an element
- `f` is the field index (1 if field is scalar, >1 if it is a vector field)
- `v` is the vertical element index in a stack
- `h` is the element stack index

Data layout is specified by the order in which they appear, e.g. `IJKFVH`
indexes the underlying array as `[i,j,k,f,v,h]`

"""
module DataLayouts

import Base: Base, @propagate_inbounds
import StaticArrays: SOneTo, MArray, SArray
import ClimaComms

import ..enable_threading, ..slab, ..slab_args, ..column, ..column_args, ..level
export slab, column, level, IJFH, IJF, IFH, IF, VF, VIJFH, VIFH, DataF

include("struct.jl")

abstract type AbstractData{S} end

Base.size(data::AbstractData, i::Integer) = size(data)[i]

function Base.show(io::IO, data::AbstractData)
    indent_width = 2
    (rows, cols) = displaysize(io)
    println(io, summary(data))
    print(io, " "^indent_width)
    print(
        IOContext(
            io,
            :compact => true,
            :limit => true,
            :displaysize => (rows, cols - indent_width),
        ),
        vec(parent(data)),
    )
    return io
end

"""
    Data0D{S}
"""
abstract type Data0D{S} <: AbstractData{S} end

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
    Data1DX{S,Ni}

Abstract type for data storage for a 1D field with extruded columns.
The horizontal is made up of `Ni` values of type `S`.

Objects `data` should define `slab(data, v, h)` to return a
`DataSlab1D{S,Ni}` object, and a `column(data, i, h)` to return a `DataColumn`.
"""
abstract type Data1DX{S, Ni} <: AbstractData{S} end

"""
    Data2DX{S,Nij}

Abstract type for data storage for a 2D field with extruded columns.
The horizontal is made  is made up of `Nij × Nij` values of type `S`.


Objects `data` should define `slab(data, v, h)` to return a
`DataSlab2D{S,Nij}` object, and a `column(data, i, j, h)` to return a `DataColumn`.
"""
abstract type Data2DX{S, Nij} <: AbstractData{S} end

"""
    Data3D{S,Nij,Nk}

Abstract type for data storage for a 3D field made up of `Nij × Nij × Nk` values of type `S`.
"""
abstract type Data3D{S, Nij, Nk} <: AbstractData{S} end

# Generic AbstractData methods

Base.eltype(::Type{<:AbstractData{S}}) where {S} = S
@inline function Base.propertynames(::AbstractData{S}) where {S}
    filter(name -> sizeof(fieldtype(S, name)) > 0, fieldnames(S))
end

Base.parent(data::AbstractData) = getfield(data, :array)

Base.similar(data::AbstractData{S}) where {S} = similar(data, S)

function Base.copyto!(dest::D, src::D) where {D <: AbstractData}
    copyto!(parent(dest), parent(src))
    return dest
end

@inline function ncomponents(data::AbstractData{S}) where {S}
    typesize(eltype(parent(data)), S)
end

@generated function _getproperty(
    data::AbstractData{S},
    ::Val{Name},
) where {S, Name}
    errorstring = "Invalid field name $(Name) for type $(S)"
    i = findfirst(isequal(Name), fieldnames(S))
    if i === nothing
        return :(error($errorstring))
    end
    static_idx = Val{i}()
    return :(Base.@_inline_meta; DataLayouts._property_view(data, $static_idx))
end

@inline function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
    _getproperty(data, Val{name}())
end
@inline function Base.dotgetproperty(
    data::AbstractData{S},
    name::Symbol,
) where {S}
    _getproperty(data, Val{name}())
end

#= Generic function impl fallback
@noinline function error_invalid_fieldname(@nospecialize(S::Type), name::Symbol)
    error("Invalid field name $(name) for type $(S)")
end

@inline function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
   i = findfirst(isequal(name), fieldnames(S))
   i === nothing && error_invalid_fieldname(S, name)
   getproperty(data, i)
end
=#

# ==================
# Data3D DataLayout
# ==================

struct IJKFVH{S, Nij, Nk, A} <: Data3D{S, Nij, Nk}
    array::A
end

function IJKFVH{S, Nij, Nk}(array::AbstractArray{T, 6}) where {S, Nij, Nk, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == Nk
    check_basetype(T, S)
    IJKFVH{S, Nij, Nk, typeof(array)}(array)
end

function replace_basetype(
    data::IJKFVH{S, Nij, Nk},
    ::Type{T},
) where {S, Nij, Nk, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return IJKFVH{S′, Nij, Nk}(similar(array, T))
end

@generated function _property_view(
    data::IJKFVH{S, Nij, Nk, A},
    ::Val{Idx},
) where {S, Nij, Nk, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(IJKFVH{$SS, $Nij, $Nk}(
        @inbounds view(parent(data), :, :, :, $field_byterange, :, :)
    ))
end

@inline function Base.getproperty(
    data::IJKFVH{S, Nij, Nk},
    i::Integer,
) where {S, Nij, Nk}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview =
        @inbounds view(array, :, :, :, (offset + 1):(offset + nbytes), :, :)
    IJKFVH{SS, Nij, Nk}(dataview)
end

function Base.size(data::IJKFVH{S, Nij, Nk}) where {S, Nij, Nk}
    Nv = size(parent(data), 5)
    Nh = size(parent(data), 6)
    return (Nij, Nij, Nk, Nv, Nh)
end

# ==================
# Data2D DataLayout
# ==================

"""
    IJFH{S, Nij, A} <: Data2D{S, Nij}

Backing `DataLayout` for 2D spectral element slabs.

Element nodal point (I,J) data is contiguous for each datatype `S` struct field (F),
for each 2D mesh element slab (H).
"""
struct IJFH{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IJFH{S, Nij}(array::AbstractArray{T, 4}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    check_basetype(T, S)
    IJFH{S, Nij, typeof(array)}(array)
end

rebuild(data::IJFH{S, Nij}, array::A) where {S, Nij, A <: AbstractArray} =
    IJFH{S, Nij}(array)

Base.copy(data::IJFH{S, Nij}) where {S, Nij} = IJFH{S, Nij}(copy(parent(data)))

function Base.size(data::IJFH{S, Nij}) where {S, Nij}
    Nv = 1
    Nh = size(parent(data), 4)
    (Nij, Nij, 1, Nv, Nh)
end

function Base.fill!(data::IJFH, val)
    (_, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh
        fill!(slab(data, h), val)
    end
    return data
end


"""
    IJFH{S,Nij}(ArrayType, nelements)

Construct a IJFH 2D Spectral DataLayout given the backing `ArrayType`,
quadrature degrees of freedom `Nij × Nij`  ,
and the number of mesh elements `nelements`.
"""
function IJFH{S, Nij}(ArrayType, nelements) where {S, Nij}
    T = eltype(ArrayType)
    IJFH{S, Nij}(ArrayType(undef, Nij, Nij, typesize(T, S), nelements))
end

function replace_basetype(data::IJFH{S, Nij}, ::Type{T}) where {S, Nij, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return IJFH{S′, Nij}(similar(array, T))
end
@propagate_inbounds function Base.getindex(
    data::IJFH{S},
    I::CartesianIndex{5},
) where {S}
    i, j, _, _, h = I.I
    @inbounds get_struct(parent(data), S, Val(3), CartesianIndex(i, j, 1, h))
end
@propagate_inbounds function Base.setindex!(
    data::IJFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    i, j, _, _, h = I.I
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(i, j, 1, h),
    )
end

@inline function Base.getindex(data::IJFH{S}, i, j, _, _, h) where {S}
    @inbounds get_struct(parent(data), S, Val(3), CartesianIndex(i, j, 1, h))
end
@inline function Base.setindex!(data::IJFH{S}, val, i, j, _, _, h) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(i, j, 1, h),
    )
end


Base.length(data::IJFH) = size(parent(data), 4)

@generated function _property_view(
    data::IJFH{S, Nij, A},
    ::Val{Idx},
) where {S, Nij, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(IJFH{$SS, $Nij}(
        @inbounds view(parent(data), :, :, $field_byterange, :)
    ))
end

@inline function Base.getproperty(data::IJFH{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, :, (offset + 1):(offset + nbytes), :)
    IJFH{SS, Nij}(dataview)
end

@inline function slab(data::IJFH{S, Nij}, h::Integer) where {S, Nij}
    @boundscheck (1 <= h <= length(data)) || throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, :, :, h)
    IJF{S, Nij}(dataview)
end

@inline function slab(data::IJFH{S, Nij}, v::Integer, h::Integer) where {S, Nij}
    @boundscheck (v >= 1 && 1 <= h <= length(data)) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), :, :, :, h)
    IJF{S, Nij}(dataview)
end

@inline function column(data::IJFH{S, Nij}, i, j, h) where {S, Nij}
    @boundscheck (1 <= j <= Nij && 1 <= i <= Nij && 1 <= h <= length(data)) ||
                 throw(BoundsError(data, (i, j, h)))
    dataview = @inbounds view(parent(data), i, j, :, h)
    DataF{S}(dataview)
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::IJFH{S, Nij},
) where {S, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        IJFH{S, Nij}(gatherdata)
    else
        nothing
    end
end

# ==================
# Data1D DataLayout
# ==================

Base.length(data::Data1D) = size(parent(data), 3)

"""
    IFH{S, Ni, A} <: Data1D{S, Ni}

Backing `DataLayout` for 1D spectral element slabs.

Element nodal point (I) data is contiguous for each datatype `S` struct field (F),
for each 1D mesh element (H).
"""
struct IFH{S, Ni, A} <: Data1D{S, Ni}
    array::A
end

function IFH{S, Ni}(array::AbstractArray{T, 3}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    check_basetype(T, S)
    IFH{S, Ni, typeof(array)}(array)
end

function replace_basetype(data::IFH{S, Ni}, ::Type{T}) where {S, Ni, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return IFH{S′, Ni}(similar(array, T))
end

"""
    IFH{S,Ni}(ArrayType, nelements)

Construct a IFH 1D Spectral DataLayout given the backing `ArrayType`,
quadrature degrees of freedom `Ni`  ,
and the number of mesh elements `nelements`.
"""
function IFH{S, Ni}(ArrayType, nelements) where {S, Ni}
    T = eltype(ArrayType)
    IFH{S, Ni}(ArrayType(undef, Ni, typesize(T, S), nelements))
end

rebuild(data::IFH{S, Ni}, array::AbstractArray{T, 3}) where {S, Ni, T} =
    IFH{S, Ni}(array)

Base.copy(data::IFH{S, Ni}) where {S, Ni} = IFH{S, Ni}(copy(parent(data)))

function Base.size(data::IFH{S, Ni}) where {S, Ni}
    Nv = 1
    Nh = size(parent(data), 3)
    (Ni, 1, 1, Nv, Nh)
end

function Base.fill!(data::IFH, val)
    (_, _, _, _, Nh) = size(data)
    @inbounds for h in 1:Nh
        fill!(slab(data, h), val)
    end
    return data
end

@inline function slab(data::IFH{S, Ni}, h::Integer) where {S, Ni}
    @boundscheck (1 <= h <= length(data)) || throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, :, h)
    IF{S, Ni}(dataview)
end
Base.@propagate_inbounds slab(data::IFH, v::Integer, h::Integer) = slab(data, h)

@inline function column(data::IFH{S, Ni}, i, h) where {S, Ni}
    @boundscheck (1 <= h <= length(data) && 1 <= i <= Ni) ||
                 throw(BoundsError(data, (i, h)))
    dataview = @inbounds view(parent(data), i, :, h)
    DataF{S}(dataview)
end
Base.@propagate_inbounds column(data::IFH{S, Ni}, i, j, h) where {S, Ni} =
    column(data, i, h)

@generated function _property_view(
    data::IFH{S, Ni, A},
    ::Val{Idx},
) where {S, Ni, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(IFH{$SS, $Ni}(
        @inbounds view(parent(data), :, $field_byterange, :)
    ))
end

@inline function Base.getproperty(data::IFH{S, Ni}, f::Integer) where {S, Ni}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, f)
    offset = fieldtypeoffset(T, S, f)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, (offset + 1):(offset + nbytes), :)
    IFH{SS, Ni}(dataview)
end

@inline function Base.getindex(data::IFH{S}, i, _, _, _, h) where {S}
    @inbounds get_struct(parent(data), S, Val(2), CartesianIndex(i, 1, h))
end
@inline function Base.getindex(data::IFH{S}, I::CartesianIndex{5}) where {S}
    i, _, _, _, h = I.I
    @inbounds get_struct(parent(data), S, Val(2), CartesianIndex(i, 1, h))
end
@inline function Base.setindex!(data::IFH{S}, val, i, _, _, _, h) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(i, 1, h),
    )
end
@inline function Base.setindex!(
    data::IFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    i, _, _, _, h = I.I
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(i, 1, h),
    )
end

# ======================
# Data0D DataLayout
# ======================

Base.length(data::Data0D) = 1
Base.size(data::Data0D) = (1, 1, 1, 1, 1)

"""
    DataF{S, A} <: Data0D{S}

Backing `DataLayout` for 0D point data.
"""
struct DataF{S, A} <: Data0D{S}
    array::A
end

function DataF{S}(array::AbstractVector{T}) where {S, T}
    check_basetype(T, S)
    DataF{S, typeof(array)}(array)
end

function DataF{S}(ArrayType) where {S}
    T = eltype(ArrayType)
    DataF{S}(ArrayType(undef, typesize(T, S)))
end

function DataF(x::T) where {T}
    if is_valid_basetype(Float64, T)
        d = DataF{T}(Array{Float64})
        d[] = x
        return d
    elseif is_valid_basetype(Float32, T)
        d = DataF{T}(Array{Float32})
        d[] = x
        return d
    else
        check_basetype(Float64, T)
    end
end


function Base.fill!(data::DataF, val)
    @inbounds data[] = val
    return data
end
@inline function Base.getproperty(data::DataF{S}, i::Integer) where {S}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, (offset + 1):(offset + nbytes))
    DataF{SS}(dataview)
end

@generated function _property_view(
    data::DataF{S, A},
    ::Val{Idx},
) where {S, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(DataF{$SS}(@inbounds view(parent(data), $field_byterange)))
end

Base.@propagate_inbounds function Base.getindex(data::DataF{S}) where {S}
    @inbounds get_struct(parent(data), S, Val(1), CartesianIndex(1))
end

@propagate_inbounds function Base.getindex(col::Data0D, I::CartesianIndex{5})
    @inbounds col[]
end

Base.@propagate_inbounds function Base.setindex!(data::DataF{S}, val) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(1),
        CartesianIndex(1),
    )
end

@propagate_inbounds function Base.setindex!(
    col::Data0D,
    val,
    I::CartesianIndex{5},
)
    @inbounds col[] = val
end

Base.copy(data::DataF{S}) where {S} = DataF{S}(copy(parent(data)))

# ======================
# DataSlab2D DataLayout
# ======================

@propagate_inbounds function Base.getindex(
    slab::DataSlab2D{S},
    I::CartesianIndex,
) where {S}
    @inbounds slab[I[1], I[2]]
end

@propagate_inbounds function Base.setindex!(
    slab::DataSlab2D{S},
    val,
    I::CartesianIndex,
) where {S}
    @inbounds slab[I[1], I[2]] = val
end

Base.size(::DataSlab2D{S, Nij}) where {S, Nij} = (Nij, Nij, 1, 1, 1)
Base.axes(::DataSlab2D{S, Nij}) where {S, Nij} = (SOneTo(Nij), SOneTo(Nij))

@inline function slab(data::DataSlab2D, h)
    @boundscheck (h >= 1) || throw(BoundsError(data, (h,)))
    data
end

@inline function slab(data::DataSlab2D, v, h)
    @boundscheck (v >= 1 && h >= 1) || throw(BoundsError(data, (v, h)))
    data
end

"""
    IJF{S, Nij, A} <: DataSlab2D{S, Nij}

Backing `DataLayout` for 2D spectral element slab data.

Nodal element data (I,J) are contiguous for each `S` datatype struct field (F) for a single element slab.

A `DataSlab2D` view can be returned from other `Data2D` objects by calling `slab(data, idx...)`.
"""
struct IJF{S, Nij, A} <: DataSlab2D{S, Nij}
    array::A
end

function IJF{S, Nij}(array::AbstractArray{T, 3}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    check_basetype(T, S)
    IJF{S, Nij, typeof(array)}(array)
end

rebuild(data::IJF{S, Nij}, array::A) where {S, Nij, A <: AbstractArray} =
    IJF{S, Nij}(array)
function IJF{S, Nij}(::Type{MArray}, ::Type{T}) where {S, Nij, T}
    Nf = typesize(T, S)
    array = MArray{Tuple{Nij, Nij, Nf}, T, 3, Nij * Nij * Nf}(undef)
    IJF{S, Nij}(array)
end
function SArray(ijf::IJF{S, Nij, <:MArray}) where {S, Nij}
    IJF{S, Nij}(SArray(parent(ijf)))
end

function replace_basetype(data::IJF{S, Nij}, ::Type{T}) where {S, Nij, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return IJF{S′, Nij}(similar(array, T))
end

function Base.size(data::IJF{S, Nij}) where {S, Nij}
    return (Nij, Nij, 1, 1, 1)
end
function Base.fill!(data::IJF{S, Nij}, val) where {S, Nij}
    @inbounds for j in 1:Nij, i in 1:Nij
        data[i, j] = val
    end
    return data
end

@generated function _property_view(
    data::IJF{S, Nij, A},
    ::Val{Idx},
) where {S, Nij, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(IJF{$SS, $Nij}(
        @inbounds view(parent(data), :, :, $field_byterange)
    ))
end

@inline function Base.getproperty(data::IJF{S, Nij}, i::Integer) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, :, (offset + 1):(offset + nbytes))
    IJF{SS, Nij}(dataview)
end

@inline function Base.getindex(
    data::IJF{S, Nij},
    i::Integer,
    j::Integer,
    k = nothing,
    v = nothing,
    h = nothing,
) where {S, Nij}
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(data, (i, j)))
    @inbounds get_struct(parent(data), S, Val(3), CartesianIndex(i, j, 1))
end

@inline function Base.setindex!(
    data::IJF{S, Nij},
    val,
    i::Integer,
    j::Integer,
) where {S, Nij}
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(data, (i, j)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(i, j, 1),
    )
end

@inline function column(data::IJF{S, Nij}, i, j) where {S, Nij}
    @boundscheck (1 <= j <= Nij && 1 <= i <= Nij) ||
                 throw(BoundsError(data, (i, j)))
    dataview = @inbounds view(parent(data), i, j, :)
    DataF{S}(dataview)
end

# ======================
# DataSlab1D DataLayout
# ======================

@propagate_inbounds function Base.getindex(slab::DataSlab1D, I::CartesianIndex)
    @inbounds slab[I[1]]
end

@propagate_inbounds function Base.setindex!(
    slab::DataSlab1D,
    val,
    I::CartesianIndex,
)
    @inbounds slab[I[1]] = val
end

function Base.size(::DataSlab1D{<:Any, Ni}) where {Ni}
    return (Ni, 1, 1, 1, 1)
end
Base.axes(::DataSlab1D{S, Ni}) where {S, Ni} = (SOneTo(Ni),)
Base.lastindex(::DataSlab1D{S, Ni}) where {S, Ni} = Ni

@inline function slab(data::DataSlab1D, h)
    @boundscheck (h >= 1) || throw(BoundsError(data, (h,)))
    data
end

@inline function slab(data::DataSlab1D, v, h)
    @boundscheck (v >= 1 && h >= 1) || throw(BoundsError(data, (v, h)))
    data
end

"""
    IF{S, Ni, A} <: DataSlab1D{S, Ni}

Backing `DataLayout` for 1D spectral element slab data.

Nodal element data (I) are contiguous for each `S` datatype struct field (F) for a single element slab.

A `DataSlab1D` view can be returned from other `Data1D` objects by calling `slab(data, idx...)`.
"""
struct IF{S, Ni, A} <: DataSlab1D{S, Ni}
    array::A
end

function IF{S, Ni}(array::AbstractArray{T, 2}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    check_basetype(T, S)
    IF{S, Ni, typeof(array)}(array)
end
function IF{S, Ni}(::Type{MArray}, ::Type{T}) where {S, Ni, T}
    Nf = typesize(T, S)
    array = MArray{Tuple{Ni, Nf}, T, 2, Ni * Nf}(undef)
    IF{S, Ni}(array)
end
function SArray(data::IF{S, Ni, <:MArray}) where {S, Ni}
    IF{S, Ni}(SArray(parent(data)))
end

function replace_basetype(data::IF{S, Ni}, ::Type{T}) where {S, Ni, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return IF{S′, Ni}(similar(array, T))
end

function Base.fill!(data::IF{S, Ni}, val) where {S, Ni}
    @inbounds for i in 1:Ni
        data[i] = val
    end
    return data
end


@generated function _property_view(
    data::IF{S, Ni, A},
    ::Val{Idx},
) where {S, Ni, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(IF{$SS, $Ni}(@inbounds view(parent(data), :, $field_byterange)))
end

@inline function Base.getproperty(data::IF{S, Ni}, f::Integer) where {S, Ni}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, f)
    offset = fieldtypeoffset(T, S, f)
    len = typesize(T, SS)
    dataview = @inbounds view(array, :, (offset + 1):(offset + len))
    IF{SS, Ni}(dataview)
end

@inline function Base.getindex(
    data::IF{S, Ni},
    i::Integer,
    j = nothing,
    k = nothing,
    v = nothing,
    h = nothing,
) where {S, Ni}
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    @inbounds get_struct(parent(data), S, Val(2), CartesianIndex(i, 1))
end

@inline function Base.setindex!(data::IF{S, Ni}, val, i::Integer) where {S, Ni}
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(2),
        CartesianIndex(i, 1),
    )
end

@inline function column(data::IF{S, Ni}, i) where {S, Ni}
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    dataview = @inbounds view(parent(data), i, :)
    DataF{S}(dataview)
end

# ======================
# DataColumn DataLayout
# ======================

Base.length(data::DataColumn) = size(parent(data), 1)
Base.size(data::DataColumn) = (1, 1, 1, length(data), 1)

"""
    VF{S, A} <: DataColumn{S}

Backing `DataLayout` for 1D FV column data.

Column level data (V) are contiguous for each `S` datatype struct field (F).

A `DataColumn` view can be returned from other `Data1DX`, `Data2DX` objects by calling `column(data, idx...)`.
"""
struct VF{S, A} <: DataColumn{S}
    array::A
end

function VF{S}(array::AbstractArray{T, 2}) where {S, T}
    check_basetype(T, S)
    VF{S, typeof(array)}(array)
end

function VF{S}(array::AbstractVector{T}) where {S, T}
    @assert typesize(T, S) == 1
    VF{S}(reshape(array, (:, 1)))
end

function VF{S}(ArrayType, nelements) where {S}
    T = eltype(ArrayType)
    VF{S}(ArrayType(undef, nelements, typesize(T, S)))
end

function replace_basetype(data::VF{S}, ::Type{T}) where {S, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return VF{S′}(similar(array, T))
end

Base.copy(data::VF{S}) where {S} = VF{S}(copy(parent(data)))
Base.lastindex(data::VF) = length(data)
function Base.size(data::VF)
    Nv = size(parent(data), 1)
    return (1, 1, 1, Nv, 1)
end
function Base.fill!(data::VF, val)
    Nv = size(parent(data), 1)
    @inbounds for v in 1:Nv
        data[v] = val
    end
    return data
end

@generated function _property_view(data::VF{S, A}, ::Val{Idx}) where {S, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(VF{$SS}(@inbounds view(parent(data), :, $field_byterange)))
end

@inline function Base.getproperty(data::VF{S}, i::Integer) where {S}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, (offset + 1):(offset + nbytes))
    VF{SS}(dataview)
end

@inline function Base.getindex(data::VF{S}, v::Integer) where {S}
    @boundscheck 1 <= v <= size(parent(data), 1) ||
                 throw(BoundsError(data, (v,)))
    @inbounds get_struct(parent(data), S, Val(2), CartesianIndex(v, 1))
end

@propagate_inbounds function Base.getindex(
    col::DataColumn,
    I::CartesianIndex{5},
)
    @inbounds col[I[4]]
end

@propagate_inbounds function Base.setindex!(
    col::DataColumn,
    val,
    I::CartesianIndex{5},
)
    @inbounds col[I[4]] = val
end

@inline function Base.setindex!(data::VF{S}, val, v::Integer) where {S}
    @boundscheck (1 <= v <= length(parent(data))) ||
                 throw(BoundsError(data, (v,)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(2),
        CartesianIndex(v, 1),
    )
end

@inline function column(data::VF, i, h)
    @boundscheck (i >= 1 && h >= 1) || throw(BoundsError(data, (i, h)))
    data
end

@inline function column(data::VF, i, j, h)
    @boundscheck (i >= 1 && j >= 1 && h >= 1) ||
                 throw(BoundsError(data, (i, j, h)))
    data
end

@inline function level(data::VF{S}, v) where {S}
    Nv = size(data, 4)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v)))
    array = parent(data)
    dataview = @inbounds view(array, v, :)
    DataF{S}(dataview)
end

# ======================
# Data2DX DataLayout
# ======================

"""
    VIJFH{S, Nij, A} <: Data2DX{S, Nij}

Backing `DataLayout` for 2D spectral element slab + extruded 1D FV column data.

Column levels (V) are contiguous for every element nodal point (I, J)
for each `S` datatype struct field (F), for each 2D mesh element slab (H).
"""
struct VIJFH{S, Nij, A} <: Data2DX{S, Nij}
    array::A
end

function VIJFH{S, Nij}(array::AbstractArray{T, 5}) where {S, Nij, T}
    @assert size(array, 2) == size(array, 3) == Nij
    VIJFH{S, Nij, typeof(array)}(array)
end

function replace_basetype(data::VIJFH{S, Nij}, ::Type{T}) where {S, Nij, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return VIJFH{S′, Nij}(similar(array, T))
end

function Base.copy(data::VIJFH{S, Nij}) where {S, Nij}
    VIJFH{S, Nij}(copy(parent(data)))
end

function Base.size(data::VIJFH{<:Any, Nij}) where {Nij}
    Nv = size(parent(data), 1)
    Nh = size(parent(data), 5)
    return (Nij, Nij, 1, Nv, Nh)
end

function Base.length(data::VIJFH)
    size(parent(data), 1) * size(parent(data), 5)
end

function Base.fill!(data::VIJFH, val)
    (Ni, Nj, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end

@generated function _property_view(
    data::VIJFH{S, Nij, A},
    ::Val{Idx},
) where {S, Nij, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(VIJFH{$SS, $Nij}(
        @inbounds view(parent(data), :, :, :, $field_byterange, :)
    ))
end

@propagate_inbounds function Base.getproperty(
    data::VIJFH{S, Nij},
    i::Integer,
) where {S, Nij}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, :, :, (offset + 1):(offset + nbytes), :)
    VIJFH{SS, Nij}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIJFH{S, Nij}, v, h) where {S, Nij}
    array = parent(data)
    Nv = size(array, 1)
    Nh = size(array, 5)
    @boundscheck (1 <= v <= Nv && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (v, h)))
    Nf = size(array, 4)
    dataview = @inbounds view(
        array,
        v,
        Base.Slice(Base.OneTo(Nij)),
        Base.Slice(Base.OneTo(Nij)),
        Base.Slice(Base.OneTo(Nf)),
        h,
    )
    IJF{S, Nij}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIJFH{S, Nij}, i, j, h) where {S, Nij}
    array = parent(data)
    Nh = size(array, 5)
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, j, h)))
    Nv = size(array, 1)
    Nf = size(array, 4)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, j, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S}(dataview)
end

@inline function level(data::VIJFH{S, Nij}, v) where {S, Nij}
    array = parent(data)
    Nv = size(array, 1)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :, :)
    IJFH{S, Nij}(dataview)
end

@propagate_inbounds function Base.getindex(
    data::VIJFH{S},
    I::CartesianIndex{5},
) where {S}
    i, j, _, v, h = I.I
    @inbounds get_struct(parent(data), S, Val(4), CartesianIndex(v, i, j, 1, h))
end

@propagate_inbounds function Base.setindex!(
    data::VIJFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    i, j, _, v, h = I.I
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(4),
        CartesianIndex(v, i, j, 1, h),
    )
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::VIJFH{S, Nij},
) where {S, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        VIJFH{S, Nij}(gatherdata)
    else
        nothing
    end
end

@inline function Base.getindex(data::VIJFH{S}, i, j, _, v, h) where {S}
    @inbounds get_struct(parent(data), S, Val(4), CartesianIndex(v, i, j, 1, h))
end
@inline function Base.setindex!(data::VIJFH{S}, val, i, j, _, v, h) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(4),
        CartesianIndex(v, i, j, 1, h),
    )
end

# ======================
# Data1DX DataLayout
# ======================

"""
    VIFH{S, Ni, A} <: Data1DX{S, Ni}

Backing `DataLayout` for 1D spectral element slab + extruded 1D FV column data.

Column levels (V) are contiguous for every element nodal point (I)
for each datatype `S` struct field (F), for each 1D mesh element slab (H).
"""
struct VIFH{S, Ni, A} <: Data1DX{S, Ni}
    array::A
end

function VIFH{S, Ni}(array::AbstractArray{T, 4}) where {S, Ni, T}
    @assert size(array, 2) == Ni
    check_basetype(T, S)
    VIFH{S, Ni, typeof(array)}(array)
end

rebuild(data::VIFH{S, Ni}, array::A) where {S, Ni, A <: AbstractArray} =
    VIFH{S, Ni}(array)

function replace_basetype(data::VIFH{S, Ni}, ::Type{T}) where {S, Ni, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return VIFH{S′, Ni}(similar(array, T))
end

Base.copy(data::VIFH{S, Ni}) where {S, Ni} = VIFH{S, Ni}(copy(parent(data)))

function Base.size(data::VIFH{<:Any, Ni}) where {Ni}
    Nv = size(parent(data), 1)
    Nh = size(parent(data), 4)
    return (Ni, 1, 1, Nv, Nh)
end

function Base.length(data::VIFH)
    size(parent(data), 1) * size(parent(data), 4)
end
function Base.fill!(data::VIFH, val)
    (Ni, _, _, Nv, Nh) = size(data)
    @inbounds for h in 1:Nh, v in 1:Nv
        fill!(slab(data, v, h), val)
    end
    return data
end

@propagate_inbounds function Base.getindex(
    data::VIFH{S},
    I::CartesianIndex{5},
) where {S}
    i, _, _, v, h = I.I
    @inbounds get_struct(parent(data), S, Val(3), CartesianIndex(v, i, 1, h))
end


@generated function _property_view(
    data::VIFH{S, Ni, A},
    ::Val{Idx},
) where {S, Ni, A, Idx}
    SS = fieldtype(S, Idx)
    T = eltype(A)
    offset = fieldtypeoffset(T, S, Idx)
    nbytes = typesize(T, SS)
    field_byterange = (offset + 1):(offset + nbytes)
    return :(VIFH{$SS, $Ni}(
        @inbounds view(parent(data), :, :, $field_byterange, :)
    ))
end

@inline function Base.getproperty(data::VIFH{S, Ni}, i::Integer) where {S, Ni}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    dataview = @inbounds view(array, :, :, (offset + 1):(offset + nbytes), :)
    VIFH{SS, Ni}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIFH{S, Ni}, v, h) where {S, Ni}
    array = parent(data)
    Nv = size(array, 1)
    Nh = size(array, 4)
    @boundscheck (1 <= v <= Nv && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (v, h)))
    Nf = size(array, 3)
    dataview = @inbounds SubArray(
        array,
        (v, Base.Slice(Base.OneTo(Ni)), Base.Slice(Base.OneTo(Nf)), h),
    )
    IF{S, Ni}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIFH{S, Ni}, i, h) where {S, Ni}
    array = parent(data)
    Nh = size(array, 4)
    @boundscheck (1 <= i <= Ni && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, h)))
    Nv = size(array, 1)
    Nf = size(array, 3)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S}(dataview)
end

@inline function column(data::VIFH{S, Ni}, i, j, h) where {S, Ni}
    array = parent(data)
    Nh = size(array, 4)
    @boundscheck (1 <= i <= Ni && j == 1 && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, j, h)))
    Nv = size(array, 1)
    Nf = size(array, 3)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S}(dataview)
end

@inline function level(data::VIFH{S, Nij}, v) where {S, Nij}
    array = parent(data)
    Nv = size(array, 1)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :)
    IFH{S, Nij}(dataview)
end

@inline function Base.getindex(data::VIFH{S}, i, _, _, v, h) where {S}
    @inbounds get_struct(parent(data), S, Val(3), CartesianIndex(v, i, 1, h))
end
@inline function Base.setindex!(data::VIFH{S}, val, i, _, _, v, h) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(v, i, 1, h),
    )
end
@inline function Base.setindex!(
    data::VIFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    i, _, _, v, h = I.I
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(3),
        CartesianIndex(v, i, 1, h),
    )
end

# =========================================
# Special DataLayouts for regular gridding
# =========================================

struct IH1JH2{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

"""
    IH1JH2{S, Nij}(data::AbstractMatrix{S})

Stores a 2D field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid for ex. plotting / field output.
"""
function IH1JH2{S, Nij}(array::AbstractMatrix{S}) where {S, Nij}
    @assert size(array, 1) % Nij == 0
    @assert size(array, 2) % Nij == 0
    IH1JH2{S, Nij, typeof(array)}(array)
end

Base.copy(data::IH1JH2{S, Nij}) where {S, Nij} =
    IH1JH2{S, Nij}(copy(parent(data)))

function Base.size(data::IH1JH2{S, Nij}) where {S, Nij}
    Nv = 1
    Nh = div(length(parent(data)), Nij * Nij)
    (Nij, Nij, 1, Nv, Nh)
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

@inline function slab(data::IH1JH2{S, Nij}, h::Integer) where {S, Nij}
    N1, N2 = size(parent(data))
    n1 = div(N1, Nij)
    n2 = div(N2, Nij)
    z2, z1 = fldmod(h - 1, n1)
    @boundscheck (1 <= h <= n1 * n2) || throw(BoundsError(data, (h,)))
    dataview =
        @inbounds view(parent(data), Nij * z1 .+ (1:Nij), Nij * z2 .+ (1:Nij))
    return dataview
end

struct IV1JH2{S, Ni, A} <: Data1DX{S, Ni}
    array::A
end

"""
    IV1JH2{S, Ni}(data::AbstractMatrix{S})

Stores values from an extruded 1D spectral field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid for ex. plotting / field output.
"""
function IV1JH2{S, Ni}(array::AbstractMatrix{S}) where {S, Ni}
    @assert size(array, 2) % Ni == 0
    IV1JH2{S, Ni, typeof(array)}(array)
end

Base.copy(data::IV1JH2{S, Ni}) where {S, Ni} = IV1JH2{S, Ni}(copy(parent(data)))

function Base.size(data::IV1JH2{S, Ni}) where {S, Ni}
    Nv = size(parent(data), 1)
    Nh = div(size(parent(data), 2), Ni)
    (Ni, 1, 1, Nv, Nh)
end

Base.length(data::IV1JH2{S, Ni}) where {S, Ni} = div(length(parent(data)), Ni)

function Base.similar(
    data::IV1JH2{S, Ni, A},
    ::Type{Eltype},
) where {S, Ni, A, Eltype}
    array = similar(A, Eltype)
    return IV1JH2{Eltype, Ni}(array)
end

@inline function slab(data::IV1JH2{S, Ni}, v::Integer, h::Integer) where {S, Ni}
    N1, N2 = size(parent(data))
    n1 = N1
    n2 = div(N2, Ni)
    _, z2 = fldmod(h - 1, n2)
    @boundscheck (1 <= v <= n1) && (1 <= h <= n2) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), v, Ni * z2 .+ (1:Ni))
    return dataview
end

rebuild(data::AbstractData, ::Type{DA}) where {DA} =
    rebuild(data, DA(getfield(data, :array)))

# broadcast machinery
include("broadcast.jl")

# GPU method specializations
include("cuda.jl")

end # module
