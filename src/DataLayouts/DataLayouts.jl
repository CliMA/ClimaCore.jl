"""
    ClimaCore.DataLayouts

Defines the following DataLayouts (see individual docs for more info):

TODO: Add links to these datalayouts

 - `IJKFVH`
 - `IJFH`
 - `IFH`
 - `DataF`
 - `IJF`
 - `IF`
 - `VF`
 - `VIJFH`
 - `VIFH`
 - `IH1JH2`
 - `IV1JH2`


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
import MultiBroadcastFusion as MBF
import Adapt

import ..slab, ..slab_args, ..column, ..column_args, ..level
export slab, column, level, IJFH, IJF, IFH, IF, VF, VIJFH, VIFH, DataF

# Internal types for managing CPU/GPU dispatching
abstract type AbstractDispatchToDevice end
struct ToCPU <: AbstractDispatchToDevice end
struct ToCUDA <: AbstractDispatchToDevice end

include("struct.jl")

abstract type AbstractData{S} end

abstract type AbstractDataSpecificCartesianIndex{N} <:
              Base.AbstractCartesianIndex{N} end

"""
    DataSpecificCartesianIndex{N} <: AbstractDataSpecificCartesianIndex{N}

A DataLayout-specific CartesianIndex, which is used to provide support for
`getindex` for DataLayouts such that indices are not swapped. This is used
to improve memory access patterns on GPUs.
"""
struct DataSpecificCartesianIndex{N} <: AbstractDataSpecificCartesianIndex{N}
    I::CartesianIndex{N}
end

@inline Base.size(data::AbstractData, i::Integer) = size(data)[i]
@inline Base.size(data::AbstractData) = universal_size(data)

"""
    struct UniversalSize{Ni, Nj, Nv, Nh} end
    UniversalSize(data::AbstractData)

A struct containing static dimensions, universal to all datalayouts:
 - `Ni` number of spectral element nodal degrees of freedom in first horizontal direction
 - `Nj` number of spectral element nodal degrees of freedom in second horizontal direction
 - `Nv` number of vertical degrees of freedom
 - `Nh` number of horizontal elements
"""
struct UniversalSize{Ni, Nj, Nv, Nh} end

@inline function UniversalSize(data::AbstractData)
    us = universal_size(data)
    UniversalSize{us[1], us[2], us[4], us[5]}()
end

@inline array_length(data::AbstractData) = prod(size(parent(data)))

"""
    (Ni, Nj, Nv, _, Nh) = universal_size(data::AbstractData)

A tuple of compile-time known type parameters,
corresponding to `UniversalSize`. The field dimension
is excluded and is returned as 1.
"""
@inline universal_size(::UniversalSize{Ni, Nj, Nv, Nh}) where {Ni, Nj, Nv, Nh} =
    (Ni, Nj, Nv, 1, Nh)

"""
    get_N(::AbstractData)
    get_N(::UniversalSize)

Statically returns `prod((Ni, Nj, Nv, Nh))`
"""
@inline get_N(::UniversalSize{Ni, Nj, Nv, Nh}) where {Ni, Nj, Nv, Nh} =
    prod((Ni, Nj, Nv, Nh))

"""
    get_Nv(::UniversalSize)

Statically returns `Nv`.
"""
@inline get_Nv(::UniversalSize{Ni, Nj, Nv}) where {Ni, Nj, Nv} = Nv

"""
    get_Nij(::UniversalSize)

Statically returns `Nij`.
"""
@inline get_Nij(::UniversalSize{Nij}) where {Nij} = Nij

"""
    get_Nh(::UniversalSize)

Statically returns `Nh`.
"""
@inline get_Nh(::UniversalSize{Ni, Nj, Nv, Nh}) where {Ni, Nj, Nv, Nh} = Nh

@inline get_Nh(data::AbstractData) = get_Nh(UniversalSize(data))
@inline get_Nij(data::AbstractData) = get_Nij(UniversalSize(data))
@inline get_Nv(data::AbstractData) = get_Nv(UniversalSize(data))
@inline get_N(data::AbstractData) = get_N(UniversalSize(data))

@inline universal_size(data::AbstractData) = universal_size(UniversalSize(data))

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
    DataColumn{S, Nv}

Abstract type for data storage for a column. Objects `data` should define a
`data[k,v]`, returning a value of type `S`.
"""
abstract type DataColumn{S, Nv} <: AbstractData{S} end

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
    Data1DX{S, Nv, Ni}

Abstract type for data storage for a 1D field with extruded columns.
The horizontal is made up of `Ni` values of type `S`.

Objects `data` should define `slab(data, v, h)` to return a
`DataSlab1D{S,Ni}` object, and a `column(data, i, h)` to return a `DataColumn`.
"""
abstract type Data1DX{S, Nv, Ni} <: AbstractData{S} end

"""
    Data2DX{S,Nv,Nij}

Abstract type for data storage for a 2D field with extruded columns.
The horizontal is made  is made up of `Nij × Nij` values of type `S`.


Objects `data` should define `slab(data, v, h)` to return a
`DataSlab2D{S,Nv,Nij}` object, and a `column(data, i, j, h)` to return a `DataColumn`.
"""
abstract type Data2DX{S, Nv, Nij} <: AbstractData{S} end

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

Base.@propagate_inbounds function Base.getproperty(
    data::AbstractData{S},
    i::Integer,
) where {S}
    array = parent(data)
    T = eltype(array)
    SS = fieldtype(S, i)
    offset = fieldtypeoffset(T, S, i)
    nbytes = typesize(T, SS)
    fdim = field_dim(data)
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ndims(data) - fdim))
    dataview =
        @inbounds view(array, Ipre..., (offset + 1):(offset + nbytes), Ipost...)
    union_all(data){SS, Base.tail(type_params(data))...}(dataview)
end

# In the past, we've sometimes needed a generated function
# for inference and constant propagation:
Base.@propagate_inbounds @generated function _property_view(
    data::AD,
    ::Val{Idx},
) where {S, Idx, AD <: AbstractData{S}}
    SS = fieldtype(S, Idx)
    T = eltype(parent_array_type(AD))
    offset = fieldtypeoffset(T, S, Val(Idx))
    nbytes = typesize(T, SS)
    fdim = field_dim(AD)
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ndims(data) - fdim))
    field_byterange = (offset + 1):(offset + nbytes)
    return :($(union_all(AD)){$SS, $(Base.tail(type_params(AD)))...}(
        @inbounds view(parent(data), $Ipre..., $field_byterange, $Ipost...)
    ))
end

function replace_basetype(data::AbstractData{S}, ::Type{T}) where {S, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return union_all(data){S′, Base.tail(type_params(data))...}(
        similar(array, T),
    )
end

# ==================
# Data3D DataLayout
# ==================

"""
    IJKFVH{S, Nij, Nk}(array::AbstractArray{T, 6}) <: Data3D{S, Nij, Nk}

A 3D DataLayout. TODO: Add more docs
"""
struct IJKFVH{S, Nij, Nk, Nv, Nh, A} <: Data3D{S, Nij, Nk}
    array::A
end

parent_array_type(
    ::Type{IJKFVH{S, Nij, Nk, Nv, Nh, A}},
) where {S, Nij, Nk, Nv, Nh, A} = A

function IJKFVH{S, Nij, Nk, Nv, Nh}(
    array::AbstractArray{T, 6},
) where {S, Nij, Nk, Nv, Nh, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == Nk
    @assert size(array, 4) == typesize(T, S)
    @assert size(array, 5) == Nv
    @assert size(array, 6) == Nh
    IJKFVH{S, Nij, Nk, Nv, Nh, typeof(array)}(array)
end

@inline universal_size(
    ::IJKFVH{S, Nij, Nk, Nv, Nh},
) where {S, Nij, Nk, Nv, Nh} = (Nij, Nij, Nk, Nv, Nh)

# ==================
# Data2D DataLayout
# ==================

"""
    IJFH{S, Nij, A} <: Data2D{S, Nij}
    IJFH{S,Nij}(ArrayType, nelements)


Backing `DataLayout` for 2D spectral element slabs.

Element nodal point (I,J) data is contiguous for each datatype `S` struct field (F),
for each 2D mesh element slab (H).

The `ArrayType`-constructor constructs a IJFH 2D Spectral
DataLayout given the backing `ArrayType`, quadrature degrees
of freedom `Nij × Nij`, and the number of mesh elements `nelements`.
"""
struct IJFH{S, Nij, Nh, A} <: Data2D{S, Nij}
    array::A
end

parent_array_type(::Type{IJFH{S, Nij, Nh, A}}) where {S, Nij, Nh, A} = A

function IJFH{S, Nij, Nh}(array::AbstractArray{T, 4}) where {S, Nij, Nh, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == typesize(T, S)
    @assert size(array, 4) == Nh
    IJFH{S, Nij, Nh, typeof(array)}(array)
end

@inline universal_size(::IJFH{S, Nij, Nh}) where {S, Nij, Nh} =
    (Nij, Nij, 1, 1, Nh)

function IJFH{S, Nij, Nh}(::Type{ArrayType}) where {S, Nij, Nh, ArrayType}
    T = eltype(ArrayType)
    IJFH{S, Nij, Nh}(ArrayType(undef, Nij, Nij, typesize(T, S), Nh))
end

@propagate_inbounds function Base.getindex(
    data::IJFH{S},
    I::CartesianIndex{5},
) where {S}
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end
@propagate_inbounds function Base.setindex!(
    data::IJFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end


Base.length(data::IJFH) = get_Nh(data)

@inline function slab(data::IJFH{S, Nij}, h::Integer) where {S, Nij}
    @boundscheck (1 <= h <= get_Nh(data)) || throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, :, :, h)
    IJF{S, Nij}(dataview)
end

@inline function slab(data::IJFH{S, Nij}, v::Integer, h::Integer) where {S, Nij}
    @boundscheck (v >= 1 && 1 <= h <= get_Nh(data)) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), :, :, :, h)
    IJF{S, Nij}(dataview)
end

@inline function column(data::IJFH{S, Nij}, i, j, h) where {S, Nij}
    @boundscheck (1 <= j <= Nij && 1 <= i <= Nij && 1 <= h <= get_Nh(data)) ||
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
        Nh = size(gatherdata, 4)
        IJFH{S, Nij, Nh}(gatherdata)
    else
        nothing
    end
end

# ==================
# Data1D DataLayout
# ==================

Base.length(data::Data1D) = get_Nh(data)

"""
    IFH{S,Ni,Nh,A} <: Data1D{S, Ni}
    IFH{S,Ni,Nh}(ArrayType)

Backing `DataLayout` for 1D spectral element slabs.

Element nodal point (I) data is contiguous for each
datatype `S` struct field (F), for each 1D mesh element (H).


The `ArrayType`-constructor makes a IFH 1D Spectral
DataLayout given the backing `ArrayType`, quadrature
degrees of freedom `Ni`, and the number of mesh elements
`Nh`.
"""
struct IFH{S, Ni, Nh, A} <: Data1D{S, Ni}
    array::A
end

parent_array_type(::Type{IFH{S, Ni, Nh, A}}) where {S, Ni, Nh, A} = A

function IFH{S, Ni, Nh}(array::AbstractArray{T, 3}) where {S, Ni, Nh, T}
    check_basetype(T, S)
    @assert size(array, 1) == Ni
    @assert size(array, 2) == typesize(T, S)
    @assert size(array, 3) == Nh
    IFH{S, Ni, Nh, typeof(array)}(array)
end

function IFH{S, Ni, Nh}(::Type{ArrayType}) where {S, Ni, Nh, ArrayType}
    T = eltype(ArrayType)
    IFH{S, Ni, Nh}(ArrayType(undef, Ni, typesize(T, S), Nh))
end

@inline universal_size(::IFH{S, Ni, Nh}) where {S, Ni, Nh} = (Ni, 1, 1, 1, Nh)

@inline function slab(data::IFH{S, Ni}, h::Integer) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh(data)) || throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, :, h)
    IF{S, Ni}(dataview)
end
Base.@propagate_inbounds slab(data::IFH, v::Integer, h::Integer) = slab(data, h)

@inline function column(data::IFH{S, Ni}, i, h) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh(data) && 1 <= i <= Ni) ||
                 throw(BoundsError(data, (i, h)))
    dataview = @inbounds view(parent(data), i, :, h)
    DataF{S}(dataview)
end
Base.@propagate_inbounds column(data::IFH{S, Ni}, i, j, h) where {S, Ni} =
    column(data, i, h)

@inline function Base.getindex(data::IFH{S}, I::CartesianIndex{5}) where {S}
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end
@inline function Base.setindex!(
    data::IFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end

# ======================
# Data0D DataLayout
# ======================

Base.length(data::Data0D) = 1
@inline universal_size(::Data0D) = (1, 1, 1, 1, 1)

"""
    DataF{S, A} <: Data0D{S}

Backing `DataLayout` for 0D point data.
"""
struct DataF{S, A} <: Data0D{S}
    array::A
end

parent_array_type(::Type{DataF{S, A}}) where {S, A} = A

function DataF{S}(array::AbstractVector{T}) where {S, T}
    check_basetype(T, S)
    @assert size(array, 1) == typesize(T, S)
    DataF{S, typeof(array)}(array)
end

function DataF{S}(::Type{ArrayType}) where {S, ArrayType}
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

Base.@propagate_inbounds function Base.getindex(data::DataF{S}) where {S}
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        CartesianIndex(1),
    )
end

@propagate_inbounds function Base.getindex(col::Data0D, I::CartesianIndex{5})
    @inbounds col[]
end

Base.@propagate_inbounds function Base.setindex!(data::DataF{S}, val) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
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

# ======================
# DataSlab2D DataLayout
# ======================

@inline universal_size(::DataSlab2D{S, Nij}) where {S, Nij} =
    (Nij, Nij, 1, 1, 1)
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

parent_array_type(::Type{IJF{S, Nij, A}}) where {S, Nij, A} = A

function IJF{S, Nij}(array::AbstractArray{T, 3}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    check_basetype(T, S)
    @assert size(array, 3) == typesize(T, S)
    IJF{S, Nij, typeof(array)}(array)
end

function IJF{S, Nij}(::Type{MArray}, ::Type{T}) where {S, Nij, T}
    Nf = typesize(T, S)
    array = MArray{Tuple{Nij, Nij, Nf}, T, 3, Nij * Nij * Nf}(undef)
    IJF{S, Nij}(array)
end
function SArray(ijf::IJF{S, Nij, <:MArray}) where {S, Nij}
    IJF{S, Nij}(SArray(parent(ijf)))
end

@inline universal_size(::IJF{S, Nij}) where {S, Nij} = (Nij, Nij, 1, 1, 1)

@inline function Base.getindex(
    data::IJF{S, Nij},
    I::CartesianIndex,
) where {S, Nij}
    i = I.I[1]
    j = I.I[2]
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(data, (i, j)))
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        CartesianIndex(i, j, 1),
    )
end

@inline function Base.setindex!(
    data::IJF{S, Nij},
    val,
    I::CartesianIndex,
) where {S, Nij}
    i = I.I[1]
    j = I.I[2]
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij) ||
                 throw(BoundsError(data, (i, j)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
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

@inline universal_size(::DataSlab1D{<:Any, Ni}) where {Ni} = (Ni, 1, 1, 1, 1)
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

parent_array_type(::Type{IF{S, Ni, A}}) where {S, Ni, A} = A

function IF{S, Ni}(array::AbstractArray{T, 2}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    check_basetype(T, S)
    @assert size(array, 2) == typesize(T, S)
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

@inline function Base.getindex(data::IF{S, Ni}, I::CartesianIndex) where {S, Ni}
    i = I.I[1]
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        CartesianIndex(i, 1),
    )
end

@inline function Base.setindex!(
    data::IF{S, Ni},
    val,
    I::CartesianIndex,
) where {S, Ni}
    i = I.I[1]
    @boundscheck (1 <= i <= Ni) || throw(BoundsError(data, (i,)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
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

Base.length(data::DataColumn) = get_Nv(data)
@inline universal_size(::DataColumn{S, Nv}) where {S, Nv} = (1, 1, 1, Nv, 1)

"""
    VF{S, A} <: DataColumn{S, Nv}

Backing `DataLayout` for 1D FV column data.

Column level data (V) are contiguous for each `S` datatype struct field (F).

A `DataColumn` view can be returned from other `Data1DX`, `Data2DX` objects by calling `column(data, idx...)`.
"""
struct VF{S, Nv, A} <: DataColumn{S, Nv}
    array::A
end

parent_array_type(::Type{VF{S, Nv, A}}) where {S, Nv, A} = A

function VF{S, Nv}(array::AbstractArray{T, 2}) where {S, Nv, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == typesize(T, S)
    VF{S, Nv, typeof(array)}(array)
end

function VF{S, Nv}(array::AbstractVector{T}) where {S, Nv, T}
    check_basetype(T, S)
    @assert typesize(T, S) == 1
    VF{S, Nv}(reshape(array, (:, 1)))
end

function VF{S, Nv}(::Type{ArrayType}, nelements) where {S, Nv, ArrayType}
    T = eltype(ArrayType)
    check_basetype(T, S)
    VF{S, Nv}(ArrayType(undef, nelements, typesize(T, S)))
end

Base.lastindex(data::VF) = length(data)

nlevels(::VF{S, Nv}) where {S, Nv} = Nv

Base.@propagate_inbounds Base.getproperty(data::VF, i::Integer) =
    _property_view(data, Val(i))

@inline function Base.getindex(data::VF{S, Nv}, I::CartesianIndex) where {S, Nv}
    v = I.I[4]
    @boundscheck 1 <= v <= nlevels(data) || throw(BoundsError(data, (v,)))
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        CartesianIndex(v, 1),
    )
end

@inline function Base.setindex!(data::VF{S}, val, I::CartesianIndex) where {S}
    v = I.I[4]
    @boundscheck (1 <= v <= nlevels(data)) || throw(BoundsError(data, (v,)))
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
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
    @boundscheck (1 <= v <= nlevels(data)) || throw(BoundsError(data, (v)))
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
struct VIJFH{S, Nv, Nij, Nh, A} <: Data2DX{S, Nv, Nij}
    array::A
end

parent_array_type(::Type{VIJFH{S, Nv, Nij, Nh, A}}) where {S, Nv, Nij, Nh, A} =
    A

function VIJFH{S, Nv, Nij, Nh}(
    array::AbstractArray{T, 5},
) where {S, Nv, Nij, Nh, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == size(array, 3) == Nij
    @assert size(array, 4) == typesize(T, S)
    @assert size(array, 5) == Nh
    VIJFH{S, Nv, Nij, Nh, typeof(array)}(array)
end

nlevels(::VIJFH{S, Nv}) where {S, Nv} = Nv

@inline universal_size(::VIJFH{<:Any, Nv, Nij, Nh}) where {Nv, Nij, Nh} =
    (Nij, Nij, 1, Nv, Nh)

Base.length(data::VIJFH) = get_Nv(data) * get_Nh(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIJFH{S, Nv, Nij, Nh}, v, h) where {S, Nv, Nij, Nh}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (v, h)))
    Nf = ncomponents(data)
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
@inline function column(
    data::VIJFH{S, Nv, Nij, Nh},
    i,
    j,
    h,
) where {S, Nv, Nij, Nh}
    array = parent(data)
    @boundscheck (1 <= i <= Nij && 1 <= j <= Nij && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, j, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIJFH{S, Nv, Nij, Nh}, v) where {S, Nv, Nij, Nh}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :, :)
    IJFH{S, Nij, Nh}(dataview)
end

@propagate_inbounds function Base.getindex(
    data::VIJFH{S},
    I::CartesianIndex{5},
) where {S}
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end

@propagate_inbounds function Base.setindex!(
    data::VIJFH{S},
    val,
    I::CartesianIndex{5},
) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        to_data_specific(data, I),
    )
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::VIJFH{S, Nv, Nij},
) where {S, Nv, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        Nh = size(gatherdata, 5)
        VIJFH{S, Nv, Nij, Nh}(gatherdata)
    else
        nothing
    end
end

# ======================
# Data1DX DataLayout
# ======================

"""
    VIFH{S, Nv, Ni, A} <: Data1DX{S, Nv, Ni}

Backing `DataLayout` for 1D spectral element slab + extruded 1D FV column data.

Column levels (V) are contiguous for every element nodal point (I)
for each datatype `S` struct field (F), for each 1D mesh element slab (H).
"""
struct VIFH{S, Nv, Ni, Nh, A} <: Data1DX{S, Nv, Ni}
    array::A
end

parent_array_type(::Type{VIFH{S, Nv, Ni, Nh, A}}) where {S, Nv, Ni, Nh, A} = A

function VIFH{S, Nv, Ni, Nh}(
    array::AbstractArray{T, 4},
) where {S, Nv, Ni, Nh, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == Ni
    @assert size(array, 3) == typesize(T, S)
    @assert size(array, 4) == Nh
    VIFH{S, Nv, Ni, Nh, typeof(array)}(array)
end

nlevels(::VIFH{S, Nv}) where {S, Nv} = Nv

@inline universal_size(::VIFH{<:Any, Nv, Ni, Nh}) where {Nv, Ni, Nh} =
    (Ni, 1, 1, Nv, Nh)

Base.length(data::VIFH) = nlevels(data) * get_Nh(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIFH{S, Nv, Ni, Nh}, v, h) where {S, Nv, Ni, Nh}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (v, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (v, Base.Slice(Base.OneTo(Ni)), Base.Slice(Base.OneTo(Nf)), h),
    )
    IF{S, Ni}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIFH{S, Nv, Ni, Nh}, i, h) where {S, Nv, Ni, Nh}
    array = parent(data)
    @boundscheck (1 <= i <= Ni && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S, Nv}(dataview)
end

@inline function column(
    data::VIFH{S, Nv, Ni, Nh},
    i,
    j,
    h,
) where {S, Nv, Ni, Nh}
    array = parent(data)
    @boundscheck (1 <= i <= Ni && j == 1 && 1 <= h <= Nh) ||
                 throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIFH{S, Nv, Nij, Nh}, v) where {S, Nv, Nij, Nh}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :)
    IFH{S, Nij, Nh}(dataview)
end

@propagate_inbounds function Base.getindex(
    data::VIFH{S},
    I::CartesianIndex{5},
) where {S}
    i, _, _, v, h = I.I
    @inbounds get_struct(
        parent(data),
        S,
        Val(field_dim(data)),
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
        Val(field_dim(data)),
        CartesianIndex(v, i, 1, h),
    )
end

# =========================================
# Special DataLayouts for regular gridding
# =========================================

"""
    IH1JH2{S, Nij}(data::AbstractMatrix{S})

Stores a 2D field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid for ex. plotting / field output.
"""
struct IH1JH2{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

parent_array_type(::Type{IH1JH2{S, Nij, A}}) where {S, Nij, A} = A

function IH1JH2{S, Nij}(array::AbstractMatrix{S}) where {S, Nij}
    @assert size(array, 1) % Nij == 0
    @assert size(array, 2) % Nij == 0
    IH1JH2{S, Nij, typeof(array)}(array)
end

@inline universal_size(data::IH1JH2{S, Nij}) where {S, Nij} =
    (Nij, Nij, 1, 1, div(array_length(data), Nij * Nij))

Base.length(data::IH1JH2{S, Nij}) where {S, Nij} =
    div(array_length(data), Nij * Nij)

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

"""
    IV1JH2{S, n1, Ni}(data::AbstractMatrix{S})

Stores values from an extruded 1D spectral field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid for ex. plotting / field output.
"""
struct IV1JH2{S, n1, Ni, A} <: Data1DX{S, n1, Ni}
    array::A
end

parent_array_type(::Type{IV1JH2{S, n1, Ni, A}}) where {S, n1, Ni, A} = A

function IV1JH2{S, n1, Ni}(array::AbstractMatrix{S}) where {S, n1, Ni}
    @assert size(array, 2) % Ni == 0
    IV1JH2{S, n1, Ni, typeof(array)}(array)
end

@inline universal_size(data::IV1JH2{S, n1, Ni}) where {S, n1, Ni} =
    (Ni, 1, 1, n1, div(size(parent(data), 2), Ni))

Base.length(data::IV1JH2{S, n1, Ni}) where {S, n1, Ni} =
    div(array_length(data), Ni)

function Base.similar(
    data::IV1JH2{S, n1, Ni, A},
    ::Type{Eltype},
) where {S, n1, Ni, A, Eltype}
    array = similar(A, Eltype)
    return IV1JH2{Eltype, n1, Ni}(array)
end

@inline function slab(
    data::IV1JH2{S, n1, Ni},
    v::Integer,
    h::Integer,
) where {S, n1, Ni}
    N1, N2 = size(parent(data))
    n2 = div(N2, Ni)
    _, z2 = fldmod(h - 1, n2)
    @boundscheck (1 <= v <= n1) && (1 <= h <= n2) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), v, Ni * z2 .+ (1:Ni))
    return dataview
end

rebuild(data::AbstractData, ::Type{DA}) where {DA} =
    rebuild(data, DA(getfield(data, :array)))

Base.copy(data::AbstractData) =
    union_all(data){type_params(data)...}(copy(parent(data)))

# broadcast machinery
include("broadcast.jl")

Adapt.adapt_structure(to, data::AbstractData{S}) where {S} =
    union_all(data){type_params(data)...}(Adapt.adapt(to, parent(data)))

rebuild(data::AbstractData, array::AbstractArray) =
    union_all(data){type_params(data)...}(array)

empty_kernel_stats(::ClimaComms.AbstractDevice) = nothing
empty_kernel_stats() = empty_kernel_stats(ClimaComms.device())

# ==================
# Helpers
# ==================

#! format: off
@inline get_Nij(::IJKFVH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IJFH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::VIJFH{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::VIFH{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::IFH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IJF{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IF{S, Nij}) where {S, Nij} = Nij

"""
    field_dim(data::AbstractData)
    field_dim(::Type{<:AbstractData})

This is an internal function, please do not use outside of ClimaCore.

Returns the field dimension in the backing array.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline field_dim(data::AbstractData) = field_dim(typeof(data))
@inline field_dim(::Type{<:IJKFVH}) = 4
@inline field_dim(::Type{<:IJFH}) = 3
@inline field_dim(::Type{<:IFH}) = 2
@inline field_dim(::Type{<:DataF}) = 1
@inline field_dim(::Type{<:IJF}) = 3
@inline field_dim(::Type{<:IF}) = 2
@inline field_dim(::Type{<:VF}) = 2
@inline field_dim(::Type{<:VIJFH}) = 4
@inline field_dim(::Type{<:VIFH}) = 3

@inline to_data_specific(::IJF, I::CartesianIndex) = CartesianIndex(I.I[1], I.I[2], 1, 1)
@inline to_data_specific(::IJFH, I::CartesianIndex) = CartesianIndex(I.I[1], I.I[2], 1, I.I[5])
@inline to_data_specific(::IFH, I::CartesianIndex) = CartesianIndex(I.I[1], 1, I.I[5])
@inline to_data_specific(::VIJFH, I::CartesianIndex) = CartesianIndex(I.I[4], I.I[1], I.I[2], 1, I.I[5])
@inline to_data_specific(::VIFH, I::CartesianIndex) = CartesianIndex(I.I[4], I.I[1], 1, I.I[5])
@inline to_data_specific(::DataSlab1D, I::CartesianIndex) = CartesianIndex(I.I[1], I.I[1], 1, I.I[5])

"""
    type_params(data::AbstractData)
    type_params(::Type{<:AbstractData})

This is an internal function, please do not use outside of ClimaCore.

Returns the type parameters for the given datalayout,
exluding the backing array type.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline type_params(data::AbstractData) = type_params(typeof(data))
@inline type_params(::Type{IJKFVH{S, Nij, Nk, Nv, Nh, A}}) where {S, Nij, Nk, Nv, Nh, A} = (S, Nij, Nk, Nv, Nh)
@inline type_params(::Type{IJFH{S, Nij, Nh, A}}) where {S, Nij, Nh, A} = (S, Nij, Nh)
@inline type_params(::Type{IFH{S, Ni, Nh, A}}) where {S, Ni, Nh, A} = (S, Ni, Nh)
@inline type_params(::Type{DataF{S, A}}) where {S, A} = (S,)
@inline type_params(::Type{IJF{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IF{S, Ni, A}}) where {S, Ni, A} = (S, Ni)
@inline type_params(::Type{VF{S, Nv, A}}) where {S, Nv, A} = (S, Nv)
@inline type_params(::Type{VIJFH{S, Nv, Nij, Nh, A}}) where {S, Nv, Nij, Nh, A} = (S, Nv, Nij, Nh)
@inline type_params(::Type{VIFH{S, Nv, Ni, Nh, A}}) where {S, Nv, Ni, Nh, A} = (S, Nv, Ni, Nh)
@inline type_params(::Type{IH1JH2{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IV1JH2{S, n1, Ni, A}}) where {S, n1, Ni, A} = (S, n1, Ni)

"""
    union_all(data::AbstractData)
    union_all(::Type{<:AbstractData})

This is an internal function, please do not use outside of ClimaCore.

Returns the UnionAll type of `data::AbstractData`. For
example, `union_all(::DataF{Float64})` will return `DataF`.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline union_all(data::AbstractData) = union_all(typeof(data))
@inline union_all(::Type{<:IJKFVH}) = IJKFVH
@inline union_all(::Type{<:IJFH}) = IJFH
@inline union_all(::Type{<:IFH}) = IFH
@inline union_all(::Type{<:DataF}) = DataF
@inline union_all(::Type{<:IJF}) = IJF
@inline union_all(::Type{<:IF}) = IF
@inline union_all(::Type{<:VF}) = VF
@inline union_all(::Type{<:VIJFH}) = VIJFH
@inline union_all(::Type{<:VIFH}) = VIFH
@inline union_all(::Type{<:IH1JH2}) = IH1JH2
@inline union_all(::Type{<:IV1JH2}) = IV1JH2

@inline slab_index(i, j) = CartesianIndex(i, j, 1, 1, 1)
@inline slab_index(i) = CartesianIndex(i, 1, 1, 1, 1)
@inline vindex(v) = CartesianIndex(1, 1, 1, v, 1)
#! format: on

Base.ndims(data::AbstractData) = Base.ndims(typeof(data))
Base.ndims(::Type{T}) where {T <: AbstractData} =
    Base.ndims(parent_array_type(T))

"""
    data2array(::AbstractData)

Reshapes the DataLayout's parent array into a `Vector`, or (for DataLayouts with vertical levels)
`Nv x N` matrix, where `Nv` is the number of vertical levels and `N` is the remaining dimensions.

The dimensions of the resulting array are
 - `([number of vertical nodes], number of horizontal nodes)`.

Also, this assumes that `eltype(data) <: Real`.
"""
function data2array end

data2array(data::Union{IF, IFH}) = reshape(parent(data), :)
data2array(data::Union{IJF, IJFH}) = reshape(parent(data), :)
data2array(data::Union{VF{S, Nv}, VIFH{S, Nv}, VIJFH{S, Nv}}) where {S, Nv} =
    reshape(parent(data), Nv, :)

"""
    array2data(array, ::AbstractData)

Reshapes `array` (of scalars) to fit into the given `DataLayout`.

The dimensions of `array` are assumed to be
 - `([number of vertical nodes], number of horizontal nodes)`.
"""
function array2data end

array2data(array::AbstractArray{T, 1}, ::IF{<:Any, Ni}) where {T, Ni} =
    IF{T, Ni}(reshape(array, Ni, 1))
array2data(array::AbstractArray{T, 1}, ::IFH{<:Any, Ni, Nh}) where {T, Ni, Nh} =
    IFH{T, Ni, Nh}(reshape(array, Ni, 1, Nh))
array2data(array::AbstractArray{T, 1}, ::IJF{<:Any, Nij}) where {T, Nij} =
    IJF{T, Nij}(reshape(array, Nij, Nij, 1))
array2data(
    array::AbstractArray{T, 1},
    ::IJFH{<:Any, Nij, Nh},
) where {T, Nij, Nh} = IJFH{T, Nij, Nh}(reshape(array, Nij, Nij, 1, Nh))
array2data(array::AbstractArray{T, 2}, ::VF{<:Any, Nv}) where {T, Nv} =
    VF{T, Nv}(reshape(array, Nv, 1))
array2data(
    array::AbstractArray{T, 2},
    ::VIFH{<:Any, Nv, Ni, Nh},
) where {T, Nv, Ni, Nh} = VIFH{T, Nv, Ni, Nh}(reshape(array, Nv, Ni, 1, Nh))
array2data(
    array::AbstractArray{T, 2},
    ::VIJFH{<:Any, Nv, Nij, Nh},
) where {T, Nv, Nij, Nh} =
    VIJFH{T, Nv, Nij, Nh}(reshape(array, Nv, Nij, Nij, 1, Nh))

"""
    device_dispatch(data::AbstractData)

Returns an `ToCPU` or a `ToCUDA` for CPU
and CUDA-backed arrays accordingly.
"""
device_dispatch(dest::AbstractData) = _device_dispatch(dest)

_device_dispatch(x::Array) = ToCPU()
_device_dispatch(x::SubArray) = _device_dispatch(parent(x))
_device_dispatch(x::Base.ReshapedArray) = _device_dispatch(parent(x))
_device_dispatch(x::AbstractData) = _device_dispatch(parent(x))
_device_dispatch(x::SArray) = ToCPU()
_device_dispatch(x::MArray) = ToCPU()

include("copyto.jl")
include("fused_copyto.jl")
include("fill.jl")
include("mapreduce.jl")
include("cartesian_index.jl")
include("has_uniform_datalayouts.jl")

end # module
