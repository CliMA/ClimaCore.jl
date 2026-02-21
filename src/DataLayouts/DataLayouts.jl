"""
    ClimaCore.DataLayouts

Defines the following DataLayouts (see individual docs for more info):

TODO: Add links to these datalayouts

 - `IJKFVH`
 - `IJFH`
 - `IJHF`
 - `IFH`
 - `IHF`
 - `DataF`
 - `IJF`
 - `IF`
 - `VF`
 - `VIJFH`
 - `VIJHF`
 - `VIFH`
 - `VIHF`
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


## Datalayouts that end with the field index

One of the fundamental features of datalayouts is to be able to
store multiple variables in the same array, and then access
those variables by name. As such, we occasionally must index into
multiple variables when performing operations with a datalayout.

We can efficiently support linear indexing with datalayouts
whose field index (`f`) is first or last. This is for the same reason
as https://docs.julialang.org/en/v1/devdocs/subarrays/#Linear-indexing:

    Linear indexing can be implemented efficiently when the entire array
    has a single stride that separates successive elements, starting from
    some offset.

Therefore, we provide special handling for these datalayouts where possible
to leverage efficient linear indexing.

Here are some references containing relevant discussions and efforts to
leverage efficient linear indexing:
 - https://github.com/CliMA/ClimaCore.jl/issues/1889
 - https://github.com/JuliaLang/julia/issues/28126
 - https://github.com/JuliaLang/julia/issues/32051
 - https://github.com/maleadt/StaticCartesian.jl
 - https://github.com/JuliaGPU/GPUArrays.jl/pull/454#issuecomment-1431575721
 - https://github.com/JuliaGPU/GPUArrays.jl/pull/520
 - https://github.com/JuliaGPU/GPUArrays.jl/pull/464
"""
module DataLayouts

import Base: Base, @propagate_inbounds
import StaticArrays: SOneTo, MArray, SArray
import ClimaComms
import MultiBroadcastFusion as MBF
import Adapt
using UnrolledUtilities

import ..Utilities: PlusHalf, unionall_type, inferred_type, broadcast_eltype
import ..Utilities:
    is_auto_broadcastable, enable_auto_broadcasting, disable_auto_broadcasting
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..slab, ..slab_args, ..column, ..column_args, ..level, ..level_args
export slab,
    column,
    level,
    IJFH,
    IJHF,
    IJF,
    IFH,
    IHF,
    IF,
    VF,
    VIJFH,
    VIJHF,
    VIFH,
    VIHF,
    DataF

# Internal types for managing CPU/GPU dispatching
abstract type AbstractDispatchToDevice end
struct ToCPU <: AbstractDispatchToDevice end
struct ToCUDA <: AbstractDispatchToDevice end

"""
    AbstractMask

An abstract mask type, for marking domains as present or absent.
"""
abstract type AbstractMask end

"""
    NoMask

A lazy non-mask type (default), used to indicate that an entire domain is
present.
"""
struct NoMask <: AbstractMask end

"""
    IJHMask

A mask type, used to house and use information for which columns
in a grid are active.
 - `is_active` a bool mask, the size of the entire grid, that indicates if the
   column is active or not
 - `N` a Int array with 1 element, containing number of active columns
 - `i_map` a Int array, containing i-indices of active columns
 - `j_map` a Int array, containing j-indices of active columns
 - `h_map` a Int array, containing h-indices of active columns
"""
struct IJHMask{B, NT, V} <: AbstractMask
    is_active::B
    N::NT
    i_map::V
    j_map::V
    h_map::V
end

include("struct.jl")

abstract type AbstractData{S} end

@inline Base.size(data::AbstractData, i::Integer) = size(data)[i]
@inline Base.size(data::AbstractData) = universal_size(data)

"""
    struct UniversalSize{Ni, Nj, Nv} end
    UniversalSize(data::AbstractData)

A struct containing static dimensions (except `Nh`),
universal to all datalayouts:

 - `Ni` number of spectral element nodal degrees of freedom in first horizontal direction
 - `Nj` number of spectral element nodal degrees of freedom in second horizontal direction
 - `Nv` number of vertical degrees of freedom
 - `Nh` number of horizontal elements

Note that this dynamically allocates a new type.
"""
struct UniversalSize{Ni, Nj, Nv, T}
    Nh::T
end

@inline function UniversalSize(data::AbstractData)
    us = universal_size(data)
    UniversalSize{us[1], us[2], us[4], typeof(us[5])}(us[5])
end

@inline array_length(data::AbstractData) = prod(size(parent(data)))

"""
    (Ni, Nj, _, Nv, Nh) = universal_size(data::AbstractData)

A tuple of compile-time known type parameters,
corresponding to `UniversalSize`. The field dimension
is excluded and is returned as 1.
"""
@inline universal_size(us::UniversalSize{Ni, Nj, Nv}) where {Ni, Nj, Nv} =
    (Ni, Nj, 1, Nv, us.Nh)

"""
    get_N(::UniversalSize) # static
    get_N(::AbstractData) # dynamic

Statically returns `prod((Ni, Nj, Nv, Nh))`
"""
@inline get_N(us::UniversalSize{Ni, Nj, Nv}) where {Ni, Nj, Nv} =
    prod((Ni, Nj, Nv, us.Nh))

"""
    get_Nv(::UniversalSize) # static
    get_Nv(::AbstractData) # dynamic

Statically returns `Nv`.
"""
@inline get_Nv(::UniversalSize{Ni, Nj, Nv}) where {Ni, Nj, Nv} = Nv

"""
    get_Nij(::UniversalSize) # static
    get_Nij(::AbstractData) # dynamic

Statically returns `Nij`.
"""
@inline get_Nij(::UniversalSize{Nij}) where {Nij} = Nij

"""
    get_Nh(::UniversalSize) # dynamic
    get_Nh(::AbstractData) # dynamic

Returns `Nh`.
"""
@inline get_Nh(us::UniversalSize{Ni, Nj, Nv}) where {Ni, Nj, Nv} = us.Nh

# TODO: inline so we don't overspecialize on these helpers
@inline get_Nh_dynamic(data::AbstractData) =
    size(parent(data), h_dim(singleton(data)))
@inline get_Nh(data::AbstractData) = get_Nh(UniversalSize(data))
@inline get_Nij(data::AbstractData) = get_Nij(UniversalSize(data))
@inline get_Nv(data::AbstractData) = get_Nv(UniversalSize(data))
@inline get_N(data::AbstractData) = get_N(UniversalSize(data))

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

@inline Base.:(==)(data1::D, data2::D) where {D <: AbstractData} =
    parent(data1) == parent(data2)

@inline function ncomponents(data::AbstractData{S}) where {S}
    typesize(eltype(parent(data)), S)
end

@inline _getproperty(data::AbstractData, ::Val{Name}) where {Name} =
    _getproperty(data, Val(Name), Name)

@generated function _getproperty(
    data::AbstractData{S},
    ::Val{Name},
    name,
) where {S, Name}
    i = findfirst(isequal(Name), fieldnames(S))
    static_idx = Val{i}()
    return :(
        Base.@_inline_meta; DataLayouts._property_view(data, $static_idx, name)
    )
end

@inline function Base.getproperty(data::AbstractData{S}, name::Symbol) where {S}
    _getproperty(data, Val{name}(), name)
end
@inline function Base.dotgetproperty(
    data::AbstractData{S},
    name::Symbol,
) where {S}
    _getproperty(data, Val{name}(), name)
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
    fdim = field_dim(singleton(data))
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ndims(data) - fdim))
    dataview =
        @inbounds view(array, Ipre..., (offset + 1):(offset + nbytes), Ipost...)
    union_all(singleton(data)){SS, Base.tail(type_params(data))...}(dataview)
end

@noinline _property_view(
    data::AbstractData{S},
    ::Val{nothing},
    name,
) where {S} = error("Invalid field name `$name` for type `$(S)`.")

# In the past, we've sometimes needed a generated function
# for inference and constant propagation:
Base.@propagate_inbounds @generated function _property_view(
    data::AD,
    ::Val{Idx},
    name,
) where {S, Idx, AD <: AbstractData{S}}
    SS = fieldtype(S, Idx)
    T = eltype(parent_array_type(AD))
    offset = fieldtypeoffset(T, S, Val(Idx))
    nbytes = typesize(T, SS)
    fdim = field_dim(AD.name.wrapper)
    Ipre = ntuple(i -> Colon(), Val(fdim - 1))
    Ipost = ntuple(i -> Colon(), Val(ndims(data) - fdim))
    field_byterange = (offset + 1):(offset + nbytes)
    return :($(AD.name.wrapper){$SS, $(Base.tail(type_params(AD)))...}(
        @inbounds view(parent(data), $Ipre..., $field_byterange, $Ipost...)
    ))
end

function replace_basetype(data::AbstractData{S}, ::Type{T}) where {S, T}
    array = parent(data)
    S′ = replace_basetype(eltype(array), T, S)
    return union_all(singleton(data)){S′, Base.tail(type_params(data))...}(
        similar(array, T),
    )
end

maybe_populate!(array, ::typeof(similar)) = nothing
maybe_populate!(array, ::typeof(ones)) = fill!(array, 1)
maybe_populate!(array, ::typeof(zeros)) = fill!(array, 0)
function maybe_populate!(array, ::typeof(rand))
    parent(array) .= typeof(array)(rand(eltype(array), size(array)))
end

# ================== Singletons

# These types mirror datalayouts, which
# we use to help reduce overspecialization
abstract type AbstractDataSingleton end
struct IJKFVHSingleton <: AbstractDataSingleton end
struct IJFHSingleton <: AbstractDataSingleton end
struct IJHFSingleton <: AbstractDataSingleton end
struct IFHSingleton <: AbstractDataSingleton end
struct IHFSingleton <: AbstractDataSingleton end
struct DataFSingleton <: AbstractDataSingleton end
struct IJFSingleton <: AbstractDataSingleton end
struct IFSingleton <: AbstractDataSingleton end
struct VFSingleton <: AbstractDataSingleton end
struct VIJFHSingleton <: AbstractDataSingleton end
struct VIJHFSingleton <: AbstractDataSingleton end
struct VIFHSingleton <: AbstractDataSingleton end
struct VIHFSingleton <: AbstractDataSingleton end
struct IH1JH2Singleton <: AbstractDataSingleton end
struct IV1JH2Singleton <: AbstractDataSingleton end

# ==================
# Data3D DataLayout
# ==================

"""
    IJKFVH{S, Nij, Nk}(array::AbstractArray{T, 6}) <: Data3D{S, Nij, Nk}

A 3D DataLayout. TODO: Add more docs

    IJKFVH{S}(ArrayType[, ones | zeros | rand]; Nij, Nk, Nv)

The keyword constructor returns a `IJKFVH` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom
 - `Nk` Number of vertical nodes within an element
 - `Nij` quadrature degrees of freedom per horizontal direction

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IJKFVH{S, Nij, Nk, Nv, A} <: Data3D{S, Nij, Nk}
    array::A
end

function IJKFVH{S, Nij, Nk, Nv}(
    array::AbstractArray{T, 6},
) where {S, Nij, Nk, Nv, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == Nk
    @assert size(array, 4) == typesize(T, S)
    @assert size(array, 5) == Nv
    IJKFVH{S, Nij, Nk, Nv, typeof(array)}(array)
end

function IJKFVH{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
    Nij::Integer,
    Nk::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nij, Nij, Nk, Nf, Nv, Nh)
    maybe_populate!(array, fun)
    IJKFVH{S, Nij, Nk, Nv}(array)
end

@inline universal_size(data::IJKFVH{S, Nij, Nk, Nv}) where {S, Nij, Nk, Nv} =
    (Nij, Nij, Nk, Nv, get_Nh_dynamic(data))

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

    IJFH{S}(ArrayType[, Base.ones | zeros | rand]; Nij, Nh)

The keyword constructor returns a `IJFH` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nij` quadrature degrees of freedom per horizontal direction
 - `Nh` number of mesh elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IJFH{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IJFH{S, Nij}(array::AbstractArray{T, 4}) where {S, Nij, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 3) == typesize(T, S)
    IJFH{S, Nij, typeof(array)}(array)
end

function IJFH{S}(
    ::Type{ArrayType},
    fun = similar;
    Nij::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nij, Nij, Nf, Nh)
    maybe_populate!(array, fun)
    IJFH{S, Nij}(array)
end

@inline universal_size(data::IJFH{S, Nij}) where {S, Nij} =
    (Nij, Nij, 1, 1, get_Nh_dynamic(data))

function IJFH{S, Nij}(::Type{ArrayType}, Nh::Integer) where {S, Nij, ArrayType}
    T = eltype(ArrayType)
    IJFH{S, Nij}(ArrayType(undef, Nij, Nij, typesize(T, S), Nh))
end

Base.length(data::IJFH) = get_Nh_dynamic(data)

Base.@propagate_inbounds slab(data::IJFH, h::Integer) = slab(data, 1, h)

@inline function slab(data::IJFH{S, Nij}, v::Integer, h::Integer) where {S, Nij}
    @boundscheck (v >= 1 && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), :, :, :, h)
    IJF{S, Nij}(dataview)
end

@inline function column(data::IJFH{S, Nij}, i, j, h) where {S, Nij}
    @boundscheck (
        1 <= j <= Nij && 1 <= i <= Nij && 1 <= h <= get_Nh_dynamic(data)
    ) || throw(BoundsError(data, (i, j, h)))
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

"""
    IJHF{S, Nij, A} <: Data2D{S, Nij}
    IJHF{S,Nij}(ArrayType, nelements)


Backing `DataLayout` for 2D spectral element slabs.

Element nodal point (I,J) data is contiguous for each datatype `S` struct field (F),
for each 2D mesh element slab (H).

The `ArrayType`-constructor constructs a IJHF 2D Spectral
DataLayout given the backing `ArrayType`, quadrature degrees
of freedom `Nij × Nij`, and the number of mesh elements `nelements`.

    IJHF{S}(ArrayType[, Base.ones | zeros | rand]; Nij, Nh)

The keyword constructor returns a `IJHF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nij` quadrature degrees of freedom per horizontal direction
 - `Nh` number of mesh elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IJHF{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IJHF{S, Nij}(array::AbstractArray{T, 4}) where {S, Nij, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    @assert size(array, 4) == typesize(T, S)
    IJHF{S, Nij, typeof(array)}(array)
end

function IJHF{S}(
    ::Type{ArrayType},
    fun = similar;
    Nij::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nij, Nij, Nh, Nf)
    maybe_populate!(array, fun)
    IJHF{S, Nij}(array)
end

@inline universal_size(data::IJHF{S, Nij}) where {S, Nij} =
    (Nij, Nij, 1, 1, get_Nh_dynamic(data))

function IJHF{S, Nij}(::Type{ArrayType}, Nh::Integer) where {S, Nij, ArrayType}
    T = eltype(ArrayType)
    IJHF{S, Nij}(ArrayType(undef, Nij, Nij, Nh, typesize(T, S)))
end

Base.length(data::IJHF) = get_Nh_dynamic(data)

Base.@propagate_inbounds slab(data::IJHF, h::Integer) = slab(data, 1, h)

@inline function slab(data::IJHF{S, Nij}, v::Integer, h::Integer) where {S, Nij}
    @boundscheck (v >= 1 && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (v, h)))
    dataview = @inbounds view(parent(data), :, :, h, :)
    IJF{S, Nij}(dataview)
end

@inline function column(data::IJHF{S, Nij}, i, j, h) where {S, Nij}
    @boundscheck (
        1 <= j <= Nij && 1 <= i <= Nij && 1 <= h <= get_Nh_dynamic(data)
    ) || throw(BoundsError(data, (i, j, h)))
    dataview = @inbounds view(parent(data), i, j, h, :)
    DataF{S}(dataview)
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::IJHF{S, Nij},
) where {S, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        IJHF{S, Nij}(gatherdata)
    else
        nothing
    end
end

# ==================
# Data1D DataLayout
# ==================

Base.length(data::Data1D) = get_Nh_dynamic(data)

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

    IFH{S}(ArrayType[, ones | zeros | rand]; Ni, Nh)

The keyword constructor returns a `IFH` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Ni` quadrature degrees of freedom in the horizontal direction
 - `Nh` number of mesh elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IFH{S, Ni, A} <: Data1D{S, Ni}
    array::A
end

function IFH{S, Ni}(array::AbstractArray{T, 3}) where {S, Ni, T}
    check_basetype(T, S)
    @assert size(array, 1) == Ni
    @assert size(array, 2) == typesize(T, S)
    IFH{S, Ni, typeof(array)}(array)
end

function IFH{S}(
    ::Type{ArrayType},
    fun = similar;
    Ni::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Ni, Nf, Nh)
    maybe_populate!(array, fun)
    IFH{S, Ni}(array)
end

function IFH{S, Ni}(::Type{ArrayType}, Nh::Integer) where {S, Ni, ArrayType}
    T = eltype(ArrayType)
    IFH{S, Ni}(ArrayType(undef, Ni, typesize(T, S), Nh))
end

@inline universal_size(data::IFH{S, Ni}) where {S, Ni} =
    (Ni, 1, 1, 1, get_Nh_dynamic(data))

@inline function slab(data::IFH{S, Ni}, h::Integer) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, :, h)
    IF{S, Ni}(dataview)
end
Base.@propagate_inbounds slab(data::IFH, v::Integer, h::Integer) = slab(data, h)

@inline function column(data::IFH{S, Ni}, i, h) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh_dynamic(data) && 1 <= i <= Ni) ||
                 throw(BoundsError(data, (i, h)))
    dataview = @inbounds view(parent(data), i, :, h)
    DataF{S}(dataview)
end
Base.@propagate_inbounds column(data::IFH{S, Ni}, i, j, h) where {S, Ni} =
    column(data, i, h)

"""
    IHF{S,Ni,Nh,A} <: Data1D{S, Ni}
    IHF{S,Ni,Nh}(ArrayType)

Backing `DataLayout` for 1D spectral element slabs.

Element nodal point (I) data is contiguous for each
datatype `S` struct field (F), for each 1D mesh element (H).


The `ArrayType`-constructor makes a IHF 1D Spectral
DataLayout given the backing `ArrayType`, quadrature
degrees of freedom `Ni`, and the number of mesh elements
`Nh`.

    IHF{S}(ArrayType[, ones | zeros | rand]; Ni, Nh)

The keyword constructor returns a `IHF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Ni` quadrature degrees of freedom in the horizontal direction
 - `Nh` number of mesh elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IHF{S, Ni, A} <: Data1D{S, Ni}
    array::A
end

function IHF{S, Ni}(array::AbstractArray{T, 3}) where {S, Ni, T}
    check_basetype(T, S)
    @assert size(array, 1) == Ni
    @assert size(array, 3) == typesize(T, S)
    IHF{S, Ni, typeof(array)}(array)
end

function IHF{S}(
    ::Type{ArrayType},
    fun = similar;
    Ni::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Ni, Nh, Nf)
    maybe_populate!(array, fun)
    IHF{S, Ni}(array)
end

function IHF{S, Ni}(::Type{ArrayType}, Nh::Integer) where {S, Ni, ArrayType}
    T = eltype(ArrayType)
    IHF{S, Ni}(ArrayType(undef, Ni, Nh, typesize(T, S)))
end

@inline universal_size(data::IHF{S, Ni}) where {S, Ni} =
    (Ni, 1, 1, 1, get_Nh_dynamic(data))

@inline function slab(data::IHF{S, Ni}, h::Integer) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (h,)))
    dataview = @inbounds view(parent(data), :, h, :)
    IF{S, Ni}(dataview)
end
Base.@propagate_inbounds slab(data::IHF, v::Integer, h::Integer) = slab(data, h)

@inline function column(data::IHF{S, Ni}, i, h) where {S, Ni}
    @boundscheck (1 <= h <= get_Nh_dynamic(data) && 1 <= i <= Ni) ||
                 throw(BoundsError(data, (i, h)))
    dataview = @inbounds view(parent(data), i, h, :)
    DataF{S}(dataview)
end
Base.@propagate_inbounds column(data::IHF{S, Ni}, i, j, h) where {S, Ni} =
    column(data, i, h)

# ======================
# Data0D DataLayout
# ======================

Base.length(data::Data0D) = 1
@inline universal_size(::Data0D) = (1, 1, 1, 1, 1)

"""
    DataF{S, A} <: Data0D{S}

Backing `DataLayout` for 0D point data.

    DataF{S}(ArrayType[, ones | zeros | rand])

The `ArrayType` constructor returns a `DataF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`).
"""
struct DataF{S, A} <: Data0D{S}
    array::A
end

function DataF{S}(array::AbstractVector{T}) where {S, T}
    check_basetype(T, S)
    @assert size(array, 1) == typesize(T, S)
    DataF{S, typeof(array)}(array)
end

function DataF{S}(::Type{ArrayType}, fun = similar;) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nf)
    maybe_populate!(array, fun)
    DataF{S}(array)
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
        Val(field_dim(singleton(data))),
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
        Val(field_dim(singleton(data))),
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

Base.@propagate_inbounds slab(data::DataSlab2D, h) = slab(data, 1, h)

@inline function slab(data::DataSlab2D, v, h)
    @boundscheck (v >= 1 && h >= 1) || throw(BoundsError(data, (v, h)))
    data
end

"""
    IJF{S, Nij, A} <: DataSlab2D{S, Nij}

Backing `DataLayout` for 2D spectral element slab data.

Nodal element data (I,J) are contiguous for each `S` datatype struct field (F) for a single element slab.

A `DataSlab2D` view can be returned from other `Data2D` objects by calling `slab(data, idx...)`.

    IJF{S}(ArrayType[, ones | zeros | rand]; Nij)

The keyword constructor returns a `IJF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nij` quadrature degrees of freedom per horizontal direction

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IJF{S, Nij, A} <: DataSlab2D{S, Nij}
    array::A
end

function IJF{S, Nij}(array::AbstractArray{T, 3}) where {S, Nij, T}
    @assert size(array, 1) == Nij
    @assert size(array, 2) == Nij
    check_basetype(T, S)
    @assert size(array, 3) == typesize(T, S)
    IJF{S, Nij, typeof(array)}(array)
end

function IJF{S}(
    ::Type{ArrayType},
    fun = similar;
    Nij::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nij, Nij, Nf)
    maybe_populate!(array, fun)
    IJF{S, Nij}(array)
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

Base.@propagate_inbounds slab(data::DataSlab1D, h) = slab(data, 1, h)

@inline function slab(data::DataSlab1D, v, h)
    @boundscheck (v >= 1 && h >= 1) || throw(BoundsError(data, (v, h)))
    data
end

"""
    IF{S, Ni, A} <: DataSlab1D{S, Ni}

Backing `DataLayout` for 1D spectral element slab data.

Nodal element data (I) are contiguous for each `S` datatype struct field (F) for a single element slab.

A `DataSlab1D` view can be returned from other `Data1D` objects by calling `slab(data, idx...)`.

    IF{S}(ArrayType[, ones | zeros | rand]; Ni)

The keyword constructor returns a `IF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Ni` quadrature degrees of freedom in the horizontal direction

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IF{S, Ni, A} <: DataSlab1D{S, Ni}
    array::A
end

function IF{S, Ni}(array::AbstractArray{T, 2}) where {S, Ni, T}
    @assert size(array, 1) == Ni
    check_basetype(T, S)
    @assert size(array, 2) == typesize(T, S)
    IF{S, Ni, typeof(array)}(array)
end

function IF{S}(
    ::Type{ArrayType},
    fun = similar;
    Ni::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Ni, Nf)
    maybe_populate!(array, fun)
    IF{S, Ni}(array)
end

function IF{S, Ni}(::Type{MArray}, ::Type{T}) where {S, Ni, T}
    Nf = typesize(T, S)
    array = MArray{Tuple{Ni, Nf}, T, 2, Ni * Nf}(undef)
    IF{S, Ni}(array)
end
function SArray(data::IF{S, Ni, <:MArray}) where {S, Ni}
    IF{S, Ni}(SArray(parent(data)))
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

    VF{S}(ArrayType[, ones | zeros | rand]; Nv)

The keyword constructor returns a `VF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct VF{S, Nv, A} <: DataColumn{S, Nv}
    array::A
end

function VF{S, Nv}(array::AbstractArray{T, 2}) where {S, Nv, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == typesize(T, S)
    VF{S, Nv, typeof(array)}(array)
end

function VF{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nv, Nf)
    maybe_populate!(array, fun)
    VF{S, Nv}(array)
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
    _property_view(data, Val(i), i)

Base.@propagate_inbounds column(data::VF, i, h) = column(data, i, 1, h)

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

    VIJFH{S}(ArrayType[, ones | zeros | rand]; Nv, Nij, Nh)

The keyword constructor returns a `VIJFH` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom
 - `Nij` quadrature degrees of freedom per horizontal direction
 - `Nh` number of horizontal elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct VIJFH{S, Nv, Nij, A} <: Data2DX{S, Nv, Nij}
    array::A
end

function VIJFH{S, Nv, Nij}(array::AbstractArray{T, 5}) where {S, Nv, Nij, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == size(array, 3) == Nij
    @assert size(array, 4) == typesize(T, S)
    VIJFH{S, Nv, Nij, typeof(array)}(array)
end

function VIJFH{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
    Nij::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nv, Nij, Nij, Nf, Nh)
    maybe_populate!(array, fun)
    VIJFH{S, Nv, Nij, typeof(array)}(array)
end

nlevels(::VIJFH{S, Nv}) where {S, Nv} = Nv

@inline universal_size(data::VIJFH{<:Any, Nv, Nij}) where {Nv, Nij} =
    (Nij, Nij, 1, Nv, get_Nh_dynamic(data))

Base.length(data::VIJFH) = get_Nv(data) * get_Nh_dynamic(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIJFH{S, Nv, Nij}, v, h) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= get_Nh_dynamic(data)) ||
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
@inline function column(data::VIJFH{S, Nv, Nij}, i, j, h) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (
        1 <= i <= Nij && 1 <= j <= Nij && 1 <= h <= get_Nh_dynamic(data)
    ) || throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, j, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIJFH{S, Nv, Nij}, v) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :, :)
    IJFH{S, Nij}(dataview)
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::VIJFH{S, Nv, Nij},
) where {S, Nv, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        VIJFH{S, Nv, Nij}(gatherdata)
    else
        nothing
    end
end

"""
    VIJHF{S, Nij, A} <: Data2DX{S, Nij}

Backing `DataLayout` for 2D spectral element slab + extruded 1D FV column data.

Column levels (V) are contiguous for every element nodal point (I, J)
for each `S` datatype struct field (F), for each 2D mesh element slab (H).

    VIJHF{S}(ArrayType[, ones | zeros | rand]; Nv, Nij, Nh)

The keyword constructor returns a `VIJHF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom
 - `Nij` quadrature degrees of freedom per horizontal direction
 - `Nh` number of horizontal elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct VIJHF{S, Nv, Nij, A} <: Data2DX{S, Nv, Nij}
    array::A
end

function VIJHF{S, Nv, Nij}(array::AbstractArray{T, 5}) where {S, Nv, Nij, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == size(array, 3) == Nij
    @assert size(array, 5) == typesize(T, S)
    VIJHF{S, Nv, Nij, typeof(array)}(array)
end

function VIJHF{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
    Nij::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nv, Nij, Nij, Nh, Nf)
    maybe_populate!(array, fun)
    VIJHF{S, Nv, Nij, typeof(array)}(array)
end

nlevels(::VIJHF{S, Nv}) where {S, Nv} = Nv

@inline universal_size(data::VIJHF{<:Any, Nv, Nij}) where {Nv, Nij} =
    (Nij, Nij, 1, Nv, get_Nh_dynamic(data))

Base.length(data::VIJHF) = get_Nv(data) * get_Nh_dynamic(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIJHF{S, Nv, Nij}, v, h) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (v, h)))
    Nf = ncomponents(data)
    dataview = @inbounds view(
        array,
        v,
        Base.Slice(Base.OneTo(Nij)),
        Base.Slice(Base.OneTo(Nij)),
        h,
        Base.Slice(Base.OneTo(Nf)),
    )
    IJF{S, Nij}(dataview)
end

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIJHF{S, Nv, Nij}, i, j, h) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (
        1 <= i <= Nij && 1 <= j <= Nij && 1 <= h <= get_Nh_dynamic(data)
    ) || throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, j, h, Base.Slice(Base.OneTo(Nf))),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIJHF{S, Nv, Nij}, v) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :, :)
    IJHF{S, Nij}(dataview)
end

function gather(
    ctx::ClimaComms.AbstractCommsContext,
    data::VIJHF{S, Nv, Nij},
) where {S, Nv, Nij}
    gatherdata = ClimaComms.gather(ctx, parent(data))
    if ClimaComms.iamroot(ctx)
        VIJHF{S, Nv, Nij}(gatherdata)
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

    VIFH{S}(ArrayType[, ones | zeros | rand]; Nv, Ni, Nh)

The keyword constructor returns a `VIFH` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom
 - `Ni` quadrature degrees of freedom in the horizontal direction
 - `Nh` number of horizontal elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct VIFH{S, Nv, Ni, A} <: Data1DX{S, Nv, Ni}
    array::A
end

function VIFH{S, Nv, Ni}(array::AbstractArray{T, 4}) where {S, Nv, Ni, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == Ni
    @assert size(array, 3) == typesize(T, S)
    VIFH{S, Nv, Ni, typeof(array)}(array)
end

function VIFH{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
    Ni::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nv, Ni, Nf, Nh)
    maybe_populate!(array, fun)
    VIFH{S, Nv, Ni, typeof(array)}(array)
end

nlevels(::VIFH{S, Nv}) where {S, Nv} = Nv

@inline universal_size(data::VIFH{<:Any, Nv, Ni}) where {Nv, Ni} =
    (Ni, 1, 1, Nv, get_Nh_dynamic(data))

Base.length(data::VIFH) = nlevels(data) * get_Nh_dynamic(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIFH{S, Nv, Ni}, v, h) where {S, Nv, Ni}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (v, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (v, Base.Slice(Base.OneTo(Ni)), Base.Slice(Base.OneTo(Nf)), h),
    )
    IF{S, Ni}(dataview)
end

Base.@propagate_inbounds column(data::VIFH, i, h) = column(data, i, 1, h)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIFH{S, Nv, Ni}, i, j, h) where {S, Nv, Ni}
    array = parent(data)
    @boundscheck (1 <= i <= Ni && j == 1 && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, Base.Slice(Base.OneTo(Nf)), h),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIFH{S, Nv, Nij}, v) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :)
    IFH{S, Nij}(dataview)
end

"""
    VIHF{S, Nv, Ni, A} <: Data1DX{S, Nv, Ni}

Backing `DataLayout` for 1D spectral element slab + extruded 1D FV column data.

Column levels (V) are contiguous for every element nodal point (I)
for each datatype `S` struct field (F), for each 1D mesh element slab (H).

    VIHF{S}(ArrayType[, ones | zeros | rand]; Nv, Ni, Nh)

The keyword constructor returns a `VIHF` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nv` number of vertical degrees of freedom
 - `Ni` quadrature degrees of freedom in the horizontal direction
 - `Nh` number of horizontal elements

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct VIHF{S, Nv, Ni, A} <: Data1DX{S, Nv, Ni}
    array::A
end

function VIHF{S, Nv, Ni}(array::AbstractArray{T, 4}) where {S, Nv, Ni, T}
    check_basetype(T, S)
    @assert size(array, 1) == Nv
    @assert size(array, 2) == Ni
    @assert size(array, 4) == typesize(T, S)
    VIHF{S, Nv, Ni, typeof(array)}(array)
end

function VIHF{S}(
    ::Type{ArrayType},
    fun = similar;
    Nv::Integer,
    Ni::Integer,
    Nh::Integer,
) where {S, ArrayType}
    Nf = typesize(eltype(ArrayType), S)
    array = similar(ArrayType, Nv, Ni, Nh, Nf)
    maybe_populate!(array, fun)
    VIHF{S, Nv, Ni, typeof(array)}(array)
end

nlevels(::VIHF{S, Nv}) where {S, Nv} = Nv

@inline universal_size(data::VIHF{<:Any, Nv, Ni}) where {Nv, Ni} =
    (Ni, 1, 1, Nv, get_Nh_dynamic(data))

Base.length(data::VIHF) = nlevels(data) * get_Nh_dynamic(data)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function slab(data::VIHF{S, Nv, Ni}, v, h) where {S, Nv, Ni}
    array = parent(data)
    @boundscheck (1 <= v <= Nv && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (v, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (v, Base.Slice(Base.OneTo(Ni)), h, Base.Slice(Base.OneTo(Nf))),
    )
    IF{S, Ni}(dataview)
end

Base.@propagate_inbounds column(data::VIHF, i, h) = column(data, i, 1, h)

# Note: construct the subarray view directly as optimizer fails in Base.to_indices (v1.7)
@inline function column(data::VIHF{S, Nv, Ni}, i, j, h) where {S, Nv, Ni}
    array = parent(data)
    @boundscheck (1 <= i <= Ni && j == 1 && 1 <= h <= get_Nh_dynamic(data)) ||
                 throw(BoundsError(data, (i, j, h)))
    Nf = ncomponents(data)
    dataview = @inbounds SubArray(
        array,
        (Base.Slice(Base.OneTo(Nv)), i, h, Base.Slice(Base.OneTo(Nf))),
    )
    VF{S, Nv}(dataview)
end

@inline function level(data::VIHF{S, Nv, Nij}, v) where {S, Nv, Nij}
    array = parent(data)
    @boundscheck (1 <= v <= Nv) || throw(BoundsError(data, (v,)))
    dataview = @inbounds view(array, v, :, :, :)
    IHF{S, Nij}(dataview)
end

# =========================================
# Special DataLayouts for regular gridding
# =========================================

"""
    IH1JH2{S, Nij}(data::AbstractMatrix{S})

Stores a 2D field in a matrix using a column-major format.
The primary use is for interpolation to a regular grid for ex. plotting / field output.

    IH1JH2{S}(ArrayType[, ones | zeros | rand]; Nij)

The keyword constructor returns a `IH1JH2` given
the `ArrayType` and (optionally) an initialization
method (one of `Base.ones`, `Base.zeros`, `Random.rand`)
and the keywords:
 - `Nij` quadrature degrees of freedom per horizontal direction

!!! note
    Objects made with the keyword constructor accept integer
    keyword inputs, so they are dynamically created. You may
    want to use a different constructor if you're making the
    object in a performance-critical section, and if you know
    the type parameters at compile time.
"""
struct IH1JH2{S, Nij, A} <: Data2D{S, Nij}
    array::A
end

function IH1JH2{S, Nij}(array::AbstractMatrix{S}) where {S, Nij}
    @assert size(array, 1) % Nij == 0
    @assert size(array, 2) % Nij == 0
    IH1JH2{S, Nij, typeof(array)}(array)
end

function IH1JH2{S}(
    ::Type{ArrayType},
    fun = similar;
    Nij::Integer,
) where {S, ArrayType}
    array = similar(ArrayType, 2 * Nij, 3 * Nij)
    maybe_populate!(array, fun)
    IH1JH2{S, Nij}(array)
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
    union_all(singleton(data)){type_params(data)...}(copy(parent(data)))

Base.reinterpret(::Type{S}, data::AbstractData) where {S} =
    union_all(singleton(data)){S, type_params(data)[2:end]...}(parent(data))

# broadcast machinery
include("non_extruded_broadcasted.jl")
include("broadcast.jl")

Adapt.adapt_structure(to, data::AbstractData{S}) where {S} =
    union_all(singleton(data)){type_params(data)...}(
        Adapt.adapt(to, parent(data)),
    )

rebuild(data::AbstractData, array::AbstractArray) =
    union_all(singleton(data)){type_params(data)...}(array)

empty_kernel_stats(::ClimaComms.AbstractDevice) = nothing
empty_kernel_stats() = empty_kernel_stats(ClimaComms.device())

# ==================
# Helpers
# ==================

#! format: off
@inline get_Nij(::IJKFVH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IJFH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IJHF{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::VIJFH{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::VIJHF{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::VIFH{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::VIHF{S, Nv, Nij}) where {S, Nv, Nij} = Nij
@inline get_Nij(::IFH{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IHF{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IJF{S, Nij}) where {S, Nij} = Nij
@inline get_Nij(::IF{S, Nij}) where {S, Nij} = Nij

"""
    field_dim(::AbstractDataSingleton)

This is an internal function, please do not use outside of ClimaCore.

Returns the field dimension in the backing array.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline field_dim(::IJKFVHSingleton) = 4
@inline field_dim(::IJFHSingleton) = 3
@inline field_dim(::IJHFSingleton) = 4
@inline field_dim(::IFHSingleton) = 2
@inline field_dim(::IHFSingleton) = 3
@inline field_dim(::DataFSingleton) = 1
@inline field_dim(::IJFSingleton) = 3
@inline field_dim(::IFSingleton) = 2
@inline field_dim(::VFSingleton) = 2
@inline field_dim(::VIJFHSingleton) = 4
@inline field_dim(::VIJHFSingleton) = 5
@inline field_dim(::VIFHSingleton) = 3
@inline field_dim(::VIHFSingleton) = 4

@inline field_dim(::Type{IJKFVH}) = 4
@inline field_dim(::Type{IJFH}) = 3
@inline field_dim(::Type{IJHF}) = 4
@inline field_dim(::Type{IFH}) = 2
@inline field_dim(::Type{IHF}) = 3
@inline field_dim(::Type{DataF}) = 1
@inline field_dim(::Type{IJF}) = 3
@inline field_dim(::Type{IF}) = 2
@inline field_dim(::Type{VF}) = 2
@inline field_dim(::Type{VIJFH}) = 4
@inline field_dim(::Type{VIJHF}) = 5
@inline field_dim(::Type{VIFH}) = 3
@inline field_dim(::Type{VIHF}) = 4

"""
    h_dim(::AbstractDataSingleton)

This is an internal function, please do not use outside of ClimaCore.

Returns the horizontal element dimension in the backing array.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline h_dim(::IJKFVHSingleton) = 5
@inline h_dim(::IJFHSingleton) = 4
@inline h_dim(::IJHFSingleton) = 3
@inline h_dim(::IFHSingleton) = 3
@inline h_dim(::IHFSingleton) = 2
@inline h_dim(::VIJFHSingleton) = 5
@inline h_dim(::VIJHFSingleton) = 4
@inline h_dim(::VIFHSingleton) = 4
@inline h_dim(::VIHFSingleton) = 3

@inline to_data_specific(::VFSingleton, I::Tuple) = (I[4], 1)
@inline to_data_specific(::IFSingleton, I::Tuple) = (I[1], 1)
@inline to_data_specific(::IJFSingleton, I::Tuple) = (I[1], I[2], 1)
@inline to_data_specific(::IJFHSingleton, I::Tuple) = (I[1], I[2], 1, I[5])
@inline to_data_specific(::IJHFSingleton, I::Tuple) = (I[1], I[2], I[5], 1)
@inline to_data_specific(::IFHSingleton, I::Tuple) = (I[1], 1, I[5])
@inline to_data_specific(::IHFSingleton, I::Tuple) = (I[1], I[5], 1)
@inline to_data_specific(::VIJFHSingleton, I::Tuple) = (I[4], I[1], I[2], 1, I[5])
@inline to_data_specific(::VIJHFSingleton, I::Tuple) = (I[4], I[1], I[2], I[5], 1)
@inline to_data_specific(::VIFHSingleton, I::Tuple) = (I[4], I[1], 1, I[5])
@inline to_data_specific(::VIHFSingleton, I::Tuple) = (I[4], I[1], I[5], 1)

@inline to_data_specific_field(::DataFSingleton, I::Tuple) = (I[3],)
@inline to_data_specific_field(::VFSingleton, I::Tuple) = (I[4], I[3])
@inline to_data_specific_field(::IFSingleton, I::Tuple) = (I[1], I[3])
@inline to_data_specific_field(::IJFSingleton, I::Tuple) = (I[1], I[2], I[3])
@inline to_data_specific_field(::IJFHSingleton, I::Tuple) = (I[1], I[2], I[3], I[5])
@inline to_data_specific_field(::IJHFSingleton, I::Tuple) = (I[1], I[2], I[5], I[3])
@inline to_data_specific_field(::IFHSingleton, I::Tuple) = (I[1], I[3], I[5])
@inline to_data_specific_field(::IHFSingleton, I::Tuple) = (I[1], I[5], I[3])
@inline to_data_specific_field(::VIJFHSingleton, I::Tuple) = (I[4], I[1], I[2], I[3], I[5])
@inline to_data_specific_field(::VIJHFSingleton, I::Tuple) = (I[4], I[1], I[2], I[5], I[3])
@inline to_data_specific_field(::VIFHSingleton, I::Tuple) = (I[4], I[1], I[3], I[5])
@inline to_data_specific_field(::VIHFSingleton, I::Tuple) = (I[4], I[1], I[5], I[3])

"""
    bounds_condition(data::AbstractData, I::Tuple)

Returns the condition used for `@boundscheck`
inside `getindex` with `CartesianIndex`s.
"""
@inline bounds_condition(data::AbstractData, I::CartesianIndex) = true # TODO: add more support
@inline bounds_condition(data::IJF, I::CartesianIndex) = (1 <= I.I[1] <= get_Nij(data) && 1 <= I.I[2] <= get_Nij(data))
@inline bounds_condition(data::VF, I::CartesianIndex) = 1 <= I.I[4] <= nlevels(data)
@inline bounds_condition(data::IF, I::CartesianIndex) = 1 <= I.I[1] <= get_Nij(data)

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
@inline type_params(::Type{IJKFVH{S, Nij, Nk, Nv, A}}) where {S, Nij, Nk, Nv, A} = (S, Nij, Nk, Nv)
@inline type_params(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IJHF{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IFH{S, Ni, A}}) where {S, Ni, A} = (S, Ni)
@inline type_params(::Type{IHF{S, Ni, A}}) where {S, Ni, A} = (S, Ni)
@inline type_params(::Type{DataF{S, A}}) where {S, A} = (S,)
@inline type_params(::Type{IJF{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IF{S, Ni, A}}) where {S, Ni, A} = (S, Ni)
@inline type_params(::Type{VF{S, Nv, A}}) where {S, Nv, A} = (S, Nv)
@inline type_params(::Type{VIJFH{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = (S, Nv, Nij)
@inline type_params(::Type{VIJHF{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = (S, Nv, Nij)
@inline type_params(::Type{VIFH{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = (S, Nv, Ni)
@inline type_params(::Type{VIHF{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = (S, Nv, Ni)
@inline type_params(::Type{IH1JH2{S, Nij, A}}) where {S, Nij, A} = (S, Nij)
@inline type_params(::Type{IV1JH2{S, n1, Ni, A}}) where {S, n1, Ni, A} = (S, n1, Ni)

"""
    union_all(data::AbstractData)
    union_all(singleton(::AbstractData))

This is an internal function, please do not use outside of ClimaCore.

Returns the UnionAll type of `data::AbstractData`. For
example, `union_all(::DataF{Float64})` will return `DataF`.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline union_all(::IJKFVHSingleton) = IJKFVH
@inline union_all(::IJFHSingleton) = IJFH
@inline union_all(::IJHFSingleton) = IJHF
@inline union_all(::IFHSingleton) = IFH
@inline union_all(::IHFSingleton) = IHF
@inline union_all(::DataFSingleton) = DataF
@inline union_all(::IJFSingleton) = IJF
@inline union_all(::IFSingleton) = IF
@inline union_all(::VFSingleton) = VF
@inline union_all(::VIJFHSingleton) = VIJFH
@inline union_all(::VIJHFSingleton) = VIJHF
@inline union_all(::VIFHSingleton) = VIFH
@inline union_all(::VIHFSingleton) = VIHF
@inline union_all(::IH1JH2Singleton) = IH1JH2
@inline union_all(::IV1JH2Singleton) = IV1JH2

"""
    array_size(data::AbstractData, [dim])
    array_size(::Type{<:AbstractData}, [dim])

This is an internal function, please do not use outside of ClimaCore.

Returns the size of the backing array, with the field dimension set to 1

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline array_size(data::AbstractData, i::Integer) = array_size(data)[i]
@inline array_size(data::IJKFVH{S, Nij, Nk, Nv}) where {S, Nij, Nk, Nv} = (Nij, Nij, Nk, 1, Nv, get_Nh_dynamic(data))
@inline array_size(data::IJFH{S, Nij}) where {S, Nij} = (Nij, Nij, 1, get_Nh_dynamic(data))
@inline array_size(data::IJHF{S, Nij}) where {S, Nij} = (Nij, Nij, get_Nh_dynamic(data), 1)
@inline array_size(data::IFH{S, Ni}) where {S, Ni} = (Ni, 1, get_Nh_dynamic(data))
@inline array_size(data::IHF{S, Ni}) where {S, Ni} = (Ni, get_Nh_dynamic(data), 1)
@inline array_size(data::DataF{S}) where {S} = (1,)
@inline array_size(data::IJF{S, Nij}) where {S, Nij} = (Nij, Nij, 1)
@inline array_size(data::IF{S, Ni}) where {S, Ni} = (Ni, 1)
@inline array_size(data::VF{S, Nv}) where {S, Nv} = (Nv, 1)
@inline array_size(data::VIJFH{S, Nv, Nij}) where {S, Nv, Nij} = (Nv, Nij, Nij, 1, get_Nh_dynamic(data))
@inline array_size(data::VIJHF{S, Nv, Nij}) where {S, Nv, Nij} = (Nv, Nij, Nij, get_Nh_dynamic(data), 1)
@inline array_size(data::VIFH{S, Nv, Ni}) where {S, Nv, Ni} = (Nv, Ni, 1, get_Nh_dynamic(data))
@inline array_size(data::VIHF{S, Nv, Ni}) where {S, Nv, Ni} = (Nv, Ni, get_Nh_dynamic(data), 1)

"""
    farray_size(data::AbstractData)

This is an internal function, please do not use outside of ClimaCore.

Returns the size of the backing array, including the field dimension

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline farray_size(data::AbstractData, i::Integer) = farray_size(data)[i]
@inline farray_size(data::IJKFVH{S, Nij, Nk, Nv}) where {S, Nij, Nk, Nv} = (Nij, Nij, Nk, ncomponents(data), Nv, get_Nh_dynamic(data))
@inline farray_size(data::IJFH{S, Nij}) where {S, Nij} = (Nij, Nij, ncomponents(data), get_Nh_dynamic(data))
@inline farray_size(data::IJHF{S, Nij}) where {S, Nij} = (Nij, Nij, get_Nh_dynamic(data), ncomponents(data))
@inline farray_size(data::IFH{S, Ni}) where {S, Ni} = (Ni, ncomponents(data), get_Nh_dynamic(data))
@inline farray_size(data::IHF{S, Ni}) where {S, Ni} = (Ni, get_Nh_dynamic(data), ncomponents(data))
@inline farray_size(data::DataF{S}) where {S} = (ncomponents(data),)
@inline farray_size(data::IJF{S, Nij}) where {S, Nij} = (Nij, Nij, ncomponents(data))
@inline farray_size(data::IF{S, Ni}) where {S, Ni} = (Ni, ncomponents(data))
@inline farray_size(data::VF{S, Nv}) where {S, Nv} = (Nv, ncomponents(data))
@inline farray_size(data::VIJFH{S, Nv, Nij}) where {S, Nv, Nij} = (Nv, Nij, Nij, ncomponents(data), get_Nh_dynamic(data))
@inline farray_size(data::VIJHF{S, Nv, Nij}) where {S, Nv, Nij} = (Nv, Nij, Nij, get_Nh_dynamic(data), ncomponents(data))
@inline farray_size(data::VIFH{S, Nv, Ni}) where {S, Nv, Ni} = (Nv, Ni, ncomponents(data), get_Nh_dynamic(data))
@inline farray_size(data::VIHF{S, Nv, Ni}) where {S, Nv, Ni} = (Nv, Ni, get_Nh_dynamic(data), ncomponents(data))

# Keep in sync with definition(s) in libs.
@inline slab_index(i::T, j::T) where {T} = CartesianIndex(i, j, T(1), T(1), T(1))
@inline slab_index(i::T) where {T} = CartesianIndex(i, T(1), T(1), T(1), T(1))
@inline vindex(v::T) where {T} = CartesianIndex(T(1), T(1), T(1), v, T(1))

"""
    parent_array_type(data::AbstractData)

This is an internal function, please do not use outside of ClimaCore.

Returns the the backing array type.

This function is helpful for writing generic
code, when reconstructing new datalayouts with new
type parameters.
"""
@inline parent_array_type(data::AbstractData) = parent_array_type(typeof(data))
# Equivalent to:
# @generated parent_array_type(::Type{A}) where {A <: AbstractData} = Tuple(A.parameters)[end]
@inline parent_array_type(::Type{IFH{S, Ni, A}}) where {S, Ni, A} = A
@inline parent_array_type(::Type{IHF{S, Ni, A}}) where {S, Ni, A} = A
@inline parent_array_type(::Type{DataF{S, A}}) where {S, A} = A
@inline parent_array_type(::Type{IJF{S, Nij, A}}) where {S, Nij, A} = A
@inline parent_array_type(::Type{IF{S, Ni, A}}) where {S, Ni, A} = A
@inline parent_array_type(::Type{VF{S, Nv, A}}) where {S, Nv, A} = A
@inline parent_array_type(::Type{VIJFH{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = A
@inline parent_array_type(::Type{VIJHF{S, Nv, Nij, A}}) where {S, Nv, Nij, A} = A
@inline parent_array_type(::Type{VIFH{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = A
@inline parent_array_type(::Type{VIHF{S, Nv, Ni, A}}) where {S, Nv, Ni, A} = A
@inline parent_array_type(::Type{IJFH{S, Nij, A}}) where {S, Nij, A} = A
@inline parent_array_type(::Type{IJHF{S, Nij, A}}) where {S, Nij, A} = A
@inline parent_array_type(::Type{IH1JH2{S, Nij, A}}) where {S, Nij, A} = A
@inline parent_array_type(::Type{IV1JH2{S, n1, Ni, A}}) where {S, n1, Ni, A} = A
@inline parent_array_type(::Type{IJKFVH{S, Nij, Nk, Nv, A}}) where {S, Nij, Nk, Nv, A} = A

#! format: on

Base.ndims(data::AbstractData) = Base.ndims(typeof(data))
Base.ndims(::Type{T}) where {T <: AbstractData} =
    Base.ndims(parent_array_type(T))

@inline function Base.getindex(
    data::Union{IJF, IJFH, IJHF, IFH, IHF, VIJFH, VIJHF, VIFH, VIHF, VF, IF},
    I::CartesianIndex,
)
    @boundscheck bounds_condition(data, I) || throw(BoundsError(data, I))
    s = singleton(data)
    return get_struct(
        parent(data),
        eltype(data),
        Val(field_dim(s)),
        CartesianIndex(to_data_specific(s, I.I)),
    )
end

@inline function Base.setindex!(
    data::Union{IJF, IJFH, IJHF, IFH, IHF, VIJFH, VIJHF, VIFH, VIHF, VF, IF},
    val,
    I::CartesianIndex,
)
    @boundscheck bounds_condition(data, I) || throw(BoundsError(data, I))
    s = singleton(data)
    @inbounds set_struct!(
        parent(data),
        convert(eltype(data), val),
        Val(field_dim(s)),
        CartesianIndex(to_data_specific(s, I.I)),
    )
end

"""
    CartesianFieldIndex{N} <: Base.AbstractCartesianIndex{N}

A CartesianIndex wrapper to dispatch `getindex` / `setindex!`
to call [`getindex_field`](@ref) and [`setindex_field!`](@ref)
for specific field variables in a datalayout.
"""
struct CartesianFieldIndex{N} <: Base.AbstractCartesianIndex{N}
    CI::CartesianIndex{N}
end
CartesianFieldIndex(I...) = CartesianFieldIndex(CartesianIndex(I...))

Base.ndims(::CartesianFieldIndex{N}) where {N} = N
Base.@propagate_inbounds Base.getindex(
    data::AbstractData,
    CI::CartesianFieldIndex,
) = getindex_field(data, CI.CI)
Base.@propagate_inbounds Base.setindex!(
    data::AbstractData,
    val::Real,
    CI::CartesianFieldIndex,
) = setindex_field!(data, val, CI.CI)

"""
    getindex_field(data, ci::CartesianIndex{5})

Returns the value of the data at universal index `ci`,
for the specific field `f` in the `CartesianIndex`.

The universal index order is `CartesianIndex(i, j, f, v, h)`, see
see the notation in [`DataLayouts`](@ref) for more information.
"""
@inline function getindex_field(
    data::Union{
        DataF,
        IJF,
        IJFH,
        IJHF,
        IFH,
        IHF,
        VIJFH,
        VIJHF,
        VIFH,
        VIHF,
        VF,
        IF,
    },
    I::CartesianIndex, # universal index
)
    @boundscheck bounds_condition(data, I) || throw(BoundsError(data, I))
    @inbounds Base.getindex(
        parent(data),
        CartesianIndex(to_data_specific_field(singleton(data), I.I)),
    )
end

"""
    setindex_field!(data, val::Real, ci::CartesianIndex{5})

Stores the value `val` of the data at universal index `ci`,
for the specific field `f` in the `CartesianIndex`.

The universal index order is `CartesianIndex(i, j, f, v, h)`, see
see the notation in [`DataLayouts`](@ref) for more information.
"""
@inline function setindex_field!(
    data::Union{
        DataF,
        IJF,
        IJFH,
        IJHF,
        IFH,
        IHF,
        VIJFH,
        VIJHF,
        VIFH,
        VIHF,
        VF,
        IF,
    },
    val::Real,
    I::CartesianIndex, # universal index
)
    @boundscheck bounds_condition(data, I) || throw(BoundsError(data, I))
    @inbounds Base.setindex!(
        parent(data),
        val,
        CartesianIndex(to_data_specific_field(singleton(data), I.I)),
    )
end

const EndsWithField{S} =
    Union{IJHF{S}, IHF{S}, IJF{S}, IF{S}, VF{S}, VIJHF{S}, VIHF{S}}

if VERSION ≥ v"1.11.0-beta"
    ### --------------- Support for multi-dimensional indexing
    # TODO: can we remove this? It's not needed for Julia 1.10,
    #       but seems needed in Julia 1.11.
    @inline Base.getindex(
        data::Union{
            IJF,
            IJFH,
            IJHF,
            IFH,
            IHF,
            VIJFH,
            VIJHF,
            VIFH,
            VIHF,
            VF,
            IF,
        },
        I::Vararg{Int, N},
    ) where {N} = Base.getindex(
        data,
        CartesianIndex(to_universal_index(singleton(data), I)),
    )

    @inline Base.setindex!(
        data::Union{
            IJF,
            IJFH,
            IJHF,
            IFH,
            IHF,
            VIJFH,
            VIJHF,
            VIFH,
            VIHF,
            VF,
            IF,
        },
        val,
        I::Vararg{Int, N},
    ) where {N} = Base.setindex!(
        data,
        val,
        CartesianIndex(to_universal_index(singleton(data), I)),
    )

    # Certain datalayouts support special indexing.
    # Like VF datalayouts with `getindex(::VF, v::Integer)`
    #! format: off
    @inline to_universal_index(::VFSingleton, I::NTuple{1, T}) where {T} =  (T(1), T(1), T(1), I[1], T(1))
    @inline to_universal_index(::IFSingleton, I::NTuple{1, T}) where {T} =  (I[1], T(1), T(1), T(1), T(1))
    @inline to_universal_index(::IFSingleton, I::NTuple{2, T}) where {T} =  (I[1], T(1), T(1), T(1), T(1))
    @inline to_universal_index(::IFSingleton, I::NTuple{3, T}) where {T} =  (I[1], T(1), T(1), T(1), T(1))
    @inline to_universal_index(::IFSingleton, I::NTuple{4, T}) where {T} =  (I[1], T(1), T(1), T(1), T(1))
    @inline to_universal_index(::IFSingleton, I::NTuple{5, T}) where {T} =  (I[1], T(1), T(1), T(1), T(1))
    @inline to_universal_index(::IJFSingleton, I::NTuple{2, T}) where {T} = (I[1], I[2], T(1), T(1), T(1))
    @inline to_universal_index(::IJFSingleton, I::NTuple{3, T}) where {T} = (I[1], I[2], T(1), T(1), T(1))
    @inline to_universal_index(::IJFSingleton, I::NTuple{4, T}) where {T} = (I[1], I[2], T(1), T(1), T(1))
    @inline to_universal_index(::IJFSingleton, I::NTuple{5, T}) where {T} = (I[1], I[2], T(1), T(1), T(1))
    @inline to_universal_index(::AbstractDataSingleton, I::NTuple{5}) = I
    #! format: on
    ### ---------------
else
    # Only support datalayouts that end with fields, since those
    # are the only layouts where we can efficiently compute the
    # strides.
    @propagate_inbounds function Base.getindex(
        data::EndsWithField{S},
        I::Integer,
    ) where {S}
        s_array = farray_size(data)
        @inbounds get_struct_linear(parent(data), S, I, s_array)
    end
    @propagate_inbounds function Base.setindex!(
        data::EndsWithField{S},
        val,
        I::Integer,
    ) where {S}
        s_array = farray_size(data)
        @inbounds set_struct_linear!(parent(data), convert(S, val), I, s_array)
    end

end

"""
    data2array(::AbstractData)

Reshapes the DataLayout's parent array into a `Vector`, or (for DataLayouts with vertical levels)
`Nv x N` matrix, where `Nv` is the number of vertical levels and `N` is the remaining dimensions.

The dimensions of the resulting array are
 - `([number of vertical nodes], number of horizontal nodes)`.

Also, this assumes that `eltype(data) <: Real`.
"""
function data2array end

data2array(data::DataF) = reshape(parent(data), :)
data2array(data::Union{IF, IFH, IHF}) = reshape(parent(data), :)
data2array(data::Union{IJF, IJFH, IJHF}) = reshape(parent(data), :)
data2array(
    data::Union{
        VF{S, Nv},
        VIFH{S, Nv},
        VIHF{S, Nv},
        VIJFH{S, Nv},
        VIJHF{S, Nv},
    },
) where {S, Nv} = reshape(parent(data), Nv, :)

"""
    array2data(array, ::AbstractData)

Reshapes `array` (of scalars) to fit into the given `DataLayout`.

The dimensions of `array` are assumed to be
 - `([number of vertical nodes], number of horizontal nodes)`.
"""
array2data(array::AbstractArray{T}, data::AbstractData) where {T} =
    union_all(singleton(data)){T, Base.tail(type_params(data))...}(
        reshape(array, array_size(data)...),
    )

"""
    device_dispatch(array::AbstractArray)

Returns an `ToCPU` or a `ToCUDA` for CPU
and CUDA-backed arrays accordingly.
"""
device_dispatch(x::AbstractArray) = ToCPU()
device_dispatch(x::Array) = ToCPU()
device_dispatch(x::SubArray) = device_dispatch(parent(x))
device_dispatch(x::Base.ReshapedArray) = device_dispatch(parent(x))
device_dispatch(x::AbstractData) = device_dispatch(parent(x))
device_dispatch(x::SArray) = ToCPU()
device_dispatch(x::MArray) = ToCPU()

@inline singleton(@nospecialize(::IJKFVH)) = IJKFVHSingleton()
@inline singleton(@nospecialize(::IJFH)) = IJFHSingleton()
@inline singleton(@nospecialize(::IJHF)) = IJHFSingleton()
@inline singleton(@nospecialize(::IFH)) = IFHSingleton()
@inline singleton(@nospecialize(::IHF)) = IHFSingleton()
@inline singleton(@nospecialize(::DataF)) = DataFSingleton()
@inline singleton(@nospecialize(::IJF)) = IJFSingleton()
@inline singleton(@nospecialize(::IF)) = IFSingleton()
@inline singleton(@nospecialize(::VF)) = VFSingleton()
@inline singleton(@nospecialize(::VIJFH)) = VIJFHSingleton()
@inline singleton(@nospecialize(::VIJHF)) = VIJHFSingleton()
@inline singleton(@nospecialize(::VIFH)) = VIFHSingleton()
@inline singleton(@nospecialize(::VIHF)) = VIHFSingleton()
@inline singleton(@nospecialize(::IH1JH2)) = IH1JH2Singleton()
@inline singleton(@nospecialize(::IV1JH2)) = IV1JH2Singleton()

@inline singleton(::Type{IJKFVH}) = IJKFVHSingleton()
@inline singleton(::Type{IJFH}) = IJFHSingleton()
@inline singleton(::Type{IJHF}) = IJHFSingleton()
@inline singleton(::Type{IFH}) = IFHSingleton()
@inline singleton(::Type{IHF}) = IHFSingleton()
@inline singleton(::Type{DataF}) = DataFSingleton()
@inline singleton(::Type{IJF}) = IJFSingleton()
@inline singleton(::Type{IF}) = IFSingleton()
@inline singleton(::Type{VF}) = VFSingleton()
@inline singleton(::Type{VIJFH}) = VIJFHSingleton()
@inline singleton(::Type{VIJHF}) = VIJHFSingleton()
@inline singleton(::Type{VIFH}) = VIFHSingleton()
@inline singleton(::Type{VIHF}) = VIHFSingleton()
@inline singleton(::Type{IH1JH2}) = IH1JH2Singleton()
@inline singleton(::Type{IV1JH2}) = IV1JH2Singleton()


include("has_uniform_datalayouts.jl")

include("copyto.jl")
include("fused_copyto.jl")
include("fill.jl")
include("mapreduce.jl")

include("struct_linear_indexing.jl")


"""
    set_mask_maps!(mask)

Sets the mask maps, such that the elements of the maps correspond
to active columns.
"""
function set_mask_maps! end

function set_mask_maps!(mask::IJHMask)
    (Ni, Nj, _, _, Nh) = size(mask.is_active)
    # This only happens during initialization, so let's just do this on the cpu:
    I = 1
    i_map = zeros(Int, length(mask.i_map))
    j_map = zeros(Int, length(mask.j_map))
    h_map = zeros(Int, length(mask.h_map))
    is_active = rebuild(mask.is_active, Array)
    # TODO: the order that this loop is performed is decoupled from correctness,
    #       but it can have a significant impact on runtime performance (on gpus).
    #       So, we should figure out a good way or heuristic to permute these arrays
    #       to maximize performance.
    @inbounds for h in 1:Nh, j in 1:Nj, i in 1:Ni
        CI = CartesianIndex(i, j, 1, 1, h)
        if is_active[CI]
            i_map[I] = i
            j_map[I] = j
            h_map[I] = h
            I += 1
        end
    end
    mask.N .= I - 1
    mask.i_map .= typeof(mask.i_map)(i_map)
    mask.j_map .= typeof(mask.j_map)(j_map)
    mask.h_map .= typeof(mask.h_map)(h_map)
    return nothing
end

"""
    ColumnMask(
        ::Type{FT},
        ::Type{horizontal_layout_type},
        ::Type{DA},
        ::Val{Nq},
        ::Val{Nh}
    )

Construct a column mask, given:
 - `FT` float type
 - `horizontal_layout_type` horizontal layout type (e.g., `IJFH` or `IJHF`)
 - `DA` device array type
 - `Nq` number of quad points
 - `Nh` number of horizontal elements
"""
function ColumnMask(
    ::Type{FT},
    ::Type{horizontal_layout_type},
    ::Type{DA},
    ::Val{Nq},
    ::Val{Nh},
) where {FT, horizontal_layout_type <: Union{IJFH, IJHF}, DA, Nq, Nh}
    @assert FT <: Real
    @assert Nq isa Integer
    @assert Nh isa Integer
    T = horizontal_layout_type
    is_active = replace_basetype(T{FT, Nq}(DA{FT}, Nh), Bool)
    parent(is_active) .= true
    return IJHMask(is_active)
end

function IJHMask(is_active::Union{IJFH, IJHF})
    DA = unionall_type(typeof(parent(is_active)))
    (Ni, Nj, _, _, Nh) = size(is_active)
    Nijh = Ni * Nj * Nh
    N = zeros(Int, 1)
    i_map = zeros(Int, Nijh)
    j_map = zeros(Int, Nijh)
    h_map = zeros(Int, Nijh)
    mask = IJHMask(rebuild(is_active, DA), N, DA(i_map), DA(j_map), DA(h_map))
    set_mask_maps!(mask)
    return mask
end

"""
    full_bitmask(mask::IJHMask, data::AbstractData)

Returns an array similar to `parent(data)`, containing
bools indicating when the `mask`'s `is_active == true`.

!!! warn

    This function provides users a work-around to compute mask-aware reductions,
    and should be deprecated in favor of providing native masked-reduction
    support. Therefore, this function should be used sparingly.

    This feature is extensible, but not performant in that it allocates
    and, on the gpu, will launch many kernels.
"""
function full_bitmask end

full_bitmask(mask::AbstractMask, data::AbstractData; complement::Bool = false) =
    full_bitmask(mask, nlevels(data), singleton(data); complement)

function full_bitmask(mask::IJHMask, Nv, s::VIJFHSingleton; complement::Bool)
    _arr = parent(mask.is_active)
    arr = complement ? .!_arr : _arr
    return repeat(reshape(arr, 1, size(arr)...), Nv)
end

full_bitmask(mask::AbstractMask, data::IJFH; complement::Bool = false) =
    complement ? .!parent(mask.is_active) : parent(mask.is_active)

end # module
