module DataLayouts

import Base: @propagate_inbounds
import LLVM: unsafe_load
import StaticArrays
import Adapt

import ClimaComms
import MultiBroadcastFusion: @make_type, @make_fused, fused_direct
using UnrolledUtilities

import ..Utilities.Unrolled:
    unrolled_setindex, unrolled_insert, unrolled_map_with_inbounds
import ..Utilities: unionall_type, replace_type_parameter, fieldtype_vals
import ..Utilities: return_type, safe_eltype, unsafe_eltype, auto_broadcasted
import ..Utilities: add_auto_broadcasters, drop_auto_broadcasters
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..slab, ..slab_args, ..column, ..column_args, ..level, ..level_args

export DataScope, DataLayout, DataF, VIJFH, VIJHF, VIH1, IH1JH2

include("bitcast_struct.jl")
include("struct_storage.jl")
include("scopes.jl")

"""
    DataLayout{T, N, F, S, A}

Wrapper for an `N`-dimensional array that represents values of type `T`.
Elements of the underlying array can be identified using the following indices:
- `f` is the field index (fields of `T` can span multiple parent array elements)
- `v` is the vertical level index
- `h` is the the horizontal element index
  - `h1` and `h2` are the orthogonal components of `h` in rectangular domains
- `i` is the horizontal quadrature index aligned with `h1` inside each element
  - `ih1` is a single index that combines `i` and `h1`
- `j` is the horizontal quadrature index aligned with `h2` inside each element
  - `jh2` is a single index that combines `j` and `h2`

Several layout options are available:
- [`DataF`](@ref) is a 0-dimensional array
  - Used to store a single data point
  - Hidden `F` axis represents values of type `T` as `Nf` parent array elements
- [`VIJFH`](@ref) is an `Nv × Ni × Nj × Nh` array of all possible indices
  - Used to store spatially-varying data
  - Hidden `F` axis represents every data point like a `DataF`
- [`VIJHF`](@ref) is like `VIJFH` with the `F` and `H` parent array axes swapped
  - Better performance for operators that only access one field at a time
  - Placing the hidden `F` axis at the end allows each field to be accessed via
    [linear indexing](https://docs.julialang.org/en/v1/devdocs/subarrays/#Linear-indexing)
- [`VIJHWithF`](@ref) generalizes `VIJFH` and `VIJHF` to any `F` axis position
  - Setting the `F` parameter to `nothing` removes the `F` axis altogether
- [`VIH1`](@ref) is an `Nv × Nih1` array of indices in a vertical plane
  - Used to store interpolated vertical data for plotting
- [`IH1JH2`](@ref) is an `Nih1 × Njh2` array of indices in a horizontal plane
  - Used to store interpolated horizontal data for plotting

`DataLayout` wrappers achieve two primary goals: they map each axis of an array
to domain-specific coordinate axes, and they avoid the downsides of a typical
["array of structs"](https://en.wikipedia.org/wiki/AoS_and_SoA) design by adding
an `F` axis, which permits efficient access to struct fields via `getproperty`.

They also provide the following functionality to simplify ClimaCore's internals:
- Assigning a [`DataScope`](@ref) to every batch of data, and automatically
  partitioning data across nestable multithreaded operations
- Storing specific array dimensions as type parameters, and allocating static
  arrays in place of regular arrays when every dimension can be inferred
- Using linear indices in place of Cartesian indices where doing so may improve
  performance, including in `getindex` and `view` operations while broadcasting
- Automatic nested broadcasting over `Tuple` and `NamedTuple` values (or other
  supported iterator types), along with standard broadcasting over array indices
- Checking for type stability before evaluating operations like broadcasts and
  reductions, avoiding inefficient CPU behavior and GPU compilation errors
- Falling back to built-in `AbstractArray` methods when specialized ClimaCore
  code is not available (this may result in highly inefficient behavior or
  compilation errors when running on GPUs, but it should generally work on CPUs)
"""
abstract type DataLayout{T, N, F, S, A} <: AbstractArray{T, N} end

@inline Base.parent(data::DataLayout) = getfield(data, :array)

DataScope(::Type{<:DataLayout{<:Any, <:Any, <:Any, S}}) where {S <: DataScope} =
    S.instance

"""
    layout_type(D)
    layout_type(data)

Type of a [`DataLayout`](@ref), but stripped of all its type parameters.
"""
@inline layout_type(::D) where {D <: DataLayout} = layout_type(D)
@inline layout_type(::Type{D}) where {D <: DataLayout} = unionall_type(D)

"""
    parent_type(D)
    parent_type(data)

Type of parent array used by a [`DataLayout`](@ref), or a similar abstract type
if the concrete type is unavailable.
"""
@inline parent_type(::D) where {D <: DataLayout} = parent_type(D)
@inline parent_type(::Type{<:DataLayout{<:Any, <:Any, <:Any, <:Any, A}}) where {A} = A

"""
    f_dim(D)
    f_dim(data)

Index of the `F` axis in a parent array for a [`DataLayout`](@ref), or `nothing`
if there is no separate `F` axis.
"""
@inline f_dim(::D) where {D <: DataLayout} = f_dim(D)
@inline f_dim(::Type{<:DataLayout{<:Any, <:Any, F}}) where {F} = F

"""
    shape_params(D)
    shape_params(data)

A `NamedTuple` with all shape-related parameters of a [`DataLayout`](@ref). This
excludes its element type, its parent array type, and its [`DataScope`](@ref).
"""
@inline shape_params(::D) where {D <: DataLayout} = shape_params(D)

"""
    inferred_size(D)
    inferred_size(data)

Size of a [`DataLayout`](@ref), with dimensions that cannot be inferred from its
type set to `missing`.
"""
@inline inferred_size(::D) where {D <: DataLayout} = inferred_size(D)

"""
    has_inferred_size(D)
    has_inferred_size(data)

Whether every dimension of a [`DataLayout`](@ref) can be inferred from its type.
"""
@inline has_inferred_size(data) = inferred_size(data) isa Tuple{Vararg{Integer}}

"""
    vijh_params(D)
    vijh_params(data)

A `NamedTuple` with `Nv`, `Ni`, `Nj`, and `Nh`, representing lengths of the `V`,
`I`, `J`, and `H` axes in a [`DataLayout`](@ref). Like [`inferred_size`](@ref),
this returns `missing` for dimensions that cannot be inferred from the type.
"""
@inline vijh_params(data) = (;
    Nv = get(shape_params(data), :Nv, 1),
    Ni = get(shape_params(data), :Ni, 1),
    Nj = get(shape_params(data), :Nj, 1),
    Nh = get(shape_params(data), :Nh, 1),
)

"""
    nlevels(D)
    nlevels(data)

Length of the `V` axis in a [`DataLayout`](@ref).
"""
@inline nlevels(data) = vijh_params(data).Nv

"""
    nquadpoints(D)
    nquadpoints(data)

Product of the lengths of the `I` and `J` axes in a [`DataLayout`](@ref).
"""
@inline nquadpoints(data) = vijh_params(data).Ni * vijh_params(data).Nj

"""
    nelems(D)
    nelems(data)

Length of the `H` axis in a [`DataLayout`](@ref). When the length cannot be
inferred from its type, a concrete instance of it must be provided instead.
"""
@inline nelems(data) =
    ismissing(vijh_params(data).Nh) ?
    throw(ArgumentError("Length of H axis cannot be inferred from layout type")) :
    vijh_params(data).Nh

"""
    ncomponents(D)
    ncomponents(data)

Length of the hidden `F` axis in a [`DataLayout`](@ref), or 1 if there is no
separate `F` axis.
"""
@inline ncomponents(data) = num_basetypes(eltype(parent_type(data)), eltype(data))

"""
    layout_constructor(D, [T]; [params...])
    layout_constructor(data, [T]; [params...])

Constructor for a similar [`DataLayout`](@ref) that can be applied as
`constructor(array)`, with the element type optionally replaced with `T`, and
with any subset of the [`shape_params`](@ref) optionally replaced with `params`.
"""
@inline layout_constructor(data, ::Type{T} = eltype(data); params...) where {T} =
    layout_type(data){T, (; shape_params(data)..., params...)..., typeof(DataScope(data))}

"""
    rebuild(data, A, [T]; [params...])
    rebuild(data, array, [T]; [params...])

Reconstruct a [`DataLayout`](@ref) with a modified parent array, either
converting its parent array to some type `A`, or replacing it with another
`array`. As in [`layout_constructor`](@ref), a new element type and new
[`shape_params`](@ref) may also be specified.
"""
@inline rebuild(data, ::Type{A}, ::Type{T} = eltype(data); params...) where {A, T} =
    layout_constructor(data, T; params...)(A(parent(data)))
@inline rebuild(data, array, ::Type{T} = eltype(data); params...) where {T} =
    layout_constructor(data, T; params...)(array)

"""
    reassign(data, scope)

Assign a [`DataLayout`](@ref) to a new [`DataScope`](@ref).
"""
@inline reassign(data, scope) =
    layout_type(data){eltype(data), shape_params(data)..., typeof(scope), parent_type(data)}(
        parent(data),
    )

Adapt.adapt_structure(to, data::DataLayout) =
    rebuild(reassign(data, Adapt.adapt(to, DataScope(data))), Adapt.adapt(to, parent(data)))

Base.copy(data::DataLayout) = rebuild(data, copy(parent(data)))
Base.reinterpret(::Type{T}, data::DataLayout) where {T} = rebuild(data, parent(data), T)

ClimaComms.gather(::ClimaComms.SingletonCommsContext, data::DataLayout) = data
ClimaComms.gather(ctx::ClimaComms.AbstractCommsContext, data::DataLayout) =
    rebuild(data, ClimaComms.gather(ctx, parent(data)))

@inline add_f_dim(dims, dim, ::Val{F}) where {F} =
    isnothing(F) ? dims : unrolled_insert(dims, dim, Val(F))

function similar_layout(data, ::Type{T}, maybe_dims...) where {T}
    B = checked_valid_basetype(eltype(parent_type(data)), T)
    return similar_layout(data, T, B, maybe_dims...)
end
function similar_layout(data, ::Type{T}, ::Type{B}, maybe_dims...) where {T, B}
    Nf = num_basetypes(B, T)
    dims_or_data_size =
        isone(length(maybe_dims)) ? first(maybe_dims) :
        has_inferred_size(data) ? inferred_size(data) : size(data)
    array_size = add_f_dim(dims_or_data_size, Nf, Val(f_dim(data)))
    new_scoped_array = has_inferred_size(data) ? scoped_static_array : scoped_array
    array = new_scoped_array(DataScope(data), B, array_size)
    return rebuild(data, array, T)
end

Base.similar(::Type{D}, maybe_dims::Dims...) where {D <: DataLayout} =
    similar_layout(D, eltype(D), maybe_dims...)
Base.similar(data::DataLayout, maybe_dims::Dims...) =
    similar_layout(data, eltype(data), maybe_dims...)
Base.similar(data::DataLayout, ::Type{T}, maybe_dims::Dims...) where {T} =
    similar_layout(data, T, maybe_dims...)

function replace_basetype(data::DataLayout, ::Type{B}) where {B}
    T = replace_type_parameter(eltype(data), eltype(parent_type(data)), B)
    return similar_layout(data, T, B)
end

@inline Base.propertynames(data::DataLayout) = fieldnames(eltype(data))

@inline function Base.getproperty(data::DataLayout, i::Integer)
    T = eltype(data)
    1 <= i <= fieldcount(T) || throw(ArgumentError(invalid_field_string(T, Val(i))))
    array = @inbounds struct_field_view(parent(data), T, Val(i), Val(f_dim(data)))
    return rebuild(data, array, fieldtype(T, i))
end
@inline function Base.getproperty(data::DataLayout, name::Symbol)
    T = eltype(data)
    hasfield(T, name) || throw(ArgumentError(invalid_field_string(T, Val(name))))
    return getproperty(data, unrolled_findfirst(==(name), fieldnames(T)))
end
@generated invalid_field_string(
    ::Type{T},
    ::Val{i_or_name},
) where {T, i_or_name} = "Type $T has no field $i_or_name"

# Reshape arrays with too few dimensions to simplify slice views and array2field.
@inline maybe_reshaped_array(array, array_size...) =
    size(array) == array_size ? array :
    isone(length(array_size)) || ndims(array) < length(array_size) ?
    reshape(array, array_size) :
    throw(DimensionMismatch("Array size is not consistent with layout type"))

@inline check_Nh_dynamic(Nh_dynamic) =
    !ismissing(Nh_dynamic) ||
    throw(ArgumentError("Nh_dynamic must be specified to construct layout type"))

"""
    DataF{T, [S]}(A)
    DataF{T, [S]}(array)

[`DataLayout`](@ref) representing a single value of type `T`, which can be
stored across multiple array indices. This is used in place of a `Ref` to wrap
data that is stored in any array. May be constructed either from the parent
array type or the parent array itself.
"""
struct DataF{T, S, A} <: DataLayout{T, 0, 1, S, A}
    array::A
end

DataF{T}(array) where {T} = DataF{T, typeof(DataScope(array))}(array)
DataF{T, S}(::Type{A}) where {T, S, A} =
    DataF{T, S}(similar(A, num_basetypes(eltype(A), T)))
function DataF{T, S}(array) where {T, S}
    check_basetype(eltype(array), T)
    parent_array = maybe_reshaped_array(array, num_basetypes(eltype(array), T))
    return DataF{T, S, typeof(parent_array)}(parent_array)
end

@inline shape_params(::Type{<:DataF}) = (;)
@inline inferred_size(::Type{<:DataF}) = ()
@inline Base.size(::DataF) = ()

"""
    VIJHWithF{T, Nv, Ni, Nj, Nh, F, [S]}(A, [Nh_dynamic])

Generalization of a [`VIJFH`](@ref) and a [`VIJHF`](@ref), which supports any
value of the parameter `F` between 1 and 5, representing `FVIJH`, `VFIJH`, and
so on. The parameter can also be `nothing`, which drops the `F` axis altogether.
"""
struct VIJHWithF{T, Nv, Ni, Nj, Nh, F, S, A} <: DataLayout{T, 4, F, S, A}
    array::A
end

"""
    VIJFH{T, Nv, Ni, Nj, Nh, [S]}(A, [Nh_dynamic])
    VIJFH{T, Nv, Ni, Nj, Nh, [S]}(array)

[`DataLayout`](@ref) representing values of type `T` stored across `Nv` vertical
levels, `Nh` horizontal elements, and `Ni × Nj` quadrature points per element.
The parameters `Nv`, `Ni`, and `Nj` must be integers, but `Nh` may be set to
`missing` and obtained at runtime from the array size. Each value of type `T`
can be stored across multiple indices along the fourth array axis. May be
constructed either from the parent array type or the parent array itself, though
using a type requires passing an additional integer if `Nh` is set to `missing`.
"""
const VIJFH{T, Nv, Ni, Nj, Nh, S, A} = VIJHWithF{T, Nv, Ni, Nj, Nh, 4, S, A}

"""
    VIJHF{T, Nv, Ni, Nj, Nh, [S]}(A, [Nh_dynamic])
    VIJHF{T, Nv, Ni, Nj, Nh, [S]}(array)

[`DataLayout`](@ref) similar to [`VIJFH`](@ref), but with the last two axes of
the parent array swapped. Offers better performance than `VIJFH` for operations
that only access one field from each value of type `T`.
"""
const VIJHF{T, Nv, Ni, Nj, Nh, S, A} = VIJHWithF{T, Nv, Ni, Nj, Nh, 5, S, A}

VIJHWithF{T, Nv, Ni, Nj, Nh, F}(array, Nh_dynamic...) where {T, Nv, Ni, Nj, Nh, F} =
    VIJHWithF{T, Nv, Ni, Nj, Nh, F, typeof(DataScope(array))}(array, Nh_dynamic...)
function VIJHWithF{T, Nv, Ni, Nj, Nh, F, S}(
    ::Type{A},
    Nh_dynamic = Nh,
) where {T, Nv, Ni, Nj, Nh, F, S, A}
    check_Nh_dynamic(Nh_dynamic)
    Nf = num_basetypes(eltype(A), T)
    array = similar(A, add_f_dim((Nv, Ni, Nj, Nh_dynamic), Nf, Val(F))...)
    return VIJHWithF{T, Nv, Ni, Nj, Nh, F, S}(array)
end
function VIJHWithF{T, Nv, Ni, Nj, Nh, F, S}(array) where {T, Nv, Ni, Nj, Nh, F, S}
    check_basetype(eltype(array), T)
    @assert (Ni == Nj || isone(Nj)) && (ismissing(Nh) || Nh isa Integer)
    Nf = num_basetypes(eltype(array), T)
    Nh_dynamic = ismissing(Nh) ? length(array) ÷ (Nv * Ni * Nj * Nf) : Nh
    array_size = add_f_dim((Nv, Ni, Nj, Nh_dynamic), Nf, Val(F))
    parent_array = maybe_reshaped_array(array, array_size...)
    return VIJHWithF{T, Nv, Ni, Nj, Nh, F, S, typeof(parent_array)}(parent_array)
end

@inline shape_params(
    ::Type{<:VIJHWithF{<:Any, Nv, Ni, Nj, Nh, F}},
) where {Nv, Ni, Nj, Nh, F} = (; Nv, Ni, Nj, Nh, F)
@inline inferred_size(
    ::Type{<:VIJHWithF{<:Any, Nv, Ni, Nj, Nh}},
) where {Nv, Ni, Nj, Nh} = (Nv, Ni, Nj, Nh)
@inline Base.size(
    data::VIJHWithF{<:Any, Nv, Ni, Nj, Nh, F},
) where {Nv, Ni, Nj, Nh, F} =
    (Nv, Ni, Nj, ismissing(Nh) ? size(parent(data), F == 5 ? 4 : 5) : Nh)
@inline nelems(data::VIJHWithF) = size(data, 4)

@propagate_inbounds function level_view(data::VIJHWithF, v)
    array = view(parent(data), add_f_dim((v, :, :, :), :, Val(f_dim(data)))...)
    return rebuild(data, array; Nv = 1)
end
@propagate_inbounds function slab_view(data::VIJHWithF, v, h)
    array = view(parent(data), add_f_dim((v, :, :, h), :, Val(f_dim(data)))...)
    return rebuild(data, array; Nv = 1, Nh = 1)
end
@propagate_inbounds function column_view(data::VIJHWithF, i, j, h)
    array = view(parent(data), add_f_dim((:, i, j, h), :, Val(f_dim(data)))...)
    return rebuild(data, array; Ni = 1, Nj = 1, Nh = 1)
end

"""
    VIH1{T, Nv, Ni, Nh, [S]}(A, [Nh_dynamic])
    VIH1{T, Nv, Ni, Nh, [S]}(array)

[`DataLayout`](@ref) representing values of type `T` stored across `Nv` vertical
levels and `Ni × Nh1` horizontal quadrature points. This ignores the second
horizontal direction, which spans `Nj × Nh2` quadrature points (`Nh` is given by
`Nh1 × Nh2`). The parameters `Nv` and `Ni` must be integers, but `Nh` may be
set to `missing` and obtained at runtime from the array size; when it is not
`missing`, `Nh` can only be set to 1. May be constructed either from the parent
array type or the parent array itself, though using a type requires passing an
additional integer if `Nh` is set to `missing`.
"""
struct VIH1{T, Nv, Ni, Nh, S, A} <: DataLayout{T, 2, nothing, S, A}
    array::A
end

VIH1{T, Nv, Ni, Nh}(array, Nh_dynamic...) where {T, Nv, Ni, Nh} =
    VIH1{T, Nv, Ni, Nh, typeof(DataScope(array))}(array, Nh_dynamic...)
VIH1{T, Nv, Ni, Nh, S}(::Type{A}, Nh_dynamic = Nh) where {T, Nv, Ni, Nh, S, A} =
    check_Nh_dynamic(Nh_dynamic) &&
    VIH1{T, Nv, Ni, Nh, S}(similar(A, Nv, Ni * Nh_dynamic))
function VIH1{T, Nv, Ni, Nh, S}(array) where {T, Nv, Ni, Nh, S}
    check_basetype(eltype(array), T)
    @assert ismissing(Nh) || isone(Nh)
    Nh1 = ismissing(Nh) ? length(array) ÷ (Nv * Ni) : Nh
    parent_array = maybe_reshaped_array(array, Nv, Ni * Nh1)
    return VIH1{T, Nv, Ni, Nh, S, typeof(parent_array)}(parent_array)
end

@inline shape_params(::Type{<:VIH1{<:Any, Nv, Ni, Nh}}) where {Nv, Ni, Nh} =
    (; Nv, Ni, Nh)
@inline inferred_size(::Type{<:VIH1{<:Any, Nv, Ni, Nh}}) where {Nv, Ni, Nh} =
    (Nv, ismissing(Nh) ? missing : Ni)
@inline Base.size(data::VIH1{<:Any, Nv, Ni, Nh}) where {Nv, Ni, Nh} =
    (Nv, ismissing(Nh) ? size(parent(data), 2) : Ni)
@inline nelems(data::VIH1) = size(data, 2) ÷ shape_params(data).Ni

@propagate_inbounds function level_view(data::VIH1, v)
    array = view(parent(data), v, :)
    return rebuild(data, array; Nv = 1)
end
@propagate_inbounds function slab_view(data::VIH1, v, h)
    (; Ni) = shape_params(data)
    array = view(parent(data), v, Ni * mod(h - 1, size(data, 2) ÷ Ni) .+ (1:Ni))
    return rebuild(data, array; Nv = 1, Nh = 1)
end
@propagate_inbounds function column_view(data::VIH1, i, _, h)
    (; Ni) = shape_params(data)
    array = view(parent(data), :, Ni * mod(h - 1, size(data, 2) ÷ Ni) + i)
    return rebuild(data, array; Ni = 1, Nh = 1)
end

"""
    IH1JH2{T, Ni, Nj, Nh, [S]}(A, [Nh_dynamic])
    IH1JH2{T, Ni, Nj, Nh, [S]}(array)

[`DataLayout`](@ref) representing values of type `T` stored across `Ni × Nh1`
quadrature points along one horizontal direction and `Nj × Nh2` quadrature
points along the other horizontal direction (`Nh` is given by `Nh1 × Nh2`). This
ignores the vertical direction, which spans `Nv` levels. The parameters `Ni` and
`Nj` must be integers, but `Nh` may be set to `missing` and obtained at runtime
from the array size; when it is not `missing`, `Nh` can only be set to 1. May be
constructed either from the parent array type or the parent array itself, though
using a type requires passing an additional integer if `Nh` is set to `missing`.
"""
struct IH1JH2{T, Ni, Nj, Nh, S, A} <: DataLayout{T, 2, nothing, S, A}
    array::A
end

IH1JH2{T, Ni, Nj, Nh}(array, Nh_dynamic...) where {T, Ni, Nj, Nh} =
    IH1JH2{T, Ni, Nj, Nh, typeof(DataScope(array))}(array, Nh_dynamic...)
IH1JH2{T, Ni, Nj, Nh, S}(::Type{A}, Nh_dynamic = Nh) where {T, Ni, Nj, Nh, S, A} =
    check_Nh_dynamic(Nh_dynamic) &&
    IH1JH2{T, Ni, Nj, Nh, S}(similar(A, Ni * Nh_dynamic, Nj))
function IH1JH2{T, Ni, Nj, Nh, S}(array) where {T, Ni, Nj, Nh, S}
    check_basetype(eltype(array), T)
    @assert (Ni == Nj || isone(Nj)) && (ismissing(Nh) || isone(Nh))
    Nh1 = ismissing(Nh) ? size(array, 1) ÷ Ni : Nh
    Nh2 = ismissing(Nh) ? size(array, 2) ÷ Nj : Nh
    parent_array = maybe_reshaped_array(array, Ni * Nh1, Nj * Nh2)
    return IH1JH2{T, Ni, Nj, Nh, S, typeof(parent_array)}(parent_array)
end

@inline shape_params(::Type{<:IH1JH2{<:Any, Ni, Nj, Nh}}) where {Ni, Nj, Nh} =
    (; Ni, Nj, Nh)
@inline inferred_size(::Type{<:IH1JH2{<:Any, Ni, Nj, Nh}}) where {Ni, Nj, Nh} =
    ismissing(Nh) ? (missing, missing) : (Ni, Nj)
@inline Base.size(data::IH1JH2{<:Any, Ni, Nj, Nh}) where {Ni, Nj, Nh} =
    ismissing(Nh) ? size(parent(data)) : (Ni, Nj)
@inline nelems(data::IH1JH2) =
    length(data) ÷ (shape_params(data).Ni * shape_params(data).Nj)

@propagate_inbounds function slab_view(data::IH1JH2, _, h)
    (; Ni, Nj) = shape_params(data)
    (h2, h1) = fldmod(h - 1, size(data, 1) ÷ Ni) .+ 1
    array = view(parent(data), Ni * (h1 - 1) .+ (1:Ni), Nj * (h2 - 1) .+ (1:Nj))
    return rebuild(data, array; Nh = 1)
end
@propagate_inbounds function column_view(data::IH1JH2, i, j, h)
    (; Ni, Nj) = shape_params(data)
    (h2, h1) = fldmod(h - 1, size(data, 1) ÷ Ni) .+ 1
    array = view(parent(data), Ni * (h1 - 1) + i, Nj * (h2 - 1) + j)
    return rebuild(data, array; Ni = 1, Nj = 1, Nh = 1)
end

include("broadcast.jl")
include("indexing.jl")
include("masks.jl")
include("loops.jl")

end # module
