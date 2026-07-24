module DataLayouts

import Base: @propagate_inbounds
import LLVM: unsafe_load
import StaticArrays
import BlockArrays
import Adapt

import ClimaComms
import MultiBroadcastFusion: @make_type, @make_fused, fused_direct
using UnrolledUtilities

import ..Utilities.Unrolled: unrolled_setindex, unrolled_insert, unrolled_map_with_inbounds
import ..Utilities: add_auto_broadcasters, drop_auto_broadcasters, auto_broadcasted
import ..Utilities: stable_view, unionall_type, replace_type_parameter, safe_mapreduce
import ..Utilities: fieldtype_vals, return_type, safe_eltype, unsafe_eltype
import ..DebugOnly: call_post_op_callback, post_op_callback
import ..slab, ..slab_args, ..column, ..column_args, ..level, ..level_args

export DataScope, DataLayout, DataF, VIJFH, VIJHF, VIH1, IH1JH2

include("bitcast_struct.jl")
include("struct_storage.jl")
include("scopes.jl")

"""
    DataLayout{T, N, F, S, A}

An `N`-dimensional `AbstractArray` containing values of type `T`, stored in a
parent array of type `A` whose memory layout is determined by the layout's type.
Every value can be identified by four indices: a vertical level `v`, horizontal
quadrature points `i` and `j`, and a horizontal element `h`. The components of
each value are optionally stored along a hidden field axis `F` of the parent
array, leading to a hybrid of the traditional "array-of-structs" (`F = 1`) and
"struct-of-arrays" (`F = ndims(A)`) approaches to storing non-scalar data. The
[`DataScope`](@ref) `S` determines how loops and reductions over the values are
parallelized on CPUs and GPUs, and it dictates which array types are allocated.

Several layouts are available, named after the order of their parent axes:
- [`DataF`](@ref) is a 0-dimensional array that stores a single value, with an
  `Nf`-element parent array (used in place of a `Ref`)
- [`VIJFH`](@ref) is an `Nv × Ni × Nj × Nh` array that stores spatially
  varying data, with each value spread along the fourth parent axis
- [`VIJHF`](@ref) is like `VIJFH` with the `F` and `H` axes swapped, which permits
  [linear indexing](https://docs.julialang.org/en/v1/devdocs/subarrays/#Linear-indexing)
  and improves performance for operators that only access one field at a time
- [`VIJHWithF`](@ref) generalizes `VIJFH` and `VIJHF` to any `F` axis
  position, with `F = nothing` removing the axis altogether
- [`VIH1`](@ref) and [`IH1JH2`](@ref) store vertical and horizontal planes of
  interpolated data for plotting, whose `ih1` and `jh2` indices combine `i` and
  `j` with `h1` and `h2` (orthogonal components of `h` in rectangular domains)

```julia-repl
julia> data = VIJFH{Tuple{Int64, Float64, Int128}, 10, 5, 5, nothing}(Array{Int64}, 20);

julia> size(data), size(parent(data)) # Nh = 20 elements, Nf = 4 Int64 storage values
((10, 5, 5, 20), (10, 5, 5, 4, 20))

julia> data[1, 2, 3, 4] = (0, 1.0, 2); data.:1[1, 2, 3, 4], data[1, 2, 3, 4].:2
(0, 1.0)
```

# Extended Help

`DataLayout`s also provide the following functionality for ClimaCore:
- Assigning a [`DataScope`](@ref) to every batch of data, and automatically
  partitioning data across nestable multithreaded operations
- Storing specific array dimensions as type parameters, and allocating static
  arrays in place of regular arrays when every dimension can be inferred
- Using linear indices in place of Cartesian indices where doing so may
  improve performance, including in `getindex` and `view` operations
- Automatic nested broadcasting over `Tuple` and `NamedTuple` values (or
  other supported iterator types), along with broadcasting over array indices
- Checking for type stability before evaluating operations like broadcasts
  and reductions, avoiding inefficient CPU behavior and GPU compilation errors
- Falling back to built-in `AbstractArray` methods when specialized ClimaCore
  code is not available (this may be highly inefficient or fail to compile on
  GPUs, but it should generally work on CPUs)
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
if there is no separate `F` axis. The value of `nothing` is chosen instead of
`missing` because GPUCompiler.jl compares type parameters with `==`, which
returns the non-boolean `missing` whenever one of its arguments is `missing`.
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
type set to `nothing`.
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
this returns `nothing` for dimensions that cannot be inferred from the type.
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
    isnothing(vijh_params(data).Nh) ?
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
    reassign(D, scope)
    reassign(data, scope)

Assign a new [`DataScope`](@ref) to a [`DataLayout`](@ref), or determine the
result type of performing such an assignment for a layout of type `D`.
"""
@inline reassign(data, scope) = reassign(typeof(data), scope)(parent(data))
@inline reassign(::Type{D}, scope) where {D} =
    layout_type(D){eltype(D), shape_params(D)..., typeof(scope), parent_type(D)}

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

The new array can be stored on a different device (e.g., `Array` vs `CuArray`),
so the [`DataScope`](@ref) is modified if it is inconsistent with the new array.
"""
@inline rebuild(data, ::Type{A}, ::Type{T} = eltype(data); params...) where {A, T} =
    rebuild(data, A(parent(data)), T; params...)
@inline function rebuild(data, array, ::Type{T} = eltype(data); params...) where {T}
    scope = DataScope(array)
    scoped_data = is_subscope(DataScope(data), scope) ? data : reassign(data, scope)
    return layout_constructor(scoped_data, T; params...)(array)
end

Adapt.adapt_structure(to, data::DataLayout) = rebuild(data, Adapt.adapt(to, parent(data)))

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

# Hide zero-size fields (e.g. the singleton bases of Tensors) from property
# iteration, so that generic code which recursively walks propertynames only
# encounters properties that contain data. Zero-size fields are still
# accessible through getproperty, which returns a view with an empty F axis.
@inline function Base.propertynames(::DataLayout{T}) where {T}
    filter(name -> sizeof(fieldtype(T, name)) > 0, fieldnames(T))
end

# Wrap the field index in a Val as soon as it is available, resolving field
# views through specialization rather than constant propagation. Making the Val
# requires only one level of constant propagation, which is done by default,
# whereas propagating the index or name through every fieldtype lookup can
# exhaust the compiler's budget when an expression has several getproperty
# calls, causing runtime allocations from dynamic types. Return a lazy view if
# the parent array's element type is not aligned with the field's element type.
@inline function property_view(data, ::Val{i}) where {i}
    T = eltype(data)
    1 <= i <= fieldcount(T) || throw(BoundsError(data, i))
    is_valid_basetype(eltype(parent(data)), fieldtype(T, i)) ||
        return Broadcast.broadcasted(Base.Fix2(getfield, i), data)
    array = @inbounds struct_field_view(parent(data), T, Val(i), Val(f_dim(data)))
    return rebuild(data, array, fieldtype(T, i))
end

@inline Base.getproperty(data::DataLayout, i::Integer) = property_view(data, Val(i))
@inline Base.getproperty(data::DataLayout, name::Symbol) =
    property_view(data, Val(Base.fieldindex(eltype(data), name)))

# Base's fallback for dotgetproperty, which is called within data.name .= ___
# expressions, often generates runtime allocations because it isn't inlined.
@inline Base.dotgetproperty(data::DataLayout, name) = getproperty(data, name)

@inline check_Nh_dynamic(Nh_dynamic) =
    !isnothing(Nh_dynamic) ||
    throw(ArgumentError("Nh_dynamic must be specified to construct layout type"))

# Check that a parent array has the canonical size for its layout. Arrays with
# other shapes must be explicitly reshaped before they are wrapped in layouts.
@inline check_parent(array, array_size...) =
    size(array) == array_size ? array :
    throw(DimensionMismatch("Array size is not consistent with layout type"))

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
    length(array) == num_basetypes(eltype(array), T) ||
        throw(ArgumentError("Array length is not consistent with element type"))
    linearly_indexable_array = IndexStyle(array) == IndexLinear() ? array : vec(array)
    return DataF{T, S, typeof(linearly_indexable_array)}(linearly_indexable_array)
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
`nothing` and obtained at runtime from the array size. Each value of type `T`
can be stored across multiple indices along the fourth array axis. May be
constructed either from the parent array type or the parent array itself, though
using a type requires passing an additional integer if `Nh` is set to `nothing`.
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
    @assert (Ni == Nj || isone(Nj)) && (isnothing(Nh) || Nh isa Integer)
    Nf = num_basetypes(eltype(array), T)
    Nh_dynamic = isnothing(Nh) ? size(array)[F == 5 ? end - 1 : end] : Nh
    check_parent(array, add_f_dim((Nv, Ni, Nj, Nh_dynamic), Nf, Val(F))...)
    return VIJHWithF{T, Nv, Ni, Nj, Nh, F, S, typeof(array)}(array)
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
    (Nv, Ni, Nj, isnothing(Nh) ? size(parent(data), isnothing(F) || F == 5 ? 4 : 5) : Nh)
@inline nelems(data::VIJHWithF) = size(data, 4)

@propagate_inbounds function level_view(data::VIJHWithF, v)
    array = stable_view(parent(data), add_f_dim((v:v, :, :, :), :, Val(f_dim(data)))...)
    return rebuild(data, array; Nv = 1)
end
@propagate_inbounds function slab_view(data::VIJHWithF, v, h)
    array = stable_view(parent(data), add_f_dim((v:v, :, :, h:h), :, Val(f_dim(data)))...)
    return rebuild(data, array; Nv = 1, Nh = 1)
end
@propagate_inbounds function column_view(data::VIJHWithF, i, j, h)
    array = stable_view(parent(data), add_f_dim((:, i:i, j:j, h:h), :, Val(f_dim(data)))...)
    return rebuild(data, array; Ni = 1, Nj = 1, Nh = 1)
end

"""
    VIH1{T, Nv, Ni, Nh, [S]}(A, [Nh_dynamic])
    VIH1{T, Nv, Ni, Nh, [S]}(array)

[`DataLayout`](@ref) representing values of type `T` stored across `Nv` vertical
levels and `Ni × Nh1` horizontal quadrature points. This ignores the second
horizontal direction, which spans `Nj × Nh2` quadrature points (`Nh` is given by
`Nh1 × Nh2`). The parameters `Nv` and `Ni` must be integers, but `Nh` may be
set to `nothing` and obtained at runtime from the array size; when it is not
`nothing`, `Nh` can only be set to 1. May be constructed either from the parent
array type or the parent array itself, though using a type requires passing an
additional integer if `Nh` is set to `nothing`.
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
    @assert isnothing(Nh) || isone(Nh)
    Nh1 = isnothing(Nh) ? size(array, 2) ÷ Ni : Nh
    check_parent(array, Nv, Ni * Nh1)
    return VIH1{T, Nv, Ni, Nh, S, typeof(array)}(array)
end

@inline shape_params(::Type{<:VIH1{<:Any, Nv, Ni, Nh}}) where {Nv, Ni, Nh} =
    (; Nv, Ni, Nh)
@inline inferred_size(::Type{<:VIH1{<:Any, Nv, Ni, Nh}}) where {Nv, Ni, Nh} =
    (Nv, isnothing(Nh) ? nothing : Ni)
@inline Base.size(data::VIH1{<:Any, Nv, Ni, Nh}) where {Nv, Ni, Nh} =
    (Nv, isnothing(Nh) ? size(parent(data), 2) : Ni)
@inline nelems(data::VIH1) = size(data, 2) ÷ shape_params(data).Ni

@propagate_inbounds function level_view(data::VIH1, v)
    array = stable_view(parent(data), v:v, :)
    return rebuild(data, array; Nv = 1)
end
@propagate_inbounds function slab_view(data::VIH1, v, h)
    (; Ni) = shape_params(data)
    array = stable_view(parent(data), v:v, Ni * mod(h - 1, size(data, 2) ÷ Ni) .+ (1:Ni))
    return rebuild(data, array; Nv = 1, Nh = 1)
end
@propagate_inbounds function column_view(data::VIH1, i, _, h)
    (; Ni) = shape_params(data)
    array = stable_view(parent(data), :, Ni * mod(h - 1, size(data, 2) ÷ Ni) .+ (i:i))
    return rebuild(data, array; Ni = 1, Nh = 1)
end

"""
    IH1JH2{T, Ni, Nj, Nh, [S]}(A, [Nh_dynamic])
    IH1JH2{T, Ni, Nj, Nh, [S]}(array)

[`DataLayout`](@ref) representing values of type `T` stored across `Ni × Nh1`
quadrature points along one horizontal direction and `Nj × Nh2` quadrature
points along the other horizontal direction (`Nh` is given by `Nh1 × Nh2`). This
ignores the vertical direction, which spans `Nv` levels. The parameters `Ni` and
`Nj` must be integers, but `Nh` may be set to `nothing` and obtained at runtime
from the array size; when it is not `nothing`, `Nh` can only be set to 1. May be
constructed either from the parent array type or the parent array itself, though
using a type requires passing an additional integer if `Nh` is set to `nothing`.
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
    @assert (Ni == Nj || isone(Nj)) && (isnothing(Nh) || isone(Nh))
    Nh1 = isnothing(Nh) ? size(array, 1) ÷ Ni : Nh
    Nh2 = isnothing(Nh) ? size(array, 2) ÷ Nj : Nh
    check_parent(array, Ni * Nh1, Nj * Nh2)
    return IH1JH2{T, Ni, Nj, Nh, S, typeof(array)}(array)
end

@inline shape_params(::Type{<:IH1JH2{<:Any, Ni, Nj, Nh}}) where {Ni, Nj, Nh} =
    (; Ni, Nj, Nh)
@inline inferred_size(::Type{<:IH1JH2{<:Any, Ni, Nj, Nh}}) where {Ni, Nj, Nh} =
    isnothing(Nh) ? (nothing, nothing) : (Ni, Nj)
@inline Base.size(data::IH1JH2{<:Any, Ni, Nj, Nh}) where {Ni, Nj, Nh} =
    isnothing(Nh) ? size(parent(data)) : (Ni, Nj)
@inline nelems(data::IH1JH2) =
    length(data) ÷ (shape_params(data).Ni * shape_params(data).Nj)

@propagate_inbounds function slab_view(data::IH1JH2, _, h)
    (; Ni, Nj) = shape_params(data)
    (h2, h1) = fldmod(h - 1, size(data, 1) ÷ Ni) .+ 1
    array = stable_view(parent(data), Ni * (h1 - 1) .+ (1:Ni), Nj * (h2 - 1) .+ (1:Nj))
    return rebuild(data, array; Nh = 1)
end
@propagate_inbounds function column_view(data::IH1JH2, i, j, h)
    (; Ni, Nj) = shape_params(data)
    (h2, h1) = fldmod(h - 1, size(data, 1) ÷ Ni) .+ 1
    array = stable_view(parent(data), Ni * (h1 - 1) .+ (i:i), Nj * (h2 - 1) .+ (j:j))
    return rebuild(data, array; Ni = 1, Nj = 1, Nh = 1)
end

include("broadcast.jl")
include("indexing.jl")
include("masks.jl")
include("loops.jl")
include("deprecated.jl")

# Drop the recursion limits of this module's Core.kwcall methods and recursive
# DataScope functions, so that kwarg functions like fill! and column_reduce! can
# be composed, multiple scopes can be combined, and is_subscope/slice_subscope
# can repeatedly partition a scope. The default limit makes the compiler widen
# argument types, leading to dynamic dispatch and runtime allocations.
@static if hasfield(Method, :recursion_relation)
    for f in (Core.kwcall, DataScope, is_subscope, slice_subscope), method in methods(f)
        method.module === (@__MODULE__) || continue
        method.recursion_relation = Returns(true)
    end
end

end # module
