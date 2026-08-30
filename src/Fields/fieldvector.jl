import BlockArrays


"""
    FieldVector

A `FieldVector` is a wrapper around one or more `Field`s that acts like vector
of the underlying arrays.

Unlike a plain concatenation of the underlying arrays, its fields can be
referred to by name.

# Constructors

    FieldVector(;name1=field1, name2=field2, ...)

Construct a `FieldVector`, wrapping `field1, field2, ...` using the names
`name1, name2, ...`.
"""
struct FieldVector{T, M} <: BlockArrays.AbstractBlockVector{T}
    values::M
end
FieldVector{T}(values::M) where {T, M} = FieldVector{T, M}(values)

function Adapt.adapt_structure(to, fv::FieldVector)
    pn = propertynames(fv)
    vals = map(key -> Adapt.adapt(to, getproperty(fv, key)), pn)
    return FieldVector(; NamedTuple{pn}(vals)...)
end

"""
    Fields.ScalarWrapper(val) <: AbstractArray{T,0}

This is a wrapper around scalar values that allows them to be mutated as part of
a FieldVector. A call `getproperty` on a `FieldVector` with this component will
return a scalar, instead of the boxed object.
"""
mutable struct ScalarWrapper{T} <: AbstractArray{T, 0}
    val::T
end
Base.size(::ScalarWrapper) = ()
Base.getindex(s::ScalarWrapper) = s.val
Base.setindex!(s::ScalarWrapper, value) = s.val = value
Base.similar(s::ScalarWrapper) = ScalarWrapper(s.val)
# Without this method, zero(::ScalarWrapper) would return a 0-dimensional
# Array, so zero(::FieldVector) would not preserve the component types of
# FieldVectors with scalar components (breaking strict equality with them).
Base.zero(s::ScalarWrapper) = ScalarWrapper(zero(s.val))

"""
    Fields.wrap(x)

Construct a mutable wrapper around `x`. This can be extended for new types
(especially immutable ones).
"""
wrap(x) = x
wrap(x::Real) = ScalarWrapper(x)
wrap(x::NamedTuple) = FieldVector(; pairs(x)...)


"""
    Fields.unwrap(x::T)

This is called when calling `getproperty` on a `FieldVector` property of element
type `T`.
"""
unwrap(x) = x

# The recursion goes through the backing array, not the `DataLayout`: `eltype`
# of a heterogeneous-`Tuple`-valued layout reaches `Any` (via `eltype(Tuple{A,
# B})`), which would type the `FieldVector` as `FieldVector{Any}`; the backing
# array bottoms out at the scalar.
recursive_bottom_eltype(field::Field) =
    recursive_bottom_eltype(parent(field))
unwrap(x::ScalarWrapper) = x[]

function FieldVector(; kwargs...)
    values = map(wrap, NamedTuple(kwargs))
    T = promote_type(
        map(recursive_bottom_eltype, values)...,
    )
    return FieldVector{T}(values)
end

_values(fv::FieldVector) = getfield(fv, :values)

"""
    backing_array(x)

The `AbstractArray` that is backs an object `x`, allowing it to be treated as a
component of a `FieldVector`.
"""
backing_array(x) = x
backing_array(x::Field) = parent(x)


Base.propertynames(fv::FieldVector) = propertynames(_values(fv))
@inline function Base.getproperty(fv::FieldVector, name::Symbol)
    unwrap(getfield(_values(fv), name))
end

@inline function Base.setproperty!(fv::FieldVector, name::Symbol, value)
    x = getfield(_values(fv), name)
    x .= value
end


BlockArrays.blockaxes(fv::FieldVector) =
    (BlockArrays.BlockRange(1:length(_values(fv))),)
Base.axes(fv::FieldVector) =
    (BlockArrays.blockedrange(map(length ∘ backing_array, Tuple(_values(fv)))),)

# The AbstractArray fallback computes length from axes, whose blockedrange is
# not inferrable for nested FieldVectors and allocates on every call; sum the
# block lengths directly instead (length recurses into nested FieldVectors,
# whose backing_array is the FieldVector itself).
Base.length(fv::FieldVector) = unrolled_reduce(
    (n, value) -> n + length(backing_array(value)),
    Tuple(_values(fv)),
    0,
)

Base.@propagate_inbounds Base.getindex(
    fv::FieldVector,
    block::BlockArrays.Block{1},
) = backing_array(_values(fv)[block.n...])
Base.@propagate_inbounds function Base.getindex(
    fv::FieldVector,
    bidx::BlockArrays.BlockIndex{1},
)
    X = fv[BlockArrays.block(bidx)]
    X[bidx.α...]
end

# TODO: drop support for this
Base.@propagate_inbounds Base.getindex(fv::FieldVector, i::Integer) =
    getindex(fv, BlockArrays.findblockindex(axes(fv, 1), i))

Base.@propagate_inbounds function Base.setindex!(
    fv::FieldVector,
    val,
    bidx::BlockArrays.BlockIndex{1},
)
    X = fv[BlockArrays.block(bidx)]
    X[bidx.α...] = val
end
# TODO: drop support for this
Base.@propagate_inbounds Base.setindex!(fv::FieldVector, val, i::Integer) =
    setindex!(fv, val, BlockArrays.findblockindex(axes(fv, 1), i))

Base.similar(fv::FieldVector{T}) where {T} =
    FieldVector{T}(map(similar, _values(fv)))
Base.similar(fv::FieldVector{T}, ::Type{T}) where {T} =
    FieldVector{T}(map(similar, _values(fv)))
_similar(x, ::Type{T}) where {T} = similar(x, T)
_similar(x::Field, ::Type{T}) where {T} =
    Field(DataLayouts.replace_basetype(field_values(x), T), axes(x))
Base.similar(fv::FieldVector{T}, ::Type{T′}) where {T, T′} =
    FieldVector{T′}(map(x -> _similar(x, T′), _values(fv)))

Base.copy(fv::FieldVector{T}) where {T} = FieldVector{T}(map(copy, _values(fv)))
Base.zero(fv::FieldVector{T}) where {T} = FieldVector{T}(map(zero, _values(fv)))

for op in (:level, :slab, :column)
    @eval Base.@propagate_inbounds $op(fv::FieldVector{T}, inds...) where {T} =
        FieldVector{T}($op(_values(fv), inds...))
end

struct FieldVectorStyle <: Base.Broadcast.AbstractArrayStyle{1} end

Base.Broadcast.BroadcastStyle(::Type{<:FieldVector}) = FieldVectorStyle()

Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.DefaultArrayStyle{0},
) = fs
Base.Broadcast.BroadcastStyle(
    as::Base.Broadcast.DefaultArrayStyle{0},
    fs::FieldVectorStyle,
) = fs
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.AbstractArrayStyle{0},
) = fs
Base.Broadcast.BroadcastStyle(
    as::Base.Broadcast.AbstractArrayStyle{0},
    fs::FieldVectorStyle,
) = fs
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.DefaultArrayStyle,
) = as
Base.Broadcast.BroadcastStyle(
    as::Base.Broadcast.DefaultArrayStyle,
    fs::FieldVectorStyle,
) = as
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.AbstractArrayStyle,
) = as
Base.Broadcast.BroadcastStyle(
    as::Base.Broadcast.AbstractArrayStyle,
    fs::FieldVectorStyle,
) = as

function Base.similar(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
    ::Type{T},
) where {T}
    for arg in bc.args
        if arg isa FieldVector ||
           arg isa Base.Broadcast.Broadcasted{FieldVectorStyle}
            return similar(arg, T)
        end
    end
    error("Cannot construct FieldVector")
end

"""
    Spaces.create_dss_buffer(fv::FieldVector)

Create a NamedTuple of buffers for communicating neighbour information of
each Field in `fv`. In this NamedTuple, the name of each field is mapped
to the buffer.
"""
function Spaces.create_dss_buffer(fv::FieldVector)
    NamedTuple{propertynames(fv)}(
        map(
            key -> Spaces.create_dss_buffer(getproperty(fv, key)),
            propertynames(fv),
        ),
    )
end

"""
    Spaces.weighted_dss!(fv::FieldVector, dss_buffer = Spaces.create_dss_buffer(fv))

Apply weighted direct stiffness summation (DSS) to each field in `fv`.
If a `dss_buffer` object is not provided, a buffer will be created for each
field in `fv`.
Note that using the `Pair` interface here parallelizes the `weighted_dss!` calls.
"""
function Spaces.weighted_dss!(
    fv::FieldVector,
    dss_buffer = Spaces.create_dss_buffer(fv),
)
    pairs = map(propertynames(fv)) do key
        Pair(getproperty(fv, key), getproperty(dss_buffer, key))
    end
    Spaces.weighted_dss!(pairs...)
end

@inline function first_fieldvector_in_bc(args::Tuple, rargs...)
    idx = unrolled_findfirst(args) do arg
        !isnothing(first_fieldvector_in_bc(arg))
    end
    return isnothing(idx) ? nothing : first_fieldvector_in_bc(args[idx])
end

@inline first_fieldvector_in_bc(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
) = first_fieldvector_in_bc(bc.args)
@inline first_fieldvector_in_bc(fv::FieldVector) = fv
@inline first_fieldvector_in_bc(x) = nothing

@inline _is_diagonal_bc_args(
    ::Type{TStart},
    args::Tuple,
) where {TStart} =
    unrolled_all(args) do arg
        _is_diagonal_bc(TStart, arg)
    end

@inline function _is_diagonal_bc(
    ::Type{TStart},
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
) where {TStart}
    return _is_diagonal_bc_args(TStart, bc.args)
end

@inline _is_diagonal_bc(
    ::Type{TStart},
    ::TStart,
) where {TStart <: FieldVector} = true
@inline _is_diagonal_bc(
    ::Type{TStart},
    x::FieldVector,
) where {TStart} = false
@inline _is_diagonal_bc(::Type{TStart}, x) where {TStart} = true

# Find the first fieldvector in the broadcast expression (BCE),
# and compare against every other fieldvector in the BCE
@inline is_diagonal_bc(bc::Base.Broadcast.Broadcasted{FieldVectorStyle}) =
    _is_diagonal_bc_args(typeof(first_fieldvector_in_bc(bc)), bc.args)

# Specialize on FieldVectorStyle to avoid inference failure
# in fieldvector broadcast expressions:
# https://github.com/JuliaArrays/BlockArrays.jl/issues/310
function Base.Broadcast.instantiate(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
)
    if bc.axes isa Nothing # Not done via dispatch to make it easier to extend instantiate(::Broadcasted{Style})
        axes = Base.Broadcast.combine_axes(bc.args...)
    else
        axes = bc.axes
        # Base.Broadcast.check_broadcast_axes is type-unstable
        # for broadcast expressions with multiple fieldvectors.
        # So, let's statically elide this when we have "diagonal"
        # broadcast expressions:
        if !is_diagonal_bc(bc)
            Base.Broadcast.check_broadcast_axes(axes, bc.args...)
        end
    end
    return Base.Broadcast.Broadcasted{FieldVectorStyle}(bc.f, bc.args, axes)
end

# Val-wrap property names so broadcast transformations and closures receive type
# parameters rather than runtime Symbols; deeply nested broadcasts can exhaust the
# constant-propagation budget before the getfield calls, causing runtime allocations.
@inline property_name_vals(fv::FieldVector) = property_name_vals(_values(fv))
@inline property_name_vals(::NamedTuple{names}) where {names} =
    unrolled_map(Val, names)
@inline unval(::Val{value}) where {value} = value

# Recursively call transform_bc_args() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
@inline transform_bc_args(args::Tuple, inds...) =
    unrolled_map(args) do arg
        transform_broadcasted(arg, inds...)
    end

@inline function transform_broadcasted(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
    symb_val,
    axes,
)
    Base.Broadcast.Broadcasted(
        bc.f,
        transform_bc_args(bc.args, symb_val, axes),
        axes,
    )
end
@inline transform_broadcasted(fv::FieldVector, ::Val{symb}, axes) where {symb} =
    parent(getfield(_values(fv), symb))
@inline transform_broadcasted(x, symb_val, axes) = x

# FieldVector entries are N-dimensional arrays, and broadcasting over them on
# GPU pays for an N-dimensional Cartesian index computation (one integer
# division and modulo per dimension) in every thread. When every array in a
# broadcast has the same contiguous layout as the destination, the expression
# can instead be evaluated over `vec`s of the arrays, with plain linear
# indexing. The GPU extension marks its array types via is_gpu_array_type;
# everything else keeps the standard path.
is_gpu_array_type(::Type) = false
is_gpu_array_type(::Type{<:SubArray{<:Any, <:Any, P}}) where {P} =
    is_gpu_array_type(P)

@inline is_contiguous_gpu_array(a::DenseArray) =
    is_gpu_array_type(typeof(a)) && ndims(a) > 0
@inline is_contiguous_gpu_array(a::SubArray) =
    is_gpu_array_type(typeof(a)) && ndims(a) > 0 && Base.iscontiguous(a)
@inline is_contiguous_gpu_array(a::AbstractArray) = false

# is_flat_compatible gates flatten_bc_arg: the argument types the gate accepts
# must be exactly those the transform below has methods for, so that any type
# added to one without the other fails with a MethodError instead of flattening
# incorrectly. Arrays must match the destination's size exactly (broadcasts
# that expand a smaller argument keep Cartesian indexing).
@inline is_flat_compatible(dest::AbstractArray, arg::AbstractArray) =
    is_contiguous_gpu_array(dest) &&
    is_contiguous_gpu_array(arg) &&
    size(dest) == size(arg)
@inline is_flat_compatible(dest::AbstractArray, arg::Number) = true
@inline is_flat_compatible(dest::AbstractArray, arg) = false
@inline is_flat_compatible(dest::AbstractArray, bc::Base.Broadcast.Broadcasted) =
    is_contiguous_gpu_array(dest) &&
    unrolled_all(Base.Fix1(is_flat_compatible, dest), bc.args)

# flatten_bc_arg replaces every array in the broadcast with its `vec`, except
# the destination itself, which must map to the same `flat_dest` object that
# is passed to `copyto!`: `Broadcast.broadcast_unalias` only skips its
# defensive copy when destination and argument are identical (`===`), and two
# separate `vec` wrappers around the same memory alias without being
# identical, so every in-place update like `x .-= dx` would allocate a device
# copy of `x`.
@inline flatten_bc_arg(
    dest::AbstractArray,
    flat_dest::AbstractArray,
    arg::AbstractArray,
) = dest === arg ? flat_dest : vec(arg)
@inline flatten_bc_arg(
    dest::AbstractArray,
    flat_dest::AbstractArray,
    arg::Number,
) = arg
@inline flatten_bc_arg(
    dest::AbstractArray,
    flat_dest::AbstractArray,
    bc::Base.Broadcast.Broadcasted,
) = Base.Broadcast.Broadcasted(
    bc.f,
    unrolled_map(arg -> flatten_bc_arg(dest, flat_dest, arg), bc.args),
    (Base.OneTo(length(dest)),),
)

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Union{FieldVector, Base.Broadcast.Broadcasted{FieldVectorStyle}},
)
    unrolled_foreach(property_name_vals(dest)) do symb_val
        array = parent(getfield(_values(dest), unval(symb_val)))
        bct = transform_broadcasted(bc, symb_val, axes(array))
        if array isa FieldVector
            copyto!(array, bct)
        elseif is_flat_compatible(array, bct)
            flat_dest = vec(array)
            copyto!(
                flat_dest,
                Base.Broadcast.instantiate(flatten_bc_arg(array, flat_dest, bct)),
            )
        else
            copyto!(array, Base.Broadcast.instantiate(bct))
        end
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

# Define separate methods for Style{Tuple} and AbstractArrayStyle{0}, instead
# of a single method for their Union, to avoid a dispatch ambiguity with the
# method for AbstractArrays in Base.Broadcast.
for S in
    (:(Base.Broadcast.Style{Tuple}), :(Base.Broadcast.AbstractArrayStyle{0}))
    @eval @inline function Base.copyto!(
        dest::FieldVector,
        bc::Base.Broadcast.Broadcasted{<:$S},
    )
        unrolled_foreach(property_name_vals(dest)) do symb_val
            array = parent(getfield(_values(dest), unval(symb_val)))
            array isa FieldVector ? copyto!(array, bc) :
            copyto!(array, Base.Broadcast.instantiate(bc))
        end
        call_post_op_callback() && post_op_callback(dest, dest, bc)
        return dest
    end
end

# Copying a scalar fills every entry with it, as in fill!. Without this method,
# Base's fallback would iterate over the scalar and call setindex!, which is a
# disallowed scalar indexing operation for a FieldVector backed by a GPU array.
@inline Base.copyto!(dest::FieldVector, value::Number) = fill!(dest, value)

@inline function Base.fill!(dest::FieldVector, value)
    unrolled_foreach(property_name_vals(dest)) do symb_val
        fill!(parent(getfield(_values(dest), unval(symb_val))), value)
    end
    call_post_op_callback() && post_op_callback(dest, dest, value)
    return dest
end

Base.mapreduce(f, op, fv::FieldVector) =
    mapreduce(x -> mapreduce(f, op, backing_array(x)), op, _values(fv))

Base.any(f, fv::FieldVector) = any(x -> any(f, backing_array(x)), _values(fv))
Base.any(f::Function, fv::FieldVector) = # avoid ambiguities
    any(x -> any(f, backing_array(x)), _values(fv))
Base.any(fv::FieldVector) = any(identity, fv)

Base.all(f, fv::FieldVector) = all(x -> all(f, backing_array(x)), _values(fv))
Base.all(f::Function, fv::FieldVector) =
    all(x -> all(f, backing_array(x)), _values(fv))
Base.all(fv::FieldVector) = all(identity, fv)

# TODO: figure out a better way to handle these
# https://github.com/JuliaArrays/BlockArrays.jl/issues/185
LinearAlgebra.ldiv!(
    x::FieldVector,
    A::LinearAlgebra.QRCompactWY,
    b::FieldVector,
) = x .= LinearAlgebra.ldiv!(A, Vector(b))
LinearAlgebra.ldiv!(A::LinearAlgebra.QRCompactWY, x::FieldVector) =
    x .= LinearAlgebra.ldiv!(A, Vector(x))

LinearAlgebra.ldiv!(x::FieldVector, A::LinearAlgebra.LU, b::FieldVector) =
    x .= LinearAlgebra.ldiv!(A, Vector(b))

LinearAlgebra.ldiv!(A::LinearAlgebra.LU, x::FieldVector) =
    x .= LinearAlgebra.ldiv!(A, Vector(x))

function LinearAlgebra.norm_sqr(x::FieldVector)
    value_norm_sqrs = unrolled_map(_values(x)) do value
        LinearAlgebra.norm_sqr(backing_array(value))
    end
    return sum(value_norm_sqrs; init = zero(eltype(x)))
end
function LinearAlgebra.norm(x::FieldVector)
    sqrt(LinearAlgebra.norm_sqr(x))
end

"""
    fieldvector2array!(array, fv)

Copy the entries of the `FieldVector` `fv` into the flat `AbstractVector`
`array` of the same length, without allocating or scalar indexing: each
component block is copied with a single array-level `copyto!`, so
`FieldVector`s backed by GPU arrays are supported (including mixed cases,
where `array` and some components live on different devices). Entries are
ordered as in the `FieldVector`'s own linear indexing: component blocks in
order, each in the linear order of its backing array.

Scalar (`ScalarWrapper`) components are written to `array` with a `fill!` on a
one-element view, which is GPU-safe.

Intended for interfacing with libraries that operate on flat vectors, such as
the Krylov.jl workspace vectors given by `Krylov.ktypeof(::FieldVector)` (see
`KrylovExt`). See [`array2fieldvector!`](@ref) for the inverse copy and
[`fieldvector2array`](@ref) for an allocating version.
"""
function fieldvector2array!(array::AbstractVector, fv::FieldVector)
    length(array) == length(fv) || throw(
        DimensionMismatch(
            "cannot copy FieldVector of length $(length(fv)) to array of \
             length $(length(array))",
        ),
    )
    _blocks2array!(array, 0, Tuple(_values(fv)))
    return array
end

"""
    array2fieldvector!(fv, array)

Copy the entries of the flat `AbstractVector` `array` into the `FieldVector`
`fv` of the same length — the inverse of [`fieldvector2array!`](@ref), with
the same entry ordering, allocation-free block-wise copies, and GPU support.

Copying a scalar (`ScalarWrapper`) component out of `array` requires a scalar
read, so scalar components are only supported when `array` is a CPU array (a
GPU-backed `array` throws a scalar-indexing error rather than performing a
hidden synchronizing transfer).

See [`array2fieldvector`](@ref) for an allocating version.
"""
function array2fieldvector!(fv::FieldVector, array::AbstractVector)
    length(array) == length(fv) || throw(
        DimensionMismatch(
            "cannot copy array of length $(length(array)) to FieldVector of \
             length $(length(fv))",
        ),
    )
    _array2blocks!(array, 0, Tuple(_values(fv)))
    return fv
end

"""
    fieldvector2array(fv)

Allocating version of [`fieldvector2array!`](@ref): copy `fv` into a freshly
allocated flat vector of `fv`'s device array type,
`ClimaComms.array_type(fv){eltype(fv), 1}`.
"""
fieldvector2array(fv::FieldVector) = fieldvector2array!(
    ClimaComms.array_type(fv){eltype(fv), 1}(undef, length(fv)),
    fv,
)

"""
    array2fieldvector(array, fv_prototype)

Allocating version of [`array2fieldvector!`](@ref): copy `array` into a
freshly allocated `FieldVector` with the same structure as `fv_prototype`
(created with `similar`, which preserves component types).
"""
array2fieldvector(array::AbstractVector, fv_prototype::FieldVector) =
    array2fieldvector!(similar(fv_prototype), array)

# Both directions fold an entry offset over the component blocks with
# unrolled_reduce; _block2array!/_array2block! copy one block and return the
# offset advanced past it, recursing into nested FieldVectors.
_blocks2array!(array, offset, vals::Tuple) = unrolled_reduce(
    (off, value) -> _block2array!(array, off, backing_array(value)),
    vals,
    offset,
)
_block2array!(array, offset, block::FieldVector) =
    _blocks2array!(array, offset, Tuple(_values(block)))
function _block2array!(array, offset, block::AbstractArray)
    n = length(block)
    copyto!(array, offset + 1, block, 1, n)
    return offset + n
end
# A 0-dimensional block (a ScalarWrapper) holds a CPU scalar; fill! on a
# one-element view writes it to `array` without allocating or scalar indexing.
function _block2array!(array, offset, block::AbstractArray{T, 0}) where {T}
    fill!(view(array, (offset + 1):(offset + 1)), block[])
    return offset + 1
end

_array2blocks!(array, offset, vals::Tuple) = unrolled_reduce(
    (off, value) -> _array2block!(array, off, backing_array(value)),
    vals,
    offset,
)
_array2block!(array, offset, block::FieldVector) =
    _array2blocks!(array, offset, Tuple(_values(block)))
function _array2block!(array, offset, block::AbstractArray)
    n = length(block)
    copyto!(block, 1, array, offset + 1, n)
    return offset + n
end
# Reading a scalar block back requires a scalar getindex on `array`; this is
# allocation-free on CPU arrays and throws a scalar-indexing error for
# GPU-backed arrays (see the docstring above).
function _array2block!(array, offset, block::AbstractArray{T, 0}) where {T}
    block[] = array[offset + 1]
    return offset + 1
end

import ClimaComms

function ClimaComms.array_type(x::FieldVector)
    T = _array_type(x)
    # Union{} means x contains nothing but scalars, which live on the CPU.
    return T === Union{} ? Array : T
end
# ScalarWrapper components hold CPU scalars regardless of where the other
# components live, so they must not participate in the promotion, at any
# nesting depth: their contribution is Union{}, the identity of promote_type
# (which a nested FieldVector of nothing but scalars also promotes to).
_array_type(x) = ClimaComms.array_type(x) # Fields
# The splatted form is used instead of unrolled_mapreduce because the latter's
# init keyword routes through Core.kwcall, which deepens the recursion cycle on
# nested FieldVectors until inference gives up on optimizing it.
_array_type(x::FieldVector) =
    promote_type(unrolled_map(_array_type, Tuple(_values(x)))...)
_array_type(::ScalarWrapper) = Union{}
_array_type(x::A) where {A <: AbstractArray} =
    parent(x) === x ? Base.typename(A).wrapper : _array_type(parent(x))

ClimaComms.device(x::FieldVector) = ClimaComms.device(ClimaComms.context(x))
function ClimaComms.context(x::FieldVector)
    isempty(_values(x)) && error("Empty FieldVector has no device or context")
    # We don't have promotion for devices or contexts, so we use the first value
    # that isn't a PointField (a PointField's data can be stored on a different
    # device from other Fields to avoid scalar indexing on GPUs). If there is no
    # such value, fall back to using the first PointField.
    index = unrolled_findfirst(Base.Fix1(!isa, PointField), _values(x))
    return ClimaComms.context(_values(x)[isnothing(index) ? 1 : index])
end

function __rprint_diff(
    io::IO,
    x::T,
    y::T;
    pc,
    xname,
    yname,
) where {T <: Union{FieldVector, Field, DataLayouts.DataLayout, NamedTuple}}
    for pn in propertynames(x)
        pc_full = (pc..., ".", pn)
        xi = getproperty(x, pn)
        yi = getproperty(y, pn)
        __rprint_diff(io, xi, yi; pc = pc_full, xname, yname)
    end
end;

function __rprint_diff(io::IO, xi, yi; pc, xname, yname) # assume we can compute difference here
    if !(xi == yi)
        xs = xname * string(join(pc))
        ys = yname * string(join(pc))
        println(io, "==================== Difference found:")
        println(io, "$xs: ", xi)
        println(io, "$ys: ", yi)
        println(io, "($xs .- $ys): ", (xi .- yi))
    end
    return nothing
end

"""
    rprint_diff(io::IO, ::T, ::T) where {T <: Union{FieldVector, NamedTuple}}
    rprint_diff(::T, ::T) where {T <: Union{FieldVector, NamedTuple}}

Recursively print differences in given `Union{FieldVector, NamedTuple}`.
"""
_rprint_diff(
    io::IO,
    x::T,
    y::T,
    xname,
    yname,
) where {T <: Union{FieldVector, NamedTuple}} =
    __rprint_diff(io, x, y; pc = (), xname, yname)
_rprint_diff(
    x::T,
    y::T,
    xname,
    yname,
) where {T <: Union{FieldVector, NamedTuple}} =
    _rprint_diff(stdout, x, y, xname, yname)

"""
    @rprint_diff(::T, ::T) where {T <: Union{FieldVector, NamedTuple}}

Recursively print differences in given `Union{FieldVector, NamedTuple}`.
"""
macro rprint_diff(x, y)
    return :(_rprint_diff(
        stdout,
        $(esc(x)),
        $(esc(y)),
        $(string(x)),
        $(string(y)),
    ))
end


# Recursively compare contents of similar fieldvectors
_rcompare(pass, x::T, y::T; strict) where {T <: Field} =
    pass && _rcompare(pass, field_values(x), field_values(y); strict)
_rcompare(pass, x::T, y::T; strict) where {T <: DataLayouts.DataLayout} =
    pass && (parent(x) == parent(y))
_rcompare(pass, x::T, y::T; strict) where {T} = pass && (x == y)

_rcompare(pass, x::NamedTuple, y::NamedTuple; strict) =
    _rcompare_nt(pass, x, y; strict)
_rcompare(pass, x::FieldVector, y::FieldVector; strict) =
    _rcompare_nt(pass, x, y; strict)

function _rcompare_nt(pass, x, y; strict)
    length(propertynames(x)) ≠ length(propertynames(y)) && return false
    if strict
        typeof(x) == typeof(y) || return false
    end
    for pn in propertynames(x)
        pass &= _rcompare(pass, getproperty(x, pn), getproperty(y, pn); strict)
    end
    return pass
end

"""
    rcompare(x::T, y::T; strict = true) where {T <: Union{FieldVector, NamedTuple}}

Recursively compare given fieldvectors via `==`.
Returns `true` if `x == y` recursively.

The keyword `strict = true` allows users to additionally
check that the types match. If `strict = false`, then
`rcompare` will return `true` for `FieldVector`s and
`NamedTuple`s with the same properties but permuted order.
For example:

  - `rcompare((;a=1,b=2), (;b=2,a=1); strict = true)` will return `false` and
  - `rcompare((;a=1,b=2), (;b=2,a=1); strict = false)` will return `true`
"""
rcompare(
    x::T,
    y::T;
    strict = true,
) where {T <: Union{FieldVector, NamedTuple}} = _rcompare(true, x, y; strict)

rcompare(x::T, y::T; strict = true) where {T <: FieldVector} =
    _rcompare(true, x, y; strict)

rcompare(x::T, y::T; strict = true) where {T <: NamedTuple} =
    _rcompare(true, x, y; strict)

# FieldVectors with different types are always different
rcompare(x::FieldVector, y::FieldVector; strict::Bool = true) =
    strict ? false : _rcompare(true, x, y; strict)

rcompare(x::NamedTuple, y::NamedTuple; strict::Bool = true) =
    strict ? false : _rcompare(true, x, y; strict)

# Define == to call rcompare for two fieldvectors
Base.:(==)(x::FieldVector, y::FieldVector) = rcompare(x, y; strict = true)
