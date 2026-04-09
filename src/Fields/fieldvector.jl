import BlockArrays


"""
    FieldVector

A `FieldVector` is a wrapper around one or more `Field`s that acts like vector
of the underlying arrays.

It is similar in spirit to [`ArrayPartition` from
RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl#arraypartition),
but allows referring to fields by name.

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
unwrap(x::ScalarWrapper) = x[]

function FieldVector(; kwargs...)
    values = map(wrap, NamedTuple(kwargs))
    T = promote_type(
        map(RecursiveArrayTools.recursive_bottom_eltype, values)...,
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

Base.@propagate_inbounds slab(fv::FieldVector{T}, inds...) where {T} =
    FieldVector{T}(slab_args(_values(fv), inds...))
Base.@propagate_inbounds column(fv::FieldVector{T}, inds...) where {T} =
    FieldVector{T}(column_args(_values(fv), inds...))

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

# Recursively call transform_bc_args() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
@inline transform_bc_args(args::Tuple, inds...) =
    unrolled_map(args) do arg
        transform_broadcasted(arg, inds...)
    end

@inline function transform_broadcasted(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
    symb,
    axes,
)
    Base.Broadcast.Broadcasted(
        bc.f,
        transform_bc_args(bc.args, symb, axes),
        axes,
    )
end
@inline transform_broadcasted(fv::FieldVector, symb, axes) =
    parent(getfield(_values(fv), symb))
@inline transform_broadcasted(x, symb, axes) = x

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Union{FieldVector, Base.Broadcast.Broadcasted{FieldVectorStyle}},
)
    copyto_per_field!(dest, bc)
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

@inline function copyto_per_field!(
    dest::FieldVector,
    bc::Union{FieldVector, Base.Broadcast.Broadcasted{FieldVectorStyle}},
)
    map(propertynames(dest)) do symb
        Base.@_inline_meta
        array = parent(getfield(_values(dest), symb))
        bct = transform_broadcasted(bc, symb, axes(array))
        if array isa FieldVector # recurse
            copyto_per_field!(array, bct)
        else
            copyto_per_field!(
                array,
                Base.Broadcast.instantiate(bct),
                DataLayouts.device_dispatch(array),
            )
        end
    end
    return dest
end

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.Style{Tuple}},
)
    copyto_per_field_scalar!(dest, bc)
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}},
)
    copyto_per_field_scalar!(dest, bc)
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

@inline function Base.copyto!(dest::FieldVector, bc::Real)
    copyto_per_field_scalar!(dest, bc)
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

@inline function copyto_per_field_scalar!(dest::FieldVector, bc)
    map(propertynames(dest)) do symb
        Base.@_inline_meta
        array = parent((getfield(_values(dest), symb)))
        if array isa FieldVector # recurse
            copyto_per_field_scalar!(array, bc)
        else
            copyto_per_field_scalar!(
                array,
                Base.Broadcast.instantiate(bc),
                DataLayouts.device_dispatch(array),
            )
        end
        nothing
    end
    return dest
end

Base.fill!(dest::FieldVector, value) = dest .= value

Base.mapreduce(f, op, fv::FieldVector) =
    mapreduce(x -> mapreduce(f, op, backing_array(x)), op, _values(fv))

Base.any(f, fv::FieldVector) = any(x -> any(f, backing_array(x)), _values(fv))
Base.any(f::Function, fv::FieldVector) = # avoid ambiguities
    any(x -> any(f, backing_array(x)), _values(fv))
Base.any(fv::FieldVector) = any(identity, A)

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

import ClimaComms

# To infer the ClimaComms device and its properties, use the first Field in a
# FieldVector that isn't a PointField, since a PointField's data can be stored
# on a different device from other Fields to avoid scalar indexing on GPUs. If
# the FieldVector only contains PointFields, fall back to using the first one.
function representative_field(x)
    all_fields = _values(x)
    isempty(all_fields) && error("Empty FieldVector has no ClimaComms device")
    field_index = unrolled_findfirst(Base.Fix2(!isa, PointField), all_fields)
    return all_fields[isnothing(field_index) ? 1 : field_index]
end

ClimaComms.array_type(x::FieldVector) =
    ClimaComms.array_type(representative_field(x))
ClimaComms.device(x::FieldVector) = ClimaComms.device(representative_field(x))
ClimaComms.context(x::FieldVector) = ClimaComms.context(representative_field(x))


function __rprint_diff(
    io::IO,
    x::T,
    y::T;
    pc,
    xname,
    yname,
) where {T <: Union{FieldVector, Field, DataLayouts.AbstractData, NamedTuple}}
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
_rcompare(pass, x::T, y::T; strict) where {T <: DataLayouts.AbstractData} =
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
