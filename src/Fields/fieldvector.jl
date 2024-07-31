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

struct FieldVectorStyle <: Base.Broadcast.AbstractArrayStyle{1} end

Base.Broadcast.BroadcastStyle(::Type{<:FieldVector}) = FieldVectorStyle()

Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.DefaultArrayStyle{0},
) = fs
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.AbstractArrayStyle{0},
) = fs
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.DefaultArrayStyle,
) = as
Base.Broadcast.BroadcastStyle(
    fs::FieldVectorStyle,
    as::Base.Broadcast.AbstractArrayStyle,
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

@inline function Base.copyto!(dest::FV, src::FV) where {FV <: FieldVector}
    for symb in propertynames(dest)
        pd = parent(getproperty(dest, symb))
        ps = parent(getproperty(src, symb))
        copyto!(pd, ps)
    end
    return dest
end

# Recursively call transform_bc_args() on broadcast arguments in a way that is statically reducible by the optimizer
# see Base.Broadcast.preprocess_args
@inline transform_bc_args(args::Tuple, inds...) = (
    transform_broadcasted(args[1], inds...),
    transform_bc_args(Base.tail(args), inds...)...,
)
@inline transform_bc_args(args::Tuple{Any}, inds...) =
    (transform_broadcasted(args[1], inds...),)
@inline transform_bc_args(args::Tuple{}, inds...) = ()

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

@inline function first_fieldvector_in_bc(args::Tuple, rargs...)
    x1 = first_fieldvector_in_bc(args[1], rargs...)
    x1 isa FieldVector && return x1
    return first_fieldvector_in_bc(Base.tail(args), rargs...)
end

@inline first_fieldvector_in_bc(args::Tuple{Any}, rargs...) =
    first_fieldvector_in_bc(args[1], rargs...)
@inline first_fieldvector_in_bc(args::Tuple{}, rargs...) = nothing
@inline first_fieldvector_in_bc(x) = nothing
@inline first_fieldvector_in_bc(x::FieldVector) = x

@inline first_fieldvector_in_bc(
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
) = first_fieldvector_in_bc(bc.args)

@inline _is_diagonal_bc_args(
    truesofar,
    ::Type{TStart},
    args::Tuple,
    rargs...,
) where {TStart} =
    truesofar &&
    _is_diagonal_bc(truesofar, TStart, args[1], rargs...) &&
    _is_diagonal_bc_args(truesofar, TStart, Base.tail(args), rargs...)

@inline _is_diagonal_bc_args(
    truesofar,
    ::Type{TStart},
    args::Tuple{Any},
    rargs...,
) where {TStart} =
    truesofar && _is_diagonal_bc(truesofar, TStart, args[1], rargs...)
@inline _is_diagonal_bc_args(
    truesofar,
    ::Type{TStart},
    args::Tuple{},
    rargs...,
) where {TStart} = truesofar

@inline function _is_diagonal_bc(
    truesofar,
    ::Type{TStart},
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
) where {TStart}
    return truesofar && _is_diagonal_bc_args(truesofar, TStart, bc.args)
end

@inline _is_diagonal_bc(
    truesofar,
    ::Type{TStart},
    ::TStart,
) where {TStart <: FieldVector} = true
@inline _is_diagonal_bc(
    truesofar,
    ::Type{TStart},
    x::FieldVector,
) where {TStart} = false
@inline _is_diagonal_bc(truesofar, ::Type{TStart}, x) where {TStart} = truesofar

# Find the first fieldvector in the broadcast expression (BCE),
# and compare against every other fieldvector in the BCE
@inline is_diagonal_bc(bc::Base.Broadcast.Broadcasted{FieldVectorStyle}) =
    _is_diagonal_bc_args(true, typeof(first_fieldvector_in_bc(bc)), bc.args)

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

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Base.Broadcast.Broadcasted{FieldVectorStyle},
)
    map(propertynames(dest)) do symb
        Base.@_inline_meta
        p = parent(getfield(_values(dest), symb))
        copyto!(p, transform_broadcasted(bc, symb, axes(p)))
    end
    return dest
end

@inline function Base.copyto!(
    dest::FieldVector,
    bc::Base.Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}},
)
    map(propertynames(dest)) do symb
        Base.@_inline_meta
        p = parent(getfield(_values(dest), symb))
        copyto!(p, bc)
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
    value_norm_sqrs = UnrolledFunctions.unrolled_map(_values(x)) do value
        LinearAlgebra.norm_sqr(backing_array(value))
    end
    return sum(value_norm_sqrs; init = zero(eltype(x)))
end
function LinearAlgebra.norm(x::FieldVector)
    sqrt(LinearAlgebra.norm_sqr(x))
end

import ClimaComms

ClimaComms.array_type(x::FieldVector) = promote_type(
    UnrolledFunctions.unrolled_map(ClimaComms.array_type, _values(x))...,
)
