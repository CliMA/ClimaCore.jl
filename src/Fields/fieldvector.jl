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
Base.getindex(s::ScalarWrapper, args...) = s.val
Base.setindex!(s::ScalarWrapper, value, args...) = s.val = value
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

Base.getindex(fv::FieldVector, block::BlockArrays.Block{1}) =
    backing_array(_values(fv)[block.n...])
function Base.getindex(fv::FieldVector, bidx::BlockArrays.BlockIndex{1})
    X = fv[BlockArrays.block(bidx)]
    X[bidx.α...]
end
Base.getindex(fv::FieldVector, i::Integer) =
    getindex(fv, BlockArrays.findblockindex(axes(fv, 1), i))

function Base.setindex!(fv::FieldVector, val, bidx::BlockArrays.BlockIndex{1})
    X = fv[BlockArrays.block(bidx)]
    X[bidx.α...] = val
end
Base.setindex!(fv::FieldVector, val, i::Integer) =
    setindex!(fv, val, BlockArrays.findblockindex(axes(fv, 1), i))


Base.similar(fv::FieldVector{T}) where {T} =
    FieldVector{T}(map(similar, _values(fv)))
Base.similar(fv::FieldVector{T}, ::Type{T}) where {T} =
    FieldVector{T}(map(similar, _values(fv)))
function Base.similar(fv::FieldVector{T}, ::Type{FT}) where {T, FT}
    FieldVector{FT}(
        map(_values(fv)) do x
            Field(DataLayouts.replace_basetype(field_values(x), FT), axes(x))
        end,
    )
end

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

LinearAlgebra.ldiv!(x::FieldVector, A::LinearAlgebra.LU, b::FieldVector) =
    x .= LinearAlgebra.ldiv!(A, Vector(b))

LinearAlgebra.ldiv!(A::LinearAlgebra.LU, x::FieldVector) =
    x .= LinearAlgebra.ldiv!(A, Vector(x))
