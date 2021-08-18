import BlockArrays

struct FieldVector{T, M} <: BlockArrays.AbstractBlockVector{T}
    values::M
end
FieldVector{T}(values::M) where {T, M} = FieldVector{T, M}(values)

_values(fv::FieldVector) = getfield(fv, :values)
_parent_values(fv::FieldVector) = map(parent, _values(fv))

BlockArrays.blockaxes(fv::FieldVector) =
    (BlockArrays.BlockRange(1:length(_values(fv))),)
Base.axes(fv::FieldVector) =
    (BlockArrays.blockedrange(map(length ∘ parent, Tuple(_values(fv)))),)

Base.getindex(fv::FieldVector, block::BlockArrays.Block{1}) =
    parent(_values(fv)[block.n...])
Base.getindex(fv::FieldVector, bidx::BlockArrays.BlockIndex{1}) =
    fv[BlockArrays.block(bidx)][bidx.α...]
Base.getindex(fv::FieldVector, i::Integer) =
    getindex(fv, BlockArrays.findblockindex(axes(fv, 1), i))

function Base.setindex!(fv::FieldVector, val, bidx::BlockArrays.BlockIndex{1})
    X = fv[BlockArrays.block(bidx)]
    X[bidx.α...] = val
end
Base.setindex!(fv::FieldVector, val, i::Integer) =
    setindex!(fv, val, BlockArrays.findblockindex(axes(fv, 1), i))


function FieldVector(; kwargs...)
    values = NamedTuple(kwargs)
    T = promote_type(
        map(RecursiveArrayTools.recursive_bottom_eltype, values)...,
    )
    return FieldVector{T}(values)
end

@inline function Base.getproperty(fv::FieldVector, name::Symbol)
    getfield(_values(fv), name)
end

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
    mapreduce(x -> mapreduce(f, op, x), op, _parent_values(fv))

Base.any(f, fv::FieldVector) = any(x -> any(f, x), _parent_values(fv))
Base.any(f::Function, fv::FieldVector) = # avoid ambiguities
    any(x -> any(f, x), _parent_values(fv))
Base.any(fv::FieldVector) = any(identity, A)

Base.all(f, fv::FieldVector) = all(x -> all(f, x), _parent_values(fv))
Base.all(f::Function, fv::FieldVector) = all(x -> all(f, x), _parent_values(fv))
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
