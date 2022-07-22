"""
    PlusHalf(i)

Represents `i + 1/2`, but stored as internally as an integer value. Used for
indexing into staggered finite difference meshes: the convention "half" values
are indexed at cell faces, whereas centers are indexed at cell centers.

Supports `+`, `-` and inequalities.

See also [`half`](@ref).
"""
struct PlusHalf{I <: Integer} <: Real
    i::I
end
PlusHalf{I}(h::PlusHalf{I}) where {I <: Integer} = h

"""
    const half = PlusHalf(0)
"""
const half = PlusHalf(0)

Base.:+(h::PlusHalf) = h
Base.:-(h::PlusHalf) = PlusHalf(-h.i - one(h.i))
Base.:+(i::Integer, h::PlusHalf) = PlusHalf(i + h.i)
Base.:+(h::PlusHalf, i::Integer) = PlusHalf(h.i + i)
Base.:+(h1::PlusHalf, h2::PlusHalf) = h1.i + h2.i + one(h1.i)
Base.:-(i::Integer, h::PlusHalf) = PlusHalf(i - h.i - one(h.i))
Base.:-(h::PlusHalf, i::Integer) = PlusHalf(h.i - i)
Base.:-(h1::PlusHalf, h2::PlusHalf) = h1.i - h2.i

Base.:<=(h1::PlusHalf, h2::PlusHalf) = h1.i <= h2.i
Base.:<(h1::PlusHalf, h2::PlusHalf) = h1.i < h2.i
Base.max(h1::PlusHalf, h2::PlusHalf) = PlusHalf(max(h1.i, h2.i))
Base.min(h1::PlusHalf, h2::PlusHalf) = PlusHalf(min(h1.i, h2.i))
# Base.isequal(h1::PlusHalf, h2::PlusHalf) = h1.i == h2.i

Base.convert(::Type{P}, i::Integer) where {P <: PlusHalf} =
    throw(InexactError(:convert, P, i))
Base.convert(::Type{I}, h::PlusHalf) where {I <: Integer} =
    throw(InexactError(:convert, I, h))

function Base.length(r::UnitRange{PlusHalf{I}}) where {I}
    # i = (last(r) - first(r) + oneunit(I))::Int
    i = (last(r) - first(r) + oneunit(I))
    return i
end

Base.step(::AbstractUnitRange{PlusHalf{I}}) where {I} = one(I)
# Base.last(r::AbstractUnitRange{PlusHalf{I}}) where {I} = length(r)


# @code_typed length(PlusHalf(0):PlusHalf(10))
