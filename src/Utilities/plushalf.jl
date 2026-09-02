"""
    PlusHalf(i)

Number representing `i + 1/2`, stored internally as the integer `i`. Used for
indexing into staggered finite difference meshes: by convention, half-integer
indices refer to cell faces and integer indices refer to cell centers.

Supports `+`, `-`, `min`, `max`, and inequalities.

See also [`half`](@ref).
"""
struct PlusHalf{I <: Integer} <: Real
    i::I
end
PlusHalf{I}(h::PlusHalf{I}) where {I <: Integer} = h

"""
    half

The constant `PlusHalf(0)`, representing the index `1/2`; `i + half` is the
[`PlusHalf`](@ref) index `i + 1/2`.
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

Base.convert(::Type{P}, i::Integer) where {P <: PlusHalf} =
    throw(InexactError(:convert, P, i))
Base.convert(::Type{I}, h::PlusHalf) where {I <: Integer} =
    throw(InexactError(:convert, I, h))

Base.length(r::UnitRange{PlusHalf{I}}) where {I} =
    last(r) - first(r) + oneunit(I)

Base.step(::AbstractUnitRange{PlusHalf{I}}) where {I} = one(I)
