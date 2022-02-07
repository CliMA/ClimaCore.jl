"""
    Cent(i)

Represents `i`, but stored as internally as an integer value. Used for
indexing into staggered finite difference meshes: the convention "cent" values
are indexed at cell centers, whereas [`PlusHalf`](@ref) are indexed at cell faces.

Supports `+`, `-` and inequalities.

This is a complementary struct to [`PlusHalf`](@ref).

This is required to differentiate
 - `getindex` to get the i-th n-tuple field
 - `getindex` to get the k-th vertical cell center
"""
struct Cent{I <: Integer}
    i::I
end

Base.:+(h::Cent) = h
Base.:-(h::Cent) = Cent(-h.i - one(h.i))
Base.:+(i::Integer, h::Cent) = Cent(i + h.i)
Base.:+(h::Cent, i::Integer) = Cent(h.i + i)
Base.:+(h1::Cent, h2::Cent) = h1.i + h2.i + one(h1.i)
Base.:-(i::Integer, h::Cent) = Cent(i - h.i - one(h.i))
Base.:-(h::Cent, i::Integer) = Cent(h.i - i)
Base.:-(h1::Cent, h2::Cent) = h1.i - h2.i

Base.:<=(h1::Cent, h2::Cent) = h1.i <= h2.i
Base.:<(h1::Cent, h2::Cent) = h1.i < h2.i
Base.max(h1::Cent, h2::Cent) = Cent(max(h1.i, h2.i))
Base.min(h1::Cent, h2::Cent) = Cent(min(h1.i, h2.i))

# Can we do this?
# # TODO: deprecate, we should not overload getindex/setindex for ordinary arrays.
# Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Cent) = Base.getindex(arr, i.i)
# Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Cent) = Base.setindex!(arr, v, i.i)
# Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::CCO.PlusHalf) = Base.getindex(arr, i.i)
# Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::CCO.PlusHalf) = Base.setindex!(arr, v, i.i)
# Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Int, j::Cent) = Base.getindex(arr, i, j.i)
# Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Int, j::Cent) = Base.setindex!(arr, v, i, j.i)
# Base.@propagate_inbounds Base.getindex(arr::AbstractArray, i::Int, j::CCO.PlusHalf) = Base.getindex(arr, i, j.i)
# Base.@propagate_inbounds Base.setindex!(arr::AbstractArray, v, i::Int, j::CCO.PlusHalf) = Base.setindex!(arr, v, i, j.i)
