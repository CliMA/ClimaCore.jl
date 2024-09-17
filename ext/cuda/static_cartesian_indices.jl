#! format: off

"""
    StaticCartesianIndices

Similar to `CartesianIndices`, but contains the array
dimensions in the type domain in order to avoid integer
division when converting from linear to cartesian indices.

See also:
 - https://github.com/maleadt/StaticCartesian.jl
 - https://github.com/JuliaGPU/GPUArrays.jl/pull/454
 - https://github.com/JuliaGPU/GPUArrays.jl/pull/520
 - https://github.com/JuliaGPU/Metal.jl/issues/101

This was adapted from Julia Base
"""
struct StaticCartesianIndices{N,R,denoms} <: AbstractArray{CartesianIndex{N},N}
end;
StaticCartesianIndices(::Tuple{}) = StaticCartesianIndices{0,()}(());
function StaticCartesianIndices(inds::NTuple{N,R}) where {N,R<:Int}
    # convert divisors to SignedMultiplicativeInverse
    # denoms = map(x-> Base.MultiplicativeInverses.SignedMultiplicativeInverse(x), inds)
    # Expose constant divisors to the compiler:
    denoms = inds
    StaticCartesianIndices{N, inds, denoms}()
end;
Base.size(iter::StaticCartesianIndices{N,R}) where {N,R} = R;
Base.length(iter::StaticCartesianIndices) = prod(size(iter));
Base.axes(::StaticCartesianIndices{N,R}) where {N,R} = R;
__tail(::StaticCartesianIndices{N,R,D}) where {N,R,D} = StaticCartesianIndices{N-1,Base.tail(R),Base.tail(D)}()

Base.@propagate_inbounds Base.getindex(iter::StaticCartesianIndices{0,R}) where {R} = CartesianIndex();
@inline function Base.getindex(iter::StaticCartesianIndices{N,R}, I::Vararg{Int, N}) where {N,R}
    # Eagerly do boundscheck before calculating each item of the CartesianIndex so that
    # we can pass `@inbounds` hint to inside the map and generates more efficient SIMD codes (#42115)
    @boundscheck checkbounds(iter, I...)
    index = map(R, I) do r, i
        @inbounds getindex(1:r, i)
    end
    CartesianIndex(index)
end;
function Base.getindex(A::StaticCartesianIndices, I...)
    Base.@_propagate_inbounds_meta
    Base.error_if_canonical_getindex(Base.IndexCartesian(), A, I...)
    _getindex(A, _to_indices(A, (), I)...)
end
_to_indices(A, inds, ::Tuple{}) = ()

to_index(A, i) = to_index(i)
_to_indices1(A::StaticCartesianIndices, inds, I1) = (I1,)
_to_indices1(A::StaticCartesianIndices, inds, I1::StaticCartesianIndices{N, R}) where {N, R} =
    map(y->to_index(A, 1:y), R)
_cutdim(inds, I1) = safe_tail(inds)
safe_tail(t::Tuple) = Base.tail(t)
safe_tail(t::Tuple{}) = ()

# but preserve StaticCartesianIndices{0} as they consume a dimension.
_to_indices1(A, inds, I1::StaticCartesianIndices{0}) = (I1,)

function _to_indices(A, inds, I::Tuple{Any, Vararg{Any}})
    @inline
    head = _to_indices1(A, inds, I[1])
    rest = _to_indices(A, _cutdim(inds, I[1]), Base.tail(I))
    (head..., rest...)
end

function _getindex(A::StaticCartesianIndices, I::Vararg{Int,M}) where M
    @inline
    @boundscheck Base.checkbounds(A, I...) # generally _to_subscript_indices requires bounds checking
    @inbounds r = Base.getindex(A, _to_subscript_indices(A, I...)...)
    r
end

_to_subscript_indices(A::StaticCartesianIndices, i::Integer) = (@inline; _unsafe_ind2sub(A, i))
_to_subscript_indices(A::StaticCartesianIndices{N}, I::Vararg{Int,N}) where {N} = I
_to_subscript_indices(A::StaticCartesianIndices{0}, i::Integer) = ()
_to_subscript_indices(A::StaticCartesianIndices{0}, I::Integer...) = ()

_unsafe_ind2sub(::Tuple{}, i) = () # _ind2sub may throw(BoundsError()) in this case
_unsafe_ind2sub(sz, i) = (@inline; _ind2sub(sz, i))

_ind2sub(inds::StaticCartesianIndices, ind::Integer) =
    (@inline; _ind2sub_recurse(inds, ind-1))

@inline _ind2sub_recurse(::StaticCartesianIndices{0}, ind) = (ind+1,)
@inline _ind2sub_recurse(indslast::StaticCartesianIndices{1,R}, ind) where {R} = (ind+1,)
@inline function _ind2sub_recurse(inds::StaticCartesianIndices{N,R,D}, ind) where {N, R,D}
    r1 = R[1]
    # Call `div` with a either a
    #  constant divisor `Int` or `SignedMultiplicativeInverse`.
    indnext, l = div(ind, D[1]), r1
    (ind-l*indnext+1, _ind2sub_recurse(__tail(inds), indnext)...)
end

function Base.show(io::IO, iter::StaticCartesianIndices)
    print(io, "StaticCartesianIndices(")
    show(io, map(_xform_index, map(x->Base.OneTo(x), size(iter))))
    print(io, ")")
end
_xform_index(i) = i
_xform_index(i::Base.OneTo) = i.stop
Base.show(io::IO, ::MIME"text/plain", iter::StaticCartesianIndices) = show(io, iter)

#! format: on
