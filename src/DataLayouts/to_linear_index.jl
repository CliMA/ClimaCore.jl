_to_linear_index(A::AbstractArray, li, ci) = _to_linear_index(A, Base.to_indices(li, (ci,))...)
_to_linear_index(A::AbstractArray, I::Integer...) = (@inline; _sub2ind(A, I...))

function _sub2ind(A::AbstractArray, I...)
    @inline
    _sub2ind(axes(A), I...)
end

# 0-dimensional arrays and indexing with []
_sub2ind(::Tuple{}) = 1
_sub2ind(::Base.DimsInteger) = 1
# _sub2ind(::Indices) = 1
_sub2ind(::Tuple{}, I::Integer...) = (@inline; _sub2ind_recurse((), 1, 1, I...))

# Generic cases
_sub2ind(dims::Base.DimsInteger, I::Integer...) = (@inline; _sub2ind_recurse(dims, 1, 1, I...))
_sub2ind(inds::Base.Indices, I::Integer...) = (@inline; _sub2ind_recurse(inds, 1, 1, I...))
# In 1d, there's a question of whether we're doing cartesian indexing
# or linear indexing. Support only the former.
_sub2ind(inds::Base.Indices{1}, I::Integer...) =
    throw(ArgumentError("Linear indexing is not defined for one-dimensional arrays"))
_sub2ind(inds::Tuple{Base.OneTo}, I::Integer...) = (@inline; _sub2ind_recurse(inds, 1, 1, I...)) # only OneTo is safe
_sub2ind(inds::Tuple{Base.OneTo}, i::Integer)    = i

_sub2ind_recurse(::Any, L, ind) = ind
function _sub2ind_recurse(::Tuple{}, L, ind, i::Integer, I::Integer...)
    @inline
    _sub2ind_recurse((), L, ind+(i-1)*L, I...)
end
function _sub2ind_recurse(inds, L, ind, i::Integer, I::Integer...)
    @inline
    r1 = inds[1]
    _sub2ind_recurse(Base.tail(inds), nextL(L, r1), ind+offsetin(i, r1)*L, I...)
end

nextL(L, l::Integer) = L*l
nextL(L, r::AbstractUnitRange) = L*length(r)
offsetin(i, l::Integer) = i-1
offsetin(i, r::AbstractUnitRange) = i-first(r)
