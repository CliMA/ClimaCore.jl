#! format: off
# ============================================================ Adapted from Base.Broadcast (julia version 1.10.4)
@inline function Base.getindex(bc::Base.Broadcast.Broadcasted, I::DataSpecificCartesianIndex)
    @boundscheck checkbounds(bc, I)
    @inbounds _broadcast_getindex(bc, I)
end

# This code path is only ever reached when all datalayouts in
# the broadcasted object are the same (e.g., ::VIJFH, ::VIJFH)
# They may have different type parameters, but this means that
# `permute_axes` will still produce the correct axes for all
# datalayouts.
@inline Base.checkbounds(bc::Base.Broadcast.Broadcasted, I::DataSpecificCartesianIndex) =
    # Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,)) # from Base
    Base.checkbounds_indices(Bool, permute_axes(axes(bc), first_datalayout_in_bc(bc)), (I.I,)) || Base.throw_boundserror(bc, (I,))

Base.@propagate_inbounds _broadcast_getindex(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds _broadcast_getindex(::Ref{Type{T}}, I) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds _broadcast_getindex(A::Tuple{Any}, I) = A[1]
Base.@propagate_inbounds _broadcast_getindex(A::Tuple, I) = A[I[1]]
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
# Base.@propagate_inbounds _broadcast_getindex(A, I) = A[Base.Broadcast.newindex(A, I)]
Base.@propagate_inbounds _broadcast_getindex(A, I) = A[I]

# For Broadcasted
Base.@propagate_inbounds function _broadcast_getindex(bc::Base.Broadcast.Broadcasted{<:Any,<:Any,<:Any,<:Any}, I)
    args = _getindex(bc.args, I)
    return _broadcast_getindex_evalf(bc.f, args...)
end
# Hack around losing Type{T} information in the final args tuple. Julia actually
# knows (in `code_typed`) the _value_ of these types, statically displaying them,
# but inference is currently skipping inferring the type of the types as they are
# transiently placed in a tuple as the argument list is lispily constructed. These
# additional methods recover type stability when a `Type` appears in one of the
# first two arguments of a function.
Base.@propagate_inbounds function _broadcast_getindex(bc::Base.Broadcast.Broadcasted{<:Any,<:Any,<:Any,<:Tuple{Ref{Type{T}},Vararg{Any}}}, I) where {T}
    args = _getindex(Base.tail(bc.args), I)
    return _broadcast_getindex_evalf(bc.f, T, args...)
end
Base.@propagate_inbounds function _broadcast_getindex(bc::Base.Broadcast.Broadcasted{<:Any,<:Any,<:Any,<:Tuple{Any,Ref{Type{T}},Vararg{Any}}}, I) where {T}
    arg1 = _broadcast_getindex(bc.args[1], I)
    args = _getindex(Base.tail(Base.tail(bc.args)), I)
    return _broadcast_getindex_evalf(bc.f, arg1, T, args...)
end
Base.@propagate_inbounds function _broadcast_getindex(bc::Base.Broadcast.Broadcasted{<:Any,<:Any,<:Any,<:Tuple{Ref{Type{T}},Ref{Type{S}},Vararg{Any}}}, I) where {T,S}
    args = _getindex(Base.tail(Base.tail(bc.args)), I)
    return _broadcast_getindex_evalf(bc.f, T, S, args...)
end

# Utilities for _broadcast_getindex
Base.@propagate_inbounds _getindex(args::Tuple, I) = (_broadcast_getindex(args[1], I), _getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _getindex(args::Tuple{Any}, I) = (_broadcast_getindex(args[1], I),)
Base.@propagate_inbounds _getindex(args::Tuple{}, I) = ()

@inline _broadcast_getindex_evalf(f::Tf, args::Vararg{Any,N}) where {Tf,N} = f(args...)  # not propagate_inbounds
# ============================================================

#! format: on
# Datalayouts
@propagate_inbounds function Base.getindex(
    data::AbstractData{S},
    I::DataSpecificCartesianIndex,
) where {S}
    @inbounds get_struct(parent(data), S, Val(field_dim(data)), I.I)
end
@propagate_inbounds function Base.setindex!(
    data::AbstractData{S},
    val,
    I::DataSpecificCartesianIndex,
) where {S}
    @inbounds set_struct!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        I.I,
    )
end

# Returns the size of the backing array.
@inline array_size(::IJKFVH{S, Nij, Nk, Nv, Nh}) where {S, Nij, Nk, Nv, Nh} =
    (Nij, Nij, Nk, 1, Nv, Nh)
@inline array_size(::IJFH{S, Nij, Nh}) where {S, Nij, Nh} = (Nij, Nij, 1, Nh)
@inline array_size(::IFH{S, Ni, Nh}) where {S, Ni, Nh} = (Ni, 1, Nh)
@inline array_size(::DataF{S}) where {S} = (1,)
@inline array_size(::IJF{S, Nij}) where {S, Nij} = (Nij, Nij, 1)
@inline array_size(::IF{S, Ni}) where {S, Ni} = (Ni, 1)
@inline array_size(::VF{S, Nv}) where {S, Nv} = (Nv, 1)
@inline array_size(::VIJFH{S, Nv, Nij, Nh}) where {S, Nv, Nij, Nh} =
    (Nv, Nij, Nij, 1, Nh)
@inline array_size(::VIFH{S, Nv, Ni, Nh}) where {S, Nv, Ni, Nh} =
    (Nv, Ni, 1, Nh)

#####
##### Helpers to support `Base.checkbounds`
#####

# Converts axes(::AbstractData) to a Data-specific axes
@inline permute_axes(A, data::AbstractData) =
    map(x -> A[x], perm_to_array(data))

# axes for IJF and IF exclude the field dimension
@inline permute_axes(A, ::IJF) = (A[1], A[2], Base.OneTo(1))
@inline permute_axes(A, ::IF) = (A[1], Base.OneTo(1))

# Permute dimensions of size(data) (the universal size) to
# output size of array for example, this should satisfy:
#     @test size(parent(data)) == map(size(data)[i], perm_to_array(data))
@inline perm_to_array(::IJKFVH) = (1, 2, 3, 4, 5)
@inline perm_to_array(::IJFH) = (1, 2, 3, 5)
@inline perm_to_array(::IFH) = (1, 3, 5)
@inline perm_to_array(::DataF) = (3,)
@inline perm_to_array(::VF) = (4, 3)
@inline perm_to_array(::VIJFH) = (4, 1, 2, 3, 5)
@inline perm_to_array(::VIFH) = (4, 1, 3, 5)
