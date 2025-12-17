#! format: off
# ============================================================ Adapted from Base.Broadcast (julia version 1.10.4)
import Base.Broadcast: BroadcastStyle
import UnrolledUtilities
struct NonExtrudedBroadcasted{
    Style <: Union{Nothing, BroadcastStyle},
    Axes,
    F,
    Args <: Tuple,
} <: Base.AbstractBroadcasted
    style::Style
    f::F
    args::Args
    axes::Axes          # the axes of the resulting object (may be bigger than implied by `args` if this is nested inside a larger `NonExtrudedBroadcasted`)

    NonExtrudedBroadcasted(style::Union{Nothing, BroadcastStyle}, f::Tuple, args::Tuple) =
        error() # disambiguation: tuple is not callable
    function NonExtrudedBroadcasted(
        style::Union{Nothing, BroadcastStyle},
        f::F,
        args::Tuple,
        axes = nothing,
    ) where {F}
        # using Core.Typeof rather than F preserves inferrability when f is a type
        return new{typeof(style), typeof(axes), Core.Typeof(f), typeof(args)}(
            style,
            f,
            args,
            axes,
        )
    end
    function NonExtrudedBroadcasted(f::F, args::Tuple, axes = nothing) where {F}
        NonExtrudedBroadcasted(combine_styles(args...)::BroadcastStyle, f, args, axes)
    end
    function NonExtrudedBroadcasted{Style}(f::F, args, axes = nothing) where {Style, F}
        return new{Style, typeof(axes), Core.Typeof(f), typeof(args)}(
            Style()::Style,
            f,
            args,
            axes,
        )
    end
    function NonExtrudedBroadcasted{Style, Axes, F, Args}(
        f,
        args,
        axes,
    ) where {Style, Axes, F, Args}
        return new{Style, Axes, F, Args}(Style()::Style, f, args, axes)
    end
end

@inline to_broadcasted(bc::NonExtrudedBroadcasted) =
    Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, bc.axes)
@inline to_non_extruded_broadcasted(bc::Base.Broadcast.Broadcasted) =
    NonExtrudedBroadcasted(bc.style, bc.f, to_non_extruded_broadcasted_args(bc.args), bc.axes)
@inline to_non_extruded_broadcasted(x) = x

@inline function to_non_extruded_broadcasted_args(args::Tuple)
    UnrolledUtilities.unrolled_map(args) do arg
        to_non_extruded_broadcasted(arg)
    end
end
@inline to_non_extruded_broadcasted_args(args::Tuple{Any}) =
    (to_non_extruded_broadcasted(args[1]),)
@inline to_non_extruded_broadcasted_args(args::Tuple{}) = ()

# CartesianIndex{0} is used for DataF and empty data cases
# And sometimes axes(bc) returns a (e.g.,) CenterFiniteDifferenceSpace
# However, this is currently only being used for pointwise
# kernels. So, for now, we always call `todata` on the broadcasted
# object to forward pointwise kernels to be handled in datalayouts.
# Therefore, `axes` here should always return a tuple of ranges.

# If we extend this, then we'll need to define _checkbounds to
# extract the resulting tuple of ranges for a field axes (which
# returns a space)
# @inline _checkbounds(bc, _, I::Union{Integer, CartesianIndex{0}}) = nothing
# CartesianIndex{0} is used for DataF and empty data cases
@inline _checkbounds(bc, ::Tuple, I::Union{Integer, CartesianIndex{0}}) = Base.checkbounds(bc, I)
@inline function Base.getindex(
    bc::NonExtrudedBroadcasted,
    I::Union{Integer, CartesianIndex},
)
    @boundscheck _checkbounds(bc, axes(bc), I)
    @inbounds _broadcast_getindex(bc, I)
end

# --- here, we define our own bounds checks
@inline function Base.checkbounds(bc::NonExtrudedBroadcasted, I::Union{Integer, CartesianIndex{0}})
    # Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,)) # from Base
    N = n_dofs(bc)
    # edge case: N == 0 means we have an empty field
    if N == 0 || !Base.checkbounds_indices(Bool, (Base.OneTo(N),), (I,))
        Base.throw_boundserror(bc, (I,))
    end
end
# getindex on DefaultArrayStyle{0} ignores the
# index value, so this should always be safe
@inline Base.checkbounds(bc::NonExtrudedBroadcasted{Style}, I::Union{Integer, CartesianIndex{0}}) where {Style <: Base.Broadcast.DefaultArrayStyle{0}} = nothing
@inline Base.checkbounds(bc::NonExtrudedBroadcasted{Style}, I::Union{Integer, CartesianIndex{0}}) where {Style <: Base.Broadcast.Style{Tuple}} = nothing


# To handle scalar cases, let's just switch back to
# Base.Broadcast.Broadcasted and allow cartesian indexing:
Base.@propagate_inbounds Base.getindex(bc::NonExtrudedBroadcasted) = bc[CartesianIndex(())]

n_dofs(bc::NonExtrudedBroadcasted) = prod(length, axes(bc); init = 1)
# ---

Base.@propagate_inbounds _broadcast_getindex(A, I::CartesianIndex{0}) = A[] # Scalar-likes (e.g., DataF) can just ignore all indices
Base.@propagate_inbounds _broadcast_getindex(
    A::Union{Ref, AbstractArray{<:Any, 0}, Number},
    I::Integer,
) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds _broadcast_getindex(
    ::Ref{Type{T}},
    I::Integer,
) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds _broadcast_getindex(A::Tuple{Any}, I::Integer) = A[1]
Base.@propagate_inbounds _broadcast_getindex(A::Tuple, I::Integer) = A[I[1]]
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
# Base.@propagate_inbounds _broadcast_getindex(A, I) = A[newindex(A, I)]
# Base.@propagate_inbounds _broadcast_getindex(A, I::Integer) = A[I]
Base.@propagate_inbounds function _broadcast_getindex(A, I::Integer)
    A[I]
end
Base.@propagate_inbounds function _broadcast_getindex(
    bc::NonExtrudedBroadcasted{<:Any, <:Any, <:Any, <:Any},
    I::Integer,
)
    args = _getindex(bc.args, I)
    return _broadcast_getindex_evalf(bc.f, args...)
end
# CartesianIndex{0} is used for DataF and empty data cases:
Base.@propagate_inbounds function _broadcast_getindex(
    bc::NonExtrudedBroadcasted{<:Any, <:Any, <:Any, <:Any},
    I::CartesianIndex{0},
)
    args = _getindex(bc.args, I)
    return _broadcast_getindex_evalf(bc.f, args...)
end
@inline _broadcast_getindex_evalf(f::Tf, args::Vararg{Any, N}) where {Tf, N} =
    f(args...)  # not propagate_inbounds
Base.@propagate_inbounds function _getindex(args::Tuple, I)
    UnrolledUtilities.unrolled_map(args) do arg
        _broadcast_getindex(arg, I)
    end
end
Base.@propagate_inbounds _getindex(args::Tuple{Any}, I) =
    (_broadcast_getindex(args[1], I),)
Base.@propagate_inbounds _getindex(args::Tuple{}, I) = ()

@inline Base.axes(bc::NonExtrudedBroadcasted) = _axes(bc, bc.axes)
_axes(::NonExtrudedBroadcasted, axes::Tuple) = axes
@inline _axes(bc::NonExtrudedBroadcasted, ::Nothing) = Base.Broadcast.combine_axes(bc.args...)
_axes(bc::NonExtrudedBroadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}, ::Nothing) = ()
@inline Base.axes(bc::NonExtrudedBroadcasted{<:Any, <:NTuple{N}}, d::Integer) where {N} =
    d <= N ? axes(bc)[d] : OneTo(1)
Base.IndexStyle(::Type{<:NonExtrudedBroadcasted{<:Any, <:Tuple{Any}}}) = IndexLinear()
@inline _axes(::NonExtrudedBroadcasted, axes) = axes
@inline Base.eltype(bc::NonExtrudedBroadcasted) = Base.Broadcast.combine_axes(bc.args...)


# ============================================================

#! format: on
