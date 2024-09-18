#! format: off
# ============================================================ Adapted from Base.Broadcast (julia version 1.10.4)
import Base.Broadcast: BroadcastStyle
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

@inline to_non_extruded_broadcasted(bc::Base.Broadcast.Broadcasted) =
    NonExtrudedBroadcasted(bc.style, bc.f, to_non_extruded_broadcasted(bc.args), bc.axes)
@inline to_non_extruded_broadcasted(x) = x
NonExtrudedBroadcasted(bc::Base.Broadcast.Broadcasted) = to_non_extruded_broadcasted(bc)

@inline to_non_extruded_broadcasted(args::Tuple) = (
    to_non_extruded_broadcasted(args[1]),
    to_non_extruded_broadcasted(Base.tail(args))...,
)
@inline to_non_extruded_broadcasted(args::Tuple{Any}) =
    (to_non_extruded_broadcasted(args[1]),)
@inline to_non_extruded_broadcasted(args::Tuple{}) = ()

@inline _checkbounds(bc, _, I) = nothing # TODO: fix this case
@inline _checkbounds(bc, ::Tuple, I) = Base.checkbounds(bc, I)
@inline function Base.getindex(
    bc::NonExtrudedBroadcasted,
    I::Union{Integer, CartesianIndex},
)
    @boundscheck _checkbounds(bc, axes(bc), I) # is this really the only issue?
    @inbounds _broadcast_getindex(bc, I)
end

# --- here, we define our own bounds checks
@inline function Base.checkbounds(bc::NonExtrudedBroadcasted, I::Integer)
    # Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,)) # from Base
    Base.checkbounds_indices(Bool, (Base.OneTo(n_dofs(bc)),), (I,)) || Base.throw_boundserror(bc, (I,))
end

import StaticArrays
to_tuple(t::Tuple) = t
to_tuple(t::NTuple{N, <: Base.OneTo}) where {N} = map(x->x.stop, t)
to_tuple(t::NTuple{N, <: StaticArrays.SOneTo}) where {N} = map(x->x.stop, t)
n_dofs(bc::NonExtrudedBroadcasted) = prod(to_tuple(axes(bc)))
# ---

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
Base.@propagate_inbounds _broadcast_getindex(A, I::Integer) = A[I]
Base.@propagate_inbounds function _broadcast_getindex(
    bc::NonExtrudedBroadcasted{<:Any, <:Any, <:Any, <:Any},
    I::Integer,
)
    args = _getindex(bc.args, I)
    return _broadcast_getindex_evalf(bc.f, args...)
end
@inline _broadcast_getindex_evalf(f::Tf, args::Vararg{Any, N}) where {Tf, N} =
    f(args...)  # not propagate_inbounds
Base.@propagate_inbounds _getindex(args::Tuple, I) =
    (_broadcast_getindex(args[1], I), _getindex(Base.tail(args), I)...)
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
# Datalayouts
@propagate_inbounds function linear_getindex(
    data::AbstractData{S},
    I::Integer,
) where {S}
    s_array = farray_size(data)
    ss = StaticSize(s_array, field_dim(data))
    @inbounds get_struct_linear(parent(data), S, Val(field_dim(data)), I, ss)
end
@propagate_inbounds function linear_setindex!(
    data::AbstractData{S},
    val,
    I::Integer,
) where {S}
    s_array = farray_size(data)
    ss = StaticSize(s_array, field_dim(data))
    @inbounds set_struct_linear!(
        parent(data),
        convert(S, val),
        Val(field_dim(data)),
        I,
        ss,
    )
end

for DL in (:IJKFVH, :IJFH, :IFH, :IJF, :IF, :VF, :VIJFH, :VIFH) # Skip DataF, since we want that to MethodError.
    @eval @propagate_inbounds Base.getindex(data::$(DL), I::Integer) =
        linear_getindex(data, I)
    @eval @propagate_inbounds Base.setindex!(data::$(DL), val, I::Integer) =
        linear_setindex!(data, val, I)
end
