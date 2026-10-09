"""
    AbstractLazyOperator

Supertype for "lazy operators", i.e., operators that users can call without any
arguments, as long as they appear in broadcast expressions that contain at least
one `Field`. If `lazy_op` is an `AbstractLazyOperator`, the expression `lazy_op.()`
is translated to `non_lazy_op.(fields...)` when it appears in a broadcast
expression with at least one `Field`. This translation is done by the function
[`replace_lazy_operator`](@ref)`(space, lazy_op)`, which every subtype of
`AbstractLazyOperator` must implement.
"""
abstract type AbstractLazyOperator end

struct LazyOperatorStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcasted(op::AbstractLazyOperator) =
    Base.Broadcast.broadcasted(LazyOperatorStyle(), op)

# Broadcasting over an AbstractLazyOperator and either a Ref, a Tuple, a Field,
# an Operator, or another AbstractLazyOperator involves using LazyOperatorStyle.
Base.Broadcast.BroadcastStyle(
    ::LazyOperatorStyle,
    ::Union{
        Base.Broadcast.AbstractArrayStyle{0},
        Base.Broadcast.Style{Tuple},
        Fields.AbstractFieldStyle,
        LazyOperatorStyle,
    },
) = LazyOperatorStyle()

# A broadcast expression that contains lazy operators, which is turned into a
# Base.Broadcast.Broadcasted once the lazy operators are replaced. Using a type
# that is distinct from Base.Broadcast.Broadcasted means that the methods of
# materialize below do not invalidate Base's methods for Broadcasted.
struct LazyOperatorBroadcasted{F, A} <: Base.AbstractBroadcasted
    f::F
    args::A
end
Base.Broadcast.BroadcastStyle(::Type{<:LazyOperatorBroadcasted}) = LazyOperatorStyle()
Base.Broadcast.broadcastable(bc::LazyOperatorBroadcasted) = bc

# TODO: This definition of Base.Broadcast.broadcasted results in 2 additional
# method invalidations when using Julia 1.8.5. However, if we were to delete it,
# we would also need to replace the following specializations on
# LazyOperatorBroadcasted with specializations on Base.Broadcast.Broadcasted.
# Specifically, we would need to modify Base.Broadcast.materialize so that it
# specializes on Base.Broadcast.Broadcasted{LazyOperatorStyle}, and this would
# result in 11 invalidations instead of 2.
Base.Broadcast.broadcasted(::LazyOperatorStyle, f::F, args...) where {F} =
    LazyOperatorBroadcasted(f, args)

function Base.Broadcast.materialize(bc::LazyOperatorBroadcasted)
    space = largest_space(bc)
    isnothing(space) && error("Cannot materialize broadcast expression with \
                               AbstractLazyOperator because it does not contain any Fields")
    return Base.Broadcast.materialize(replace_lazy_operators(space, bc))
end

Base.Broadcast.materialize!(dest::Fields.Field, bc::LazyOperatorBroadcasted) =
    Base.Broadcast.materialize!(dest, replace_lazy_operators(axes(dest), bc))

replace_lazy_operators(_, arg) = arg
replace_lazy_operators(space, bc::LazyOperatorBroadcasted) =
    bc.f isa AbstractLazyOperator ? replace_lazy_operator(space, bc.f) :
    Base.Broadcast.broadcasted(
        bc.f,
        unrolled_map(Base.Fix1(replace_lazy_operators, space), bc.args)...,
    )

"""
    replace_lazy_operator(space, lazy_op)

Return a `LazyField` that corresponds to the expression `lazy_op.()`, where the
broadcast in which this expression appears is evaluated on the given `space`.
The staggering (`CellCenter` or `CellFace`) of this `space` depends on the
specifics of the broadcast and is not predetermined.
"""
replace_lazy_operator(_, ::AbstractLazyOperator) =
    error("Every subtype of AbstractLazyOperator must implement a method for
           replace_lazy_operator(space, lazy_op)")

# The largest space in a broadcast expression with lazy operators, found like
# the shared space of an ordinary broadcast (see Fields.shared_space). Spaces on
# the same grid with different staggerings are interchangeable here, since
# replace_lazy_operator only depends on the grid of its space.
largest_space(arg) = Fields.shared_space(arg)
largest_space(bc::LazyOperatorBroadcasted) =
    unrolled_reduce(bc.args; init = nothing) do space, arg
        arg_space = largest_space(arg)
        !isnothing(arg_space) &&
        (isnothing(space) || Spaces.maybe_issubspace(space, arg_space)) ?
        arg_space : space
    end
