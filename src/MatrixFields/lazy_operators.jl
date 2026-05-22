"""
    AbstractLazyOperator

Supertype for "lazy operators", i.e., operators that can be called without any
arguments by users, as long as they appear in broadcast expressions that contain
at least one `Field`. If `lazy_op` is an `AbstractLazyOperator`, the expression
`lazy_op.()` will internally be translated to `non_lazy_op.(fields...)`, as long
as it appears in a broadcast expression with at least one `Field`. This
translation is done by the function `replace_lazy_operator(space, lazy_op)`,
which must be implemented by every subtype of `AbstractLazyOperator`.
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

struct LazyOperatorBroadcasted{F, A} <:
       Operators.OperatorBroadcasted{LazyOperatorStyle}
    f::F
    args::A
end

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

Generates an instance of `Base.AbstractBroadcasted` that corresponds to the
expression `lazy_op.()`, where the broadcast in which this expression appears is
being evaluated on the given `space`. Note that the staggering (`CellCenter` or
`CellFace`) of this `space` depends on the specifics of the broadcast and is not
predetermined.
"""
replace_lazy_operator(_, ::AbstractLazyOperator) =
    error("Every subtype of AbstractLazyOperator must implement a method for
           replace_lazy_operator(space, lazy_op)")

largest_space(_) = nothing
largest_space(field::Fields.Field) = axes(field)
largest_space(bc::Base.AbstractBroadcasted) =
    unrolled_reduce(larger_space, unrolled_map(largest_space, bc.args); init = nothing)

larger_space(::Nothing, ::Nothing) = nothing
larger_space(space1, ::Nothing) = space1
larger_space(::Nothing, space2) = space2
larger_space(space1::S, ::S) where {S} = space1 # Neither space is larger.
larger_space(
    space1::Spaces.FiniteDifferenceSpace,
    ::Spaces.FiniteDifferenceSpace,
) = space1 # The staggering does not matter here, so neither space is larger.
larger_space(
    space1::Spaces.ExtrudedFiniteDifferenceSpace,
    ::Spaces.ExtrudedFiniteDifferenceSpace,
) = space1 # The staggering does not matter here, so neither space is larger.
larger_space(
    space1::Spaces.ExtrudedFiniteDifferenceSpace,
    ::Spaces.FiniteDifferenceSpace,
) = space1 # The types indicate that space2 is a subspace of space1.
larger_space(
    ::Spaces.FiniteDifferenceSpace,
    space2::Spaces.ExtrudedFiniteDifferenceSpace,
) = space2 # The types indicate that space1 is a subspace of space2.
larger_space(
    space1::Spaces.ExtrudedFiniteDifferenceSpace,
    ::Spaces.AbstractSpectralElementSpace,
) = space1 # The types indicate that space2 is a subspace of space1.
larger_space(
    ::Spaces.AbstractSpectralElementSpace,
    space2::Spaces.ExtrudedFiniteDifferenceSpace,
) = space2 # The types indicate that space1 is a subspace of space2.
larger_space(::S1, ::S2) where {S1, S2} =
    error("Mismatched spaces ($(S1.name.name) and $(S2.name.name))")
