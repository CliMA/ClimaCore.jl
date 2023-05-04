"""
    AbstractOperator

Supertype for ClimaCore operators. An operator is a pseudo-function: it can't be
called directly, but can be broadcasted over `Field`s.
"""
abstract type AbstractOperator end


"""
    return_space(::Op, spaces...)

Defines the space upon which the operator `Op` returns given arguments on input
`spaces`.
"""
function return_space end


abstract type OperatorBroadcasted{Style} <: Base.AbstractBroadcasted end

Base.Broadcast.BroadcastStyle(
    ::Type{<:OperatorBroadcasted{Style}},
) where {Style} = Style()


# recursively unwrap axes broadcast arguments in a way that is statically reducible by the optimizer
@inline axes_args(args::Tuple) = (axes(args[1]), axes_args(Base.tail(args))...)
@inline axes_args(arg::Tuple{Any}) = (axes(arg[1]),)
@inline axes_args(::Tuple{}) = ()

@inline instantiate_args(args::Tuple) =
    (Base.Broadcast.instantiate(args[1]), instantiate_args(Base.tail(args))...)
@inline instantiate_args(args::Tuple{Any}) =
    (Base.Broadcast.instantiate(args[1]),)
@inline instantiate_args(::Tuple{}) = ()

function Base.axes(opbc::OperatorBroadcasted)
    if isnothing(opbc.axes)
        return_space(opbc.op, axes_args(opbc.args)...)
    else
        opbc.axes
    end
end
function Base.similar(opbc::OperatorBroadcasted, ::Type{Eltype}) where {Eltype}
    space = axes(opbc)
    return Field(Eltype, space)
end
function Base.copy(opbc::OperatorBroadcasted)
    # figure out return type
    dest = similar(opbc, eltype(opbc))
    # allocate dest
    copyto!(dest, opbc)
end
Base.Broadcast.broadcastable(opbc::OperatorBroadcasted) = opbc

function Base.Broadcast.materialize(opbc::OperatorBroadcasted)
    copy(Base.Broadcast.instantiate(opbc))
end

function Base.Broadcast.materialize!(dest, opbc::OperatorBroadcasted)
    copyto!(dest, Base.Broadcast.instantiate(opbc))
end
