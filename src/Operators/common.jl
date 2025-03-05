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


# when sending Broadcasted objects to the GPU, we strip out the space
# information at each level of the broadcast tree and Fields, replacing with
# PlaceholderSpace. this reduces the amount runtime parameter data we send to
# the GPU, which is quite limited (~4kB).

# Functions for CUDASpectralStyle
struct PlaceholderSpace{N, IP} <: Spaces.AbstractSpace end
struct LevelPlaceholderSpace{N, IP} <: Spaces.AbstractSpace end
struct CenterPlaceholderSpace{N, IP} <: Spaces.AbstractSpace end
struct FacePlaceholderSpace{N, IP} <: Spaces.AbstractSpace end
Spaces.nlevels(::FacePlaceholderSpace{N}) where {N} = N
Spaces.nlevels(::CenterPlaceholderSpace{N}) where {N} = N

PlaceholderSpace(space) = PlaceholderSpace{
    Spaces.nlevels(space),
    Topologies.isperiodic(Spaces.vertical_topology(space)),
}()
LevelPlaceholderSpace(space) = LevelPlaceholderSpace{
    Spaces.nlevels(space),
    Topologies.isperiodic(Spaces.vertical_topology(space)),
}()
CenterPlaceholderSpace(space) = CenterPlaceholderSpace{
    Spaces.nlevels(space),
    Topologies.isperiodic(Spaces.vertical_topology(space)),
}()
FacePlaceholderSpace(space) = FacePlaceholderSpace{
    Spaces.nlevels(space),
    Topologies.isperiodic(Spaces.vertical_topology(space)),
}()

placeholder_space(current_space, parent_space) = current_space
placeholder_space(current_space::T, parent_space::T) where {T} =
    PlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.AbstractPointSpace,
    parent_space::Spaces.AbstractFiniteDifferenceSpace,
) = LevelPlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.AbstractSpectralElementSpace,
    parent_space::Spaces.ExtrudedFiniteDifferenceSpace,
) = LevelPlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.CenterFiniteDifferenceSpace,
    parent_space::Spaces.FaceFiniteDifferenceSpace,
) = CenterPlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = CenterPlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.FaceFiniteDifferenceSpace,
    parent_space::Spaces.CenterFiniteDifferenceSpace,
) = FacePlaceholderSpace(current_space)
placeholder_space(
    current_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = FacePlaceholderSpace(current_space)

@inline reconstruct_placeholder_space(current_space, parent_space) =
    current_space
@inline reconstruct_placeholder_space(::PlaceholderSpace, parent_space) =
    parent_space
@inline reconstruct_placeholder_space(::LevelPlaceholderSpace, parent_space) =
    Spaces.level(parent_space, left_idx(parent_space)) # extract arbitrary level
@inline reconstruct_placeholder_space(
    ::CenterPlaceholderSpace,
    parent_space::Spaces.FaceFiniteDifferenceSpace,
) = Spaces.CenterFiniteDifferenceSpace(parent_space)
@inline reconstruct_placeholder_space(
    ::CenterPlaceholderSpace,
    parent_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = Spaces.CenterExtrudedFiniteDifferenceSpace(parent_space)
@inline reconstruct_placeholder_space(
    ::FacePlaceholderSpace,
    parent_space::Spaces.CenterFiniteDifferenceSpace,
) = Spaces.FaceFiniteDifferenceSpace(parent_space)
@inline reconstruct_placeholder_space(
    ::FacePlaceholderSpace,
    parent_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = Spaces.FaceExtrudedFiniteDifferenceSpace(parent_space)

strip_space(obj, parent_space) = obj

function strip_space(field::Field, parent_space)
    current_space = axes(field)
    new_space = placeholder_space(current_space, parent_space)
    return Field(Fields.field_values(field), new_space)
end

function strip_space(
    bc::Base.Broadcast.Broadcasted{Style},
    parent_space,
) where {Style}
    current_space = axes(bc)
    new_space = placeholder_space(current_space, parent_space)
    return Base.Broadcast.Broadcasted{Style}(
        bc.f,
        strip_space_args(bc.args, current_space),
        new_space,
    )
end

strip_space_args(::Tuple{}, space) = ()
strip_space_args(args::Tuple, space) =
    (strip_space(args[1], space), strip_space_args(Base.tail(args), space)...)

function unstrip_space(field::Field, parent_space)
    new_space = reconstruct_placeholder_space(axes(field), parent_space)
    return Field(Fields.field_values(field), new_space)
end
