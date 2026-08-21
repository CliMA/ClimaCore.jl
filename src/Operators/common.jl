import UnrolledUtilities: unrolled_map

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
@inline axes_args(args::Tuple) = unrolled_map(axes, args)

@inline instantiate_args(args::Tuple) =
    unrolled_map(Base.Broadcast.instantiate, args)

function Base.axes(opbc::OperatorBroadcasted)
    if isnothing(opbc.axes)
        return_space(opbc.op, axes_args(opbc.args)...)
    else
        opbc.axes
    end
end
Base.Broadcast.broadcastable(opbc::OperatorBroadcasted) = opbc
Base.copy(opbc::OperatorBroadcasted) = copyto!(similar(opbc), opbc)
Base.similar(opbc::OperatorBroadcasted, ::Type{Eltype}) where {Eltype} =
    Field(Eltype, axes(opbc))

# Define similar to match DataStyle and AbstractFieldStyle broadcasting
Base.similar(opbc::OperatorBroadcasted) =
    similar(opbc, drop_auto_broadcasters(eltype(opbc)))

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
struct PlaceholderSpace <: Spaces.AbstractSpace end
struct LevelPlaceholderSpace <: Spaces.AbstractSpace end
struct CenterPlaceholderSpace <: Spaces.AbstractSpace end
struct FacePlaceholderSpace <: Spaces.AbstractSpace end

placeholder_space(current_space, parent_space) = current_space
placeholder_space(current_space::T, parent_space::T) where {T} =
    PlaceholderSpace()
placeholder_space(
    current_space::Spaces.AbstractPointSpace,
    parent_space::Spaces.AbstractFiniteDifferenceSpace,
) = LevelPlaceholderSpace()
placeholder_space(
    current_space::Spaces.AbstractSpectralElementSpace,
    parent_space::Spaces.ExtrudedFiniteDifferenceSpace,
) = LevelPlaceholderSpace()
placeholder_space(
    current_space::Spaces.CenterFiniteDifferenceSpace,
    parent_space::Spaces.FaceFiniteDifferenceSpace,
) = CenterPlaceholderSpace()
placeholder_space(
    current_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = CenterPlaceholderSpace()
placeholder_space(
    current_space::Spaces.CenterMultiColumnFiniteDifferenceSpace,
    parent_space::Spaces.FaceMultiColumnFiniteDifferenceSpace,
) = CenterPlaceholderSpace()
placeholder_space(
    current_space::Spaces.FaceFiniteDifferenceSpace,
    parent_space::Spaces.CenterFiniteDifferenceSpace,
) = FacePlaceholderSpace()
placeholder_space(
    current_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = FacePlaceholderSpace()
placeholder_space(
    current_space::Spaces.FaceMultiColumnFiniteDifferenceSpace,
    parent_space::Spaces.CenterMultiColumnFiniteDifferenceSpace,
) = FacePlaceholderSpace()

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
@inline reconstruct_placeholder_space(
    ::FacePlaceholderSpace,
    parent_space::Union{
        Spaces.FaceFiniteDifferenceSpace,
        Spaces.FaceExtrudedFiniteDifferenceSpace,
    },
) = parent_space
@inline reconstruct_placeholder_space(
    ::CenterPlaceholderSpace,
    parent_space::Union{
        Spaces.CenterFiniteDifferenceSpace,
        Spaces.CenterExtrudedFiniteDifferenceSpace,
    },
) = parent_space

@inline reconstruct_placeholder_space(
    ::CenterPlaceholderSpace,
    parent_space::Spaces.FaceMultiColumnFiniteDifferenceSpace,
) = Spaces.CenterMultiColumnFiniteDifferenceSpace(parent_space)
@inline reconstruct_placeholder_space(
    ::FacePlaceholderSpace,
    parent_space::Spaces.CenterMultiColumnFiniteDifferenceSpace,
) = Spaces.FaceMultiColumnFiniteDifferenceSpace(parent_space)
@inline reconstruct_placeholder_space(
    ::FacePlaceholderSpace,
    parent_space::Spaces.FaceMultiColumnFiniteDifferenceSpace,
) = parent_space
@inline reconstruct_placeholder_space(
    ::CenterPlaceholderSpace,
    parent_space::Spaces.CenterMultiColumnFiniteDifferenceSpace,
) = parent_space

strip_space(arg, parent_space) = arg
function strip_space(field::Field, parent_space)
    space = placeholder_space(axes(field), parent_space)
    return Field(Fields.field_values(field), space)
end
function strip_space(bc::Broadcast.Broadcasted, parent_space)
    args = unrolled_map(Base.Fix2(strip_space, axes(bc)), bc.args)
    space = placeholder_space(axes(bc), parent_space)
    return Broadcast.Broadcasted(bc.style, bc.f, args, space)
end

unstrip_space(arg, parent_space) = arg
function unstrip_space(field::Field, parent_space)
    space = reconstruct_placeholder_space(axes(field), parent_space)
    return Field(Fields.field_values(field), space)
end
function unstrip_space(bc::Broadcast.Broadcasted, parent_space)
    args = unrolled_map(Base.Fix2(unstrip_space, axes(bc)), bc.args)
    space = reconstruct_placeholder_space(axes(bc), parent_space)
    return Broadcast.Broadcasted(bc.style, bc.f, args, space)
end
