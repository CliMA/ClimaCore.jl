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
struct PlaceholderSpace <: Spaces.AbstractSpace end
struct CenterPlaceholderSpace <: Spaces.AbstractSpace end
struct FacePlaceholderSpace <: Spaces.AbstractSpace end


placeholder_space(current_space::T, parent_space::T) where {T} =
    PlaceholderSpace()
placeholder_space(current_space, parent_space) = current_space
placeholder_space(
    current_space::Spaces.CenterFiniteDifferenceSpace,
    parent_space::Spaces.FaceFiniteDifferenceSpace,
) = CenterPlaceholderSpace()
placeholder_space(
    current_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
) = CenterPlaceholderSpace()
placeholder_space(
    current_space::Spaces.FaceFiniteDifferenceSpace,
    parent_space::Spaces.CenterFiniteDifferenceSpace,
) = FacePlaceholderSpace()
placeholder_space(
    current_space::Spaces.FaceExtrudedFiniteDifferenceSpace,
    parent_space::Spaces.CenterExtrudedFiniteDifferenceSpace,
) = FacePlaceholderSpace()

@inline reconstruct_placeholder_space(::PlaceholderSpace, parent_space) =
    parent_space
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
@inline reconstruct_placeholder_space(current_space, parent_space) =
    current_space


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

struct PlaceHolderLocalGeometry end
isa_local_geometry(f::Fields.Field) = eltype(f) <: Geometry.LocalGeometry
isa_local_geometry(x) = false
strip_local_geometry(f::Fields.Field) = isa_local_geometry(f) ? PlaceHolderLocalGeometry() : f
strip_local_geometry(x) = x

@inline _replace_placeholder_local_geometry(arg, lg) = arg
@inline _replace_placeholder_local_geometry(arg::PlaceHolderLocalGeometry, lg::Geometry.LocalGeometry) = lg
@inline _replace_placeholder_local_geometry(args::Tuple, lg) =
    (_replace_placeholder_local_geometry(first(args), lg),
        _replace_placeholder_local_geometry(Base.tail(args), lg)...)
@inline _replace_placeholder_local_geometry(args::Tuple{Any}, lg) =
    (_replace_placeholder_local_geometry(first(args), lg),)
@inline _replace_placeholder_local_geometry(args::Tuple{}, lg) = ()

# import .Utilities.UnrolledFunctions: unrolled_map
@inline function replace_placeholder_local_geometry(args::Tuple, lg)
    map(args) do a
        Base.@_inline_meta
        if a isa PlaceHolderLocalGeometry
            lg
        else
            a
        end
    end
end
@inline has_placeholder_local_geometry(args::Tuple) = has_placeholder_local_geometry(false, args)
@inline has_placeholder_local_geometry(found, ::Tuple{}) = found
@inline has_placeholder_local_geometry(found, arg) = found | (arg isa PlaceHolderLocalGeometry)
@inline has_placeholder_local_geometry(found, args::Tuple) =
    found | has_placeholder_local_geometry(found, Base.first(args)) | has_placeholder_local_geometry(found, Base.tail(args))

@inline function maybe_call_modified_bc(f::F, args, lg) where {F}
    if has_placeholder_local_geometry(args)
        call_modified_bc(f, args, lg)
    else
        return f(args...)
    end
end

@inline function call_modified_bc(f::F, args, lg) where {F}
    args′ = replace_placeholder_local_geometry(args, lg)
    return f(args′...)
end

function strip_local_geometry(bc::Base.Broadcast.Broadcasted{Style}) where {Style}
    return Base.Broadcast.Broadcasted{Style}(
        bc.f,
        strip_local_geometry_args(bc.args),
        bc.axes,
    )
end
strip_local_geometry_args(::Tuple{}) = ()
strip_local_geometry_args(args::Tuple{Any}) =
    (strip_local_geometry(args[1]), )
strip_local_geometry_args(args::Tuple) =
    (strip_local_geometry(args[1]), strip_local_geometry_args(Base.tail(args))...)

append_local_geometry(::Type{<:Geometry.CartesianVector}) = true
append_local_geometry(::Type{<:Geometry.AxisVector}) = true
append_local_geometry(::typeof(Geometry.project)) = true
append_local_geometry(::typeof(Geometry.transform)) = true
append_local_geometry(::typeof(LinearAlgebra.norm)) = true
append_local_geometry(::typeof(LinearAlgebra.norm_sqr)) = true
append_local_geometry(::typeof(LinearAlgebra.cross)) = true
append_local_geometry(x) = false
