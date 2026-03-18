import ..DebugOnly: allow_mismatched_spaces_unsafe
import UnrolledUtilities: unrolled_map

"""
    AbstractFieldStyle

The supertype of all broadcasting-like operations on Fields.
"""
abstract type AbstractFieldStyle <: Base.BroadcastStyle end

"""
    FieldStyle{DS <: DataStyle}

Standard broadcasting on Fields. Delegates the actual work to `DS`.
"""
struct FieldStyle{DS <: DataStyle} <: AbstractFieldStyle end

FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()
FieldStyle(x::Base.Broadcast.Unknown) = x

FieldLevelStyle(::Type{S}) where {DS, S <: FieldStyle{DS}} =
    FieldStyle{DataLayouts.DataLevelStyle(DS)}
FieldColumnStyle(::Type{S}) where {DS, S <: FieldStyle{DS}} =
    FieldStyle{DataLayouts.DataColumnStyle(DS)}
FieldSlabStyle(::Type{S}) where {DS, S <: FieldStyle{DS}} =
    FieldStyle{DataLayouts.DataSlabStyle(DS)}

Base.Broadcast.BroadcastStyle(::Type{Field{V, S}}) where {V, S} =
    FieldStyle(DataStyle(V))

# Broadcasting over scalars (Ref or Tuple)
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    fs::AbstractFieldStyle,
) = fs
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.Style{Tuple},
    fs::AbstractFieldStyle,
) = fs

Base.Broadcast.BroadcastStyle(
    ::FieldStyle{DS1},
    ::FieldStyle{DS2},
) where {DS1, DS2} = FieldStyle(Base.Broadcast.BroadcastStyle(DS1(), DS2()))

"""
    FieldConflict

Analogue of the built-in `Broadcast.ArrayConflict` for Fields. Used in place of
`Broadcast.Unknown` to call `Broadcast.broadcasted(::AbstractFieldStyle, ...)`.
Without this broadcast style, such `broadcasted` methods would need more complex
definitions that specialize on argument types, rather than just the style type.
"""
struct FieldConflict <: AbstractFieldStyle end

Base.Broadcast.result_join(
    ::AbstractFieldStyle,
    ::AbstractFieldStyle,
    ::Base.Broadcast.Unknown,
    ::Base.Broadcast.Unknown,
) = FieldConflict()

# Override the recursive unrolling used in combine_styles (which can lead to
# inference failures in broadcast expressions with more than 10 arguments) with
# manual unrolling (which can have higher latency but is always inferrable).
Base.Broadcast.combine_styles(
    arg1::Union{Field, Base.Broadcast.Broadcasted{<:AbstractFieldStyle}},
    arg2,
    arg3,
    args...,
) = unrolled_mapreduce(
    Base.Broadcast.combine_styles,
    Base.Broadcast.result_style,
    (arg1, arg2, arg3, args...),
)

# Define broadcastable, broadcasted, and eltype to match DataStyle broadcasting.
Base.Broadcast.broadcastable(field::Field) =
    Field(Base.Broadcast.broadcastable(field_values(field)), axes(field))
Base.Broadcast.broadcastable(
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
) =
    is_auto_broadcastable(eltype(bc)) ?
    Base.Broadcast.broadcasted(bc.style, enable_auto_broadcasting, bc) : bc

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, f::F, args...) where {F} =
    Base.Broadcast.Broadcasted(fs, f, unrolled_map(broadcast_arg, args))

Base.eltype(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    broadcast_eltype(bc)

# Define copy to match DataStyle broadcasting, but with the addition of a mask.
Base.copy(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    copyto!(similar(bc), bc, Spaces.get_mask(axes(bc)))

"""
    call_with_first(bc)

Tries to evaluate a broadcast expression at a single index, throwing an error
from the first function call whose result type cannot be inferred. Each call is
first checked for runtime errors, and then it is checked for type instabilities.
"""
function call_with_first(bc::Base.Broadcast.Broadcasted)
    args_at_single_index = unrolled_map(call_with_first, bc.args)
    result_at_single_index = bc.f(args_at_single_index...)
    arg_types = unrolled_map(typeof, args_at_single_index)
    inferred_type(bc.f, arg_types...) # throws an error if type inference fails
    return result_at_single_index
end
call_with_first(field::Field) =
    ClimaComms.allowscalar(ClimaComms.device(field)) do
        field_values(field)[CartesianIndex(1, 1, 1, 1, 1)]
    end
call_with_first(tup::Tuple) = tup[1]
call_with_first(ref::Ref) = ref[]
call_with_first(obj) = obj

# Define similar to match DataStyle broadcasting, but with extra error handling.
function Base.similar(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle})
    isconcretetype(eltype(bc)) || call_with_first(bc)
    return similar(bc, inferred_type(disable_auto_broadcasting, eltype(bc)))
end

Base.@propagate_inbounds function slab(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldSlabStyle(Style)
    _args = slab_args(bc.args, inds...)
    _axes = slab(axes(bc), inds...)
    Base.Broadcast.Broadcasted{_Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function slab(
    bc::DataLayouts.NonExtrudedBroadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldSlabStyle(Style)
    _args = slab_args(bc.args, inds...)
    _axes = slab(axes(bc), inds...)
    DataLayouts.NonExtrudedBroadcasted{_Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function level(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldLevelStyle(Style)
    _args = level_args(bc.args, inds...)
    _axes = level(axes(bc), inds...)
    Base.Broadcast.Broadcasted{_Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function level(
    bc::DataLayouts.NonExtrudedBroadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldLevelStyle(Style)
    _args = level_args(bc.args, inds...)
    _axes = level(axes(bc), inds...)
    DataLayouts.NonExtrudedBroadcasted{_Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function column(
    bc::Base.Broadcast.Broadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldColumnStyle(Style)
    _args = column_args(bc.args, inds...)
    _axes = column(axes(bc), inds...)
    Base.Broadcast.Broadcasted{_Style}(bc.f, _args, _axes)
end

Base.@propagate_inbounds function column(
    bc::DataLayouts.NonExtrudedBroadcasted{Style},
    inds...,
) where {Style <: AbstractFieldStyle}
    _Style = FieldColumnStyle(Style)
    _args = column_args(bc.args, inds...)
    _axes = column(axes(bc), inds...)
    DataLayouts.NonExtrudedBroadcasted{_Style}(bc.f, _args, _axes)
end

# Return underlying DataLayout object, DataStyle of broadcasted
# for `Base.similar` of a Field
# _todata_args(args::Tuple) = (todata(args[1]), _todata_args(Base.tail(args))...)
_todata_args(args::Tuple) =
    unrolled_map(args) do arg
        todata(arg)
    end

todata(obj) = obj
todata(field::Field) = Fields.field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    _args = _todata_args(bc.args)
    Base.Broadcast.Broadcasted{DS}(bc.f, _args)
end
function todata(bc::Base.Broadcast.Broadcasted{Style}) where {Style}
    _args = _todata_args(bc.args)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args)
end
function todata(
    bc::DataLayouts.NonExtrudedBroadcasted{FieldStyle{DS}},
) where {DS}
    _args = _todata_args(bc.args)
    DataLayouts.NonExtrudedBroadcasted{DS}(bc.f, _args)
end
function todata(bc::DataLayouts.NonExtrudedBroadcasted{Style}) where {Style}
    _args = _todata_args(bc.args)
    DataLayouts.NonExtrudedBroadcasted{Style}(bc.f, _args)
end

field_values(bc::Base.AbstractBroadcasted) = todata(bc)

# same logic as Base.Broadcast.Broadcasted (which only defines it for Tuples)
Base.axes(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    _axes(bc, bc.axes)
_axes(bc, ::Nothing) = Base.Broadcast.combine_axes(bc.args...)
_axes(bc, axes) = axes

Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
    ::Type{Eltype},
) where {Eltype} = Field(Eltype, axes(bc))

Base.similar(
    bc::Base.Broadcast.Broadcasted{<:FieldStyle},
    ::Type{Eltype},
) where {Eltype} = Field(similar(todata(bc), Eltype), axes(bc))

@inline function Base.copyto!(
    dest::Field,
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
    mask = get_mask(axes(dest)),
)
    copyto!(field_values(dest), Base.Broadcast.instantiate(todata(bc)), mask)
    return dest
end

# Fused multi-broadcast entry point for Fields
function Base.copyto!(
    fmbc::FusedMultiBroadcast{T},
) where {N, T <: NTuple{N, Pair{<:Field, <:Any}}}
    fmb_data = FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            bc = Base.Broadcast.instantiate(todata(pair.second))
            Pair(field_values(pair.first), bc)
        end,
    )
    check_mismatched_spaces(fmbc)
    check_fused_broadcast_axes(fmbc)
    Base.copyto!(fmb_data) # forward to DataLayouts
end

@inline check_mismatched_spaces(fmbc::FusedMultiBroadcast) =
    check_mismatched_spaces(
        map(x -> axes(x.first), fmbc.pairs),
        axes(first(fmbc.pairs).first),
    )
@inline check_mismatched_spaces(axs::Tuple{<:Any}, ax1) =
    _check_mismatched_spaces(first(axs), ax1)
@inline check_mismatched_spaces(axs::Tuple{}, ax1) = nothing
@inline function check_mismatched_spaces(axs::Tuple, ax1)
    _check_mismatched_spaces(first(axs), ax1)
    check_mismatched_spaces(Base.tail(axs), ax1)
end

_check_mismatched_spaces(::T, ::T) where {T <: AbstractSpace} = nothing
_check_mismatched_spaces(space1, space2) =
    error("FusedMultiBroadcast spaces are not the same.")

@noinline function error_mismatched_spaces(space1::Type, space2::Type)
    error("Broacasted spaces are not the same.")
end

@inline function Base.Broadcast.broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    if space1 !== space2 && !allow_mismatched_spaces_unsafe()
        if Spaces.issubspace(space2, space1)
            return space1
        elseif Spaces.issubspace(space1, space2)
            return space2
        else
            error_mismatched_spaces(typeof(space1), typeof(space2))
        end
    end
    return space1
end
@inline Base.Broadcast.broadcast_shape(space::AbstractSpace, ::Tuple) = space
@inline Base.Broadcast.broadcast_shape(::Tuple, space::AbstractSpace) = space

@inline Base.Broadcast.broadcast_shape(
    pointspace::AbstractPointSpace,
    space::AbstractSpace,
) = space
@inline Base.Broadcast.broadcast_shape(
    space::AbstractSpace,
    pointspace::AbstractPointSpace,
) = space

# Avoid method ambiguity:
@inline Base.Broadcast.broadcast_shape(
    a::AbstractPointSpace,
    b::AbstractPointSpace,
) = a

# Overload broadcast axes shape checking for more useful error message for Field Spaces
@inline function Base.Broadcast.check_broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    # When DebugOnly.allow_mismatched_spaces_unsafe() returns true, we ignore the check
    # and blindly return `space1`. Responsibility is left up to the user to
    # handle this correctly. This is useful to work with spaces that are == but
    # not ===, e.g., deepcopied spaces.
    if space1 !== space2 && !allow_mismatched_spaces_unsafe()
        if Spaces.issubspace(space2, space1) ||
           Spaces.issubspace(space1, space2)
            nothing
        else
            error_mismatched_spaces(typeof(space1), typeof(space2))
        end
    end
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    space::AbstractSpace,
    ax2::Tuple,
)
    error_mismatched_spaces(typeof(space), typeof(ax2))
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::Tuple{},
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::Tuple{T},
) where {T}
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::AbstractPointSpace,
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractPointSpace,
    ::AbstractSpace,
)
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractPointSpace,
    ::AbstractPointSpace,
)
    return nothing
end

# By default, broadcasted Vals are put in Refs, leading to type instabilities
Base.Broadcast.broadcasted(
    ::typeof(Base.literal_pow),
    ::typeof(^),
    f::Union{Field, Base.Broadcast.Broadcasted{<:AbstractFieldStyle}},
    ::Val{n},
) where {n} = Base.Broadcast.broadcasted(x -> Base.literal_pow(^, x, Val(n)), f)

# Specialize handling of vector-based functions to automatically add LocalGeometry information
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm,
        arg,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm_sqr),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm_sqr,
        arg,
        local_geometry_field(space),
    )
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.cross),
    arg1,
    arg2,
)
    space = Fields.axes(arg1)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._cross,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(Geometry.transform),
    arg1,
    arg2,
)
    space = Fields.axes(arg2)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry.transform,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(Geometry.project),
    arg1,
    arg2,
)
    space = Fields.axes(arg2)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry.project,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.AxisVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(fs, V, arg, local_geometry_field(space))
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.CartesianVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        V,
        arg,
        tuple(Spaces.global_geometry(space)),
        local_geometry_field(space),
    )
end

function Base.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
)
    mask = get_mask(axes(field))
    copyto!(Fields.field_values(field), todata(bc), mask)
    return field
end
function Base.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}},
)
    mask = get_mask(axes(field))
    copyto!(Fields.field_values(field), todata(bc), mask)
    return field
end

function Base.copyto!(field::Field, nt::NamedTuple)
    mask = get_mask(axes(field))
    copyto!(
        field,
        Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}(
            identity,
            (nt,),
            axes(field),
        ),
        mask,
    )
end
