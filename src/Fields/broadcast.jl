import ..DebugOnly: allow_mismatched_spaces_unsafe
import ..Utilities.Unrolled: unrolled_tuple_map

"""
    AbstractFieldStyle

Abstract supertype of the broadcast styles of `Field`s. Subtypes: `FieldStyle` and
`FieldConflict`.
"""
abstract type AbstractFieldStyle <: Base.BroadcastStyle end

const LazyField{S <: AbstractFieldStyle} = Base.Broadcast.Broadcasted{S}
const MaybeLazyField = Union{Field, LazyField}

"""
    FieldStyle{DS <: DataStyle}

Broadcast style of `Field`s whose values have the `DataStyle` `DS`, to which the
work is delegated.
"""
struct FieldStyle{DS <: DataStyle} <: AbstractFieldStyle end

FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()
FieldStyle(x::Base.Broadcast.Unknown) = x

Base.Broadcast.BroadcastStyle(::Type{Field{V, S}}) where {V, S} =
    FieldStyle(Base.Broadcast.BroadcastStyle(V))

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

Analog of the built-in `Broadcast.ArrayConflict` for `Field`s. Used in place of
`Broadcast.Unknown` to call `Broadcast.broadcasted(::AbstractFieldStyle, ...)`.
Without this broadcast style, such `broadcasted` methods would need definitions
that specialize on argument types rather than on the style type alone.
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
Base.Broadcast.combine_styles(arg1::MaybeLazyField, arg2, arg3, args...) =
    unrolled_mapreduce(
        Base.Broadcast.combine_styles,
        Base.Broadcast.result_style,
        (arg1, arg2, arg3, args...),
    )

# Base's _axes only supports Tuple values of bc.axes, so broadcasts whose axes
# are spaces need this additional method.
Base.Broadcast._axes(::Base.Broadcast.Broadcasted, space::AbstractSpace) = space

# Define broadcastable/broadcasted/newindex/eltype/similar/copy to match
# DataStyle broadcasting (see broadcast.jl in the DataLayouts module).
Base.Broadcast.broadcastable(field::Field) =
    Field(Base.Broadcast.broadcastable(field_values(field)), axes(field))
Base.Broadcast.broadcastable(bc::LazyField) =
    is_auto_broadcastable(eltype(bc)) ?
    Base.Broadcast.Broadcasted(bc.style, add_auto_broadcasters, (bc,)) : bc

Base.Broadcast.broadcasted(style::AbstractFieldStyle, f::F, args...) where {F} =
    auto_broadcasted(style, f, args)

Base.Broadcast.newindex(arg::MaybeLazyField, index::Integer) =
    iszero(ndims(arg)) ? CartesianIndex() : index

Base.eltype(bc::LazyField) = unsafe_eltype(bc)

Base.similar(bc::LazyField) = similar(bc, drop_auto_broadcasters(safe_eltype(bc)))

Base.copy(bc::LazyField) =
    copyto!(similar(bc), bc; mask = Spaces.get_mask(axes(bc)))

field_values(bc::Broadcast.Broadcasted) = bc
@inline field_values(bc::LazyField{FieldStyle{DS}}) where {DS} =
    Broadcast.Broadcasted{DS}(
        bc.f,
        unrolled_tuple_map(
            arg -> arg isa MaybeLazyField ? field_values(arg) : arg,
            bc.args,
        ),
    )

# Forward size/scope primitives from Base and DataLayouts to the field_values.
for f in (:size, :length, :ndims)
    @eval Base.$f(arg::MaybeLazyField) = $f(field_values(arg))
end
for f in (:DataScope, :shape_params, :inferred_size, :nelems)
    @eval DataLayouts.$f(arg::MaybeLazyField) = DataLayouts.$f(field_values(arg))
end

@inline DataLayouts.reassign(bc::LazyField, scope) = Broadcast.Broadcasted(
    bc.style,
    bc.f,
    unrolled_tuple_map(
        arg -> arg isa MaybeLazyField ? DataLayouts.reassign(arg, scope) : arg,
        bc.args,
    ),
    bc.axes,
)

# Analogue of Broadcast.broadcasted for rebuilding the nodes of an existing
# broadcast expression from slices of their arguments, skipping the
# Utilities.auto_broadcasted analysis the expression already went through.
# Re-running it is redundant and, starting with Julia 1.11, harmful: GPU kernel
# inference stops constant-folding unsafe_eltype partway down deeply nested
# operator expressions, turning the slice operators and return_eltype into
# dynamic calls with runtime allocations in GPU kernels.
@inline sliced_broadcasted(f::F, args, axes) where {F} =
    Broadcast.Broadcasted(Broadcast.combine_styles(args...), f, args, axes)

# Body of a slice operator applied to one node of a broadcast expression; a
# generated function makes slicing a node one method instance rather than three
# per node per distinct expression type (as in DataLayouts/indexing.jl).
sliced_broadcast_body(op::Symbol, bc_type) = quote
    Base.@_propagate_inbounds_meta
    f = getfield(bc, :f)
    f′ = f isa Union{Function, Type} ? f : $op(f, inds...)
    args = getfield(bc, :args)
    return sliced_broadcasted(
        f′,
        Base.Cartesian.@ntuple(
            $(length(bc_type.parameters[4].parameters)),
            n -> let arg = getfield(args, n)
                arg isa MaybeLazyField ? $op(arg, inds...) : arg
            end,
        ),
        $op(bc.axes, inds...),
    )
end

for op in (:level, :slab, :column)
    @eval @generated $op(bc::LazyField, inds...) =
        sliced_broadcast_body($(QuoteNode(op)), bc)
end

# Extend the DataLayout methods of IndexStyle and eachindex to Field broadcasts.
Base.IndexStyle(bc::LazyField) = IndexStyle(field_values(bc))
Base.eachindex(arg::MaybeLazyField, args::MaybeLazyField...) =
    eachindex(field_values(arg), unrolled_tuple_map(field_values, args)...)

Base.similar(bc::LazyField, ::Type{T}) where {T} = Field(T, axes(bc))

# Allocate pointwise broadcast results from the broadcast's own data instead of
# going through the space, whose coordinate data can be a dynamically-sized view
# even when the broadcast's layout shape is static.
Base.similar(bc::LazyField{FieldStyle{DS}}, ::Type{T}) where {DS, T} =
    Field(similar(field_values(bc), T), axes(bc))

@inline function Base.copyto!(dest::Field, bc::LazyField; mask = get_mask(axes(dest)))
    copyto!(field_values(dest), Base.Broadcast.instantiate(field_values(bc)); mask)
    return dest
end

# Fused multi-broadcast entry point for Fields. The mask argument must be
# constrained to DataMask because an unconstrained second argument makes this
# ambiguous with copyto! methods that only constrain their second arguments,
# like the ones for Lmul and Rmul in ArrayLayouts.
function Base.copyto!(
    fmbc::FusedMultiBroadcast{T};
    mask::DataLayouts.DataMask = get_mask(axes(first(fmbc.pairs).first)),
) where {N, T <: NTuple{N, Pair{<:Field, <:Any}}}
    fmb_data = FusedMultiBroadcast(
        map(fmbc.pairs) do pair
            bc = Base.Broadcast.instantiate(field_values(pair.second))
            Pair(field_values(pair.first), bc)
        end,
    )
    check_mismatched_spaces(fmbc)
    copyto!(fmb_data; mask)
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

error_mismatched_spaces() = error("Broacasted spaces are not the same.")

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
            error_mismatched_spaces()
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
    # When DebugOnly.allow_mismatched_spaces_unsafe() returns true, the check is skipped
    # and `space1` is returned. The caller is responsible for the spaces being
    # compatible. This allows working with spaces that are == but not ===, e.g.,
    # deepcopied spaces.
    if space1 !== space2 && !allow_mismatched_spaces_unsafe()
        if Spaces.issubspace(space2, space1) ||
           Spaces.issubspace(space1, space2)
            nothing
        else
            error_mismatched_spaces()
        end
    end
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    space::AbstractSpace,
    ax2::Tuple,
)
    error_mismatched_spaces()
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
    f::MaybeLazyField,
    ::Val{n},
) where {n} = Base.Broadcast.broadcasted(x -> Base.literal_pow(^, x, Val(n)), f)

# Specialize vector-based functions to add LocalGeometry information.
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm),
    arg,
)
    space = axes(arg)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
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
    space = axes(arg)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
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
    space = axes(arg1)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
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
    space = axes(arg2)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
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
    space = axes(arg2)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
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
) where {V <: Geometry.AbstractTensor{1}}
    space = axes(arg)
    # Wrap in a Field so that the axes line up (the Field is unwrapped again, so this is
    # a no-op).
    Base.Broadcast.broadcasted(fs, V, arg, local_geometry_field(space))
end

function Base.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
)
    mask = get_mask(axes(field))
    copyto!(field_values(field), bc; mask)
    return field
end
function Base.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}},
)
    mask = get_mask(axes(field))
    copyto!(field_values(field), bc; mask)
    return field
end

function Base.copyto!(field::Field, nt::NamedTuple)
    mask = get_mask(axes(field))
    fill!(field_values(field), nt; mask)
    return field
end
