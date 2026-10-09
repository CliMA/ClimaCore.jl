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

# Base's combine_axes checks pairwise combinations of arguments, incorrectly
# flagging expressions like field .+ subfield_1 .+ subfield_2 as broadcast
# errors when subfield_1 and subfield_2 are defined on different spaces, so it
# must be replaced with a method that recursively combines all broadcast inputs.
# An instantiated broadcast has already been checked, so its stored axes are
# used in place of its inputs (whose spaces may not be comparable after slicing,
# e.g., when a level field is sliced along with an extruded field).
@inline shared_space(_) = nothing
@inline shared_space(field::Field) = axes(field)
@inline shared_space(bc::LazyField) =
    !isnothing(bc.axes) ? bc.axes :
    unrolled_reduce(bc.args; init = nothing) do space, arg
        !isnothing(shared_space(arg)) &&
        (isnothing(space) || Spaces.maybe_issubspace(space, shared_space(arg))) ?
        shared_space(arg) : space
    end

# A similar recursive method takes the place of Base's check_broadcast_shape,
# with an only_check_size flag used when checking broadcast destination spaces.
@inline check_broadcast_space(_, _, _) = nothing
@inline check_broadcast_space(space, field::Field, ::Val{true}) =
    Base.Broadcast.check_broadcast_axes(
        axes(Fields.local_geometry_field(space)),
        Fields.field_values(field),
    )
@inline check_broadcast_space(space, field::Field, ::Val{false}) =
    check_subspace(space, axes(field))
@inline check_broadcast_space(space, bc::LazyField, only_check_size) =
    only_check_size == Val(false) && !isnothing(bc.axes) ?
    check_subspace(space, bc.axes) :
    unrolled_foreach(arg -> check_broadcast_space(space, arg, only_check_size), bc.args)
@inline check_subspace(space, subspace) =
    !isnothing(space) && Spaces.issubspace(subspace, space) ? nothing :
    throw(DimensionMismatch("Fields could not be broadcast to a shared space"))

@drop_recursion_limits shared_space, check_broadcast_space

# Modify axes and check_broadcast_axes to support AbstractSpaces, and make
# instantiate call axes instead of Base's combine_axes.
@inline Base.Broadcast._axes(::Base.Broadcast.Broadcasted, space::AbstractSpace) = space
@inline function Base.Broadcast._axes(bc::LazyField, ::Nothing)
    check_broadcast_space(shared_space(bc), bc, Val(false))
    return shared_space(bc)
end

@inline Base.Broadcast.check_broadcast_axes(space::AbstractSpace, arg, args...) =
    unrolled_foreach(arg -> check_broadcast_space(space, arg, Val(true)), (arg, args...))

@inline function Base.Broadcast.instantiate(bc::LazyField)
    isnothing(bc.axes) || Base.Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    return Base.Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(bc))
end

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

Base.copy(bc::LazyField) = copyto!(similar(bc), bc, Spaces.get_mask(axes(bc)))

field_values(bc::Broadcast.Broadcasted) = bc
@inline field_values(bc::LazyField{FieldStyle{DS}}) where {DS} =
    Broadcast.Broadcasted{DS}(
        bc.f,
        unrolled_map(arg -> arg isa MaybeLazyField ? field_values(arg) : arg, bc.args),
    )

# Forward size primitives from Base and DataLayouts to the field_values.
for f in (:size, :length, :ndims)
    @eval Base.$f(arg::MaybeLazyField) = $f(field_values(arg))
end
for f in (:shape_params, :inferred_size, :nelems)
    @eval DataLayouts.$f(arg::MaybeLazyField) = DataLayouts.$f(field_values(arg))
end

@inline has_field_style(::T) where {T} =
    Base.Broadcast.BroadcastStyle(T) isa Fields.AbstractFieldStyle

@inline DataLayouts.DataScope(bc::LazyField) =
    DataLayouts.DataScope(unrolled_filter(has_field_style, bc.args)...)
@inline DataLayouts.reassign(bc::LazyField, scope) = Broadcast.Broadcasted(
    bc.style,
    bc.f,
    unrolled_map(
        arg -> arg isa MaybeLazyField ? DataLayouts.reassign(arg, scope) : arg,
        bc.args,
    ),
    bc.axes,
)

# Duplicate grid pointers in Fields and LazyFields are toggled off before
# launching a kernel, then toggled back on when the kernel starts executing.
# Note that this could be optimized further by dropping grid pointers from arg1.
Grids.toggle_compact_args(arg1::MaybeLazyField, args...) =
    (arg1, unrolled_map(arg -> Grids.toggle_placeholder_grid(arg, axes(arg1)), args)...)

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
sliced_broadcast_body(op, bc_type) = quote
    Base.@_propagate_inbounds_meta
    f = getfield(bc, :f)
    f′ = f isa Union{Function, Type} ? f : $op(f, inds...)
    args = getfield(bc, :args)
    return sliced_broadcasted(
        f′,
        Base.Cartesian.@ntuple(
            $(length(bc_type.parameters[4].parameters)),
            n -> let arg = getfield(args, n)
                arg isa MaybeLazyField ? $op(arg, arg_slice_indices($op, arg, inds)...) :
                arg
            end,
        ),
        $op(bc.axes, inds...),
    )
end

# Like Broadcast.newindex, project slice indices onto the dimensions that are
# missing from each argument's space, so that, e.g., a level field is sliced
# along with an extruded field. Nested broadcasts without stored axes are not
# projected, since their arguments get projected when they are sliced, and
# computing their axes would add a significant compilation cost.
@inline function arg_slice_indices(op::O, arg, inds) where {O}
    space = arg isa Field ? axes(arg) : getfield(arg, :axes)
    (op == Base.view || isnothing(space)) && return inds
    v = Spaces.has_vertical(space) ? inds[1] : 1
    has_h = Spaces.has_horizontal(space)
    op == level && return (v,)
    op == column && return has_h ? inds : ntuple(Returns(1), Val(length(inds)))
    return length(inds) == 1 ? (has_h ? inds[1] : 1,) : (v, has_h ? inds[2] : 1)
end

for op in (:(Base.view), :level, :slab, :column)
    @eval @generated $op(bc::LazyField, inds...) =
        sliced_broadcast_body($(QuoteNode(op)), bc)
end

# Extend the DataLayout methods of IndexStyle and eachindex to Field broadcasts.
Base.IndexStyle(bc::LazyField) = IndexStyle(field_values(bc))
Base.eachindex(arg::MaybeLazyField, args::MaybeLazyField...) =
    eachindex(field_values(arg), unrolled_map(field_values, args)...)

Base.similar(bc::LazyField, ::Type{T}) where {T} = Field(T, axes(bc))

# Allocate pointwise broadcast results from the broadcast's own data instead of
# going through the space, whose coordinate data can be a dynamically-sized view
# even when the broadcast's layout shape is static.
Base.similar(bc::LazyField{FieldStyle{DS}}, ::Type{T}) where {DS, T} =
    Field(similar(field_values(bc), T), axes(bc))

# The mask is an optional positional argument, as for DataLayouts (see the
# copyto! methods in DataLayouts/loops.jl), and every copyto! method for a
# LazyField style accepts it in the same position.
@inline function Base.copyto!(
    dest::Field,
    bc::LazyField,
    mask::DataLayouts.DataMask = get_mask(axes(dest)),
)
    copyto!(field_values(dest), Base.Broadcast.instantiate(field_values(bc)), mask)
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
    copyto!(fmb_data; mask)
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
    copyto!(field_values(field), bc, get_mask(axes(field)))
    return field
end
function Base.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.Style{Tuple}},
)
    copyto!(field_values(field), bc, get_mask(axes(field)))
    return field
end

function Base.copyto!(field::Field, nt::NamedTuple)
    mask = get_mask(axes(field))
    fill!(field_values(field), nt; mask)
    return field
end
