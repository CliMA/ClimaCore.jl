"""
    OperatorStyle(op)

Abstract supertype of the broadcast styles used by expressions with at least
one [`AbstractOperator`](@ref). Every operator must provide a constructor for
its associated broadcast style.
"""
abstract type OperatorStyle <: Fields.AbstractFieldStyle end

"""
    slice_operator(style)

One of [`level`](@ref), [`slab`](@ref), or [`column`](@ref), which is passed to
[`DataLayouts.foreach_slice`](@ref) when evaluating a broadcast expression with
the specified [`OperatorStyle`](@ref).
"""
function slice_operator end

# Operator styles take precedence over pointwise broadcast styles.
Broadcast.BroadcastStyle(style::OperatorStyle, ::Fields.FieldStyle) = style

# Combine the operator style with all argument styles to construct broadcasts.
function Broadcast.broadcasted(op::AbstractOperator, args...)
    args′ = unrolled_map(Broadcast.broadcastable, args)
    style = Broadcast.result_style(OperatorStyle(op), Broadcast.combine_styles(args′...))
    return Broadcast.broadcasted(style, op, args′...)
end
@inline function Fields.sliced_broadcasted(op::AbstractOperator, args, axes)
    style = Broadcast.result_style(OperatorStyle(op), Broadcast.combine_styles(args...))
    return Broadcast.Broadcasted(style, op, args, axes)
end

function Broadcast.broadcasted(style::OperatorStyle, op::AbstractOperator, args...)
    FT = Spaces.undertype(return_space(op, args...))
    args′ = unrolled_map(args) do arg
        is_auto_broadcastable(eltype(arg)) ?
        Broadcast.broadcasted(add_auto_broadcasters, arg) : arg
    end
    return Broadcast.Broadcasted(style, promote_bcs(op, FT), args′)
end

const NonPointwiseBroadcasted{F} = Broadcast.Broadcasted{<:OperatorStyle, <:Any, F}
const OperatorBroadcasted{Op <: AbstractOperator} = NonPointwiseBroadcasted{Op}

Fields.sliced_function(slice, op::AbstractOperator, inds) = slice(op, inds...)

# Base implements .&& and .|| by flattening their second operands into single
# functions of their leaves (see Fields.ShortCircuit), which would call any
# operators in them pointwise, so an operator broadcast is kept as an operand
# that is evaluated at every point. First operands are always kept as operands.
Broadcast.broadcasted(::Broadcast.AndAnd, a, bc::NonPointwiseBroadcasted) =
    Broadcast.broadcasted(logical_and, a, bc)
Broadcast.broadcasted(::Broadcast.OrOr, a, bc::NonPointwiseBroadcasted) =
    Broadcast.broadcasted(logical_or, a, bc)
logical_and(a, b) = a && b
logical_or(a, b) = a || b

Utilities.unsafe_eltype(bc::OperatorBroadcasted) = return_eltype(bc.f, bc.args...)

# Allocate the results of operator broadcasts through their own scopes (like the
# results of pointwise broadcasts), not through their spaces, since a slice of a
# fused slice loop has a narrower scope than its space (e.g., one thread on a
# CPU, or part of a thread block on a GPU, where only that scope can allocate).
Base.similar(bc::NonPointwiseBroadcasted, ::Type{T}) where {T} =
    register_similar(point_template(bc), T)
@inline Fields.shared_space(bc::OperatorBroadcasted) = return_space(bc.f, bc.args...)

# The arguments of an operator can have spaces that differ from the result space
# (e.g., an F2C operator reads from a face space and writes to a center space),
# so they cannot be used to compute axes or verified with check_broadcast_space.
@inline Base.Broadcast._axes(bc::OperatorBroadcasted, ::Nothing) = Fields.shared_space(bc)
@inline Fields.check_broadcast_space(space, bc::OperatorBroadcasted, only_check_size) =
    Fields.check_broadcast_space(
        space,
        Fields.local_geometry_field(axes(bc)),
        only_check_size,
    )
# Checking that the result space is a subspace only needs that space. Building the
# local geometry of a column space slices the local geometry data with a bounds
# check, which would stay in every kernel that computes the axes of a broadcast
# over an operator (e.g., to cache an argument), along with the array sizes it
# reads.
@inline Fields.check_broadcast_space(space, bc::OperatorBroadcasted, ::Val{false}) =
    Fields.check_subspace(space, axes(bc))

# Since check_broadcast_space does not recursively check operator arguments,
# instantiate must be applied recursively to verify nested argument spaces.
@inline function Broadcast.instantiate(bc::NonPointwiseBroadcasted)
    (isnothing(bc.axes) || bc isa OperatorBroadcasted) ||
        Base.Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    instantiated_args = unrolled_map(Broadcast.instantiate, bc.args)
    return Broadcast.Broadcasted(bc.style, bc.f, instantiated_args, axes(bc))
end

# Upper bound on the per-thread shared memory required to inline all operator
# applications in a broadcast expression, with every application charged two
# buffers of its largest element type. Assume that each slice's scope has no
# fewer threads than points, which is guaranteed by DataLayouts.foreach_slice.
inlined_buffer_bytes(arg) = 0
inlined_buffer_bytes(bc::NonPointwiseBroadcasted) =
    unrolled_sum(inlined_buffer_bytes, bc.args)
inlined_buffer_bytes(bc::OperatorBroadcasted) =
    2 * unrolled_maximum(sizeof ∘ eltype, (bc, bc.args...)) +
    unrolled_sum(inlined_buffer_bytes, bc.args)

# Inline each operator unless its expression asks a block for more than CUDA's
# 48 KB of static shared memory (a compilation error). The budget is 192 bytes
# per thread of a 256-thread block; launched blocks hold at most 128 threads
# (MAX_SUBBLOCK_LAUNCH_THREADS in ext/cuda/scopes.jl), so a single expression
# reserves no more than half of the 48 KB. This allows two buffers to be live at
# the same time, but three or more can still overflow and cause a ptxas error.
const MAX_INLINED_BUFFER_BYTES = 48 * 1024 ÷ 256

# Recursively replace all operator broadcasts with the result of apply_operator.
# The evaluation of pointwise functions of operators can depend on the style.
apply_operators(arg) = arg
apply_operators(bc::NonPointwiseBroadcasted) = apply_pointwise_operators(bc.style, bc)
apply_pointwise_operators(_, bc) =
    Broadcast.broadcasted(bc.f, unrolled_map(apply_operators, bc.args)...)
apply_operators(bc::OperatorBroadcasted) =
    inlined_buffer_bytes(bc) <= MAX_INLINED_BUFFER_BYTES ?
    inline_apply_operators(bc) : noinline_apply_operators(bc)

@inline inline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_map(operator_arg, bc.args)...)
@noinline noinline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_map(operator_arg, bc.args)...)

# Like apply_operators, but for an argument of an operator, which may keep some
# operator broadcasts lazy and evaluate them wherever it reads their results.
operator_arg(arg) = apply_operators(arg)

# Copy one slice of an operator-free expression into one slice of the
# destination, whose space can differ from the expression's (e.g., when copying
# between levels). Broadcast expressions skip .= to avoid inferring the point
# loop into both of its materialize! layers; other arguments are rare enough to
# keep .=.
@inline copyto_slice!(dest, bc::LazyField) =
    copyto!(Fields.field_values(dest), Broadcast.instantiate(Fields.field_values(bc)))
@inline copyto_slice!(dest, arg) = (dest .= arg)

# Like apply_operators, but for an entire expression whose result is copied into
# dest. Styles can override this to write the result directly into dest.
apply_operators!(dest, bc) = copyto_slice!(dest, apply_operators(bc))

# Masks can only skip some types of slices (e.g., a slab is live whenever any of
# its columns is active), so they are ignored when they cannot be applied. The
# slice loop is entered through its positional form, since the keyword form adds
# two method instances to every operator broadcast, each inferred with the whole
# loop inlined into it.
function Base.copyto!(dest::Fields.Field, bc::NonPointwiseBroadcasted)
    op = slice_operator(bc.style)
    space_mask = Spaces.get_mask(axes(dest))
    mask =
        DataLayouts.is_valid_slice_mask(space_mask, op) ? space_mask : DataLayouts.NoMask()
    # The entry body of DataLayouts._foreach_slice is replicated here rather
    # than called: every layer around a slice loop is a method instance that is
    # inferred and optimized with the whole loop inlined into it.
    unrolled_allequal(Base.Fix1(DataLayouts.each_slice_index, op), (dest, bc)) ||
        throw(DimensionMismatch("Inputs to foreach_slice must have compatible dimensions"))
    scope = DataLayouts.DataScope(dest, bc)
    if DataLayouts.needs_loop_setup(scope)
        DataLayouts._foreach_slice(scope, op, apply_operators!, mask, Val(false), dest, bc)
    else
        DataLayouts.scoped_slice_loop(
            DataLayouts.slice_subscope(scope, op, dest, bc),
            scope,
            op,
            apply_operators!,
            mask,
            Val(false),
            dest,
            bc,
        )
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc)
    return dest
end

"""
    has_private_buffers(arg)

Return whether a buffer allocated for `arg` is private to the allocating thread,
which holds exactly when `arg`'s [`DataLayouts.DataScope`](@ref) is
[`DataLayouts.ThisThread`](@ref) (as on CPUs); otherwise buffers live in shared
memory and must obey the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline has_private_buffers(arg) = DataLayouts.DataScope(arg) == DataLayouts.ThisThread()

"""
    register_similar(arg, T)

Allocate a `Field` like `Base.similar(arg, T)`, but with its data in each
thread's registers; used for every [`apply_operator`](@ref) destination, which only its
own thread reads and writes. This lets two applications in one fused expression
be live at once (see the buffer reuse invariant in [`apply_operator`](@ref)).
"""
register_similar(arg, ::Type{T}) where {T} =
    Field(DataLayouts.register_similar(Fields.field_values(arg), T), axes(arg))

"""
    buffer_similar(arg, [T])

Allocate a `Field` like `Base.similar(arg, T)`, but always through the argument's
[`DataLayouts.DataScope`](@ref) (shared memory on GPUs), never in per-thread
registers; used for every buffer whose values cross a thread boundary. By
default, `T` is the element type of `arg`, including any `AutoBroadcaster`
wrappers.
"""
@inline buffer_similar(arg, ::Type{T}) where {T} =
    Field(DataLayouts.buffer_similar(Fields.field_values(arg), T), axes(arg))
@inline buffer_similar(arg) = buffer_similar(arg, eltype(arg))

"""
    constant_field(arg)

When `arg` is a `Field` with thread-local data (i.e., data that is either owned
by a single thread or stored in registers), copy its data into an immutable
`StaticArrays.SArray`], which is always stack-allocated. The default
`StaticArrays.MArray` used to store thread-local data is heap-allocated unless
every read and write is inlined, but such excessive inlining generally blows up
compilation time. GPUs only have stack memory, so this enables GPU compilation
without full inlining.
"""
@inline constant_field(arg) =
    has_private_buffers(arg) || DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    Field(DataLayouts.rebuild(Fields.field_values(arg), StaticArrays.SArray), axes(arg)) :
    arg

"""
    materialize_buffer(arg)

Materialize `arg` like `Base.materialize(arg)`, but store the result of a lazy
`Broadcasted` expression in a buffer from [`buffer_similar`](@ref) that every
thread in the argument's scope can read; register-resident `Field`s (see
[`register_similar`](@ref)) are also copied into such a buffer, and other
arguments are returned unchanged. On GPUs two
buffers of equal byte size can share memory, so callers must keep their
lifetimes disjoint; see the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline materialize_buffer(arg) = arg
@inline materialize_buffer(arg::MaybeLazyField) =
    arg isa LazyField || DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    constant_field(copyto!(buffer_similar(arg), arg)) : arg

"""
    maybe_private_buffer(arg)

Like [`materialize_buffer`](@ref), but only applied to arguments that are never
read across a thread boundary (avoiding the latency of a shared memory buffer).
"""
@inline maybe_private_buffer(arg) = has_private_buffers(arg) ? materialize_buffer(arg) : arg

"""
    own_point_values(arg)

Like [`maybe_private_buffer`](@ref), for an argument that every thread reads
only at its own points, but once for each horizontal dimension of an operator:
when a device scope has registers (see [`register_values`](@ref)), the argument
is evaluated there once, instead of being re-evaluated (and its inputs re-read
from memory) for every dimension. An argument that is read at other threads'
points (like the weighted argument of `Restrict`) must not use this.
"""
@inline own_point_values(arg) =
    has_private_buffers(arg) ? materialize_buffer(arg) :
    arg isa MaybeLazyField &&
    DataLayouts.has_scope_registers(Fields.field_values(point_template(arg))) ?
    register_values(arg) : arg

"""
    fused_buffer(arg)

Like [`cached_arg`](@ref) for an argument that is read in lockstep, but leaving
`arg` lazy when it is a shallow `Broadcasted` over materialized values and a
single thread owns the data (larger scopes have to publish the values to every
thread).
"""
@inline fused_buffer(arg) = arg
@inline fused_buffer(bc::LazyField) =
    has_private_buffers(bc) && unrolled_all(arg -> !(arg isa LazyField), bc.args) ?
    bc : cached_arg(bc, Val(true))

"""
    cached_arg(arg, lockstep)

Prepare an argument that an operator reads at points other than the ones it
evaluates, so that every value of the argument is computed only once (by the
thread that owns its point) and can be read by every thread in its scope. A lazy
expression, or a `Field` in registers (see [`register_similar`](@ref)), is
evaluated into a buffer from [`materialize_buffer`](@ref), whose lifetime must
obey the buffer reuse invariant in [`apply_operator`](@ref); other arguments,
like constants and `Field`s that every thread can read, are returned unchanged.

When `lockstep` is `Val(true)`, the operator reads the argument with every
thread of its scope at the same time (see [`DataLayouts.update_points!`](@ref)),
which lets a device keep the values in the registers of the threads that compute
them, instead of in a buffer (see [`cached_values`](@ref)).

The cached values keep their `AutoBroadcaster` wrappers (unlike the values of a
materialized `Field`), so the cached argument has the same element type as `arg`.
"""
@inline cached_arg(arg, lockstep) =
    is_cached(arg) ? cached_values(DataLayouts.DataScope(arg), arg, lockstep) : arg

"""
    cached_values(scope, arg, lockstep)

Evaluate an argument for [`cached_arg`](@ref) in the given
[`DataLayouts.DataScope`](@ref), by default into a buffer from
[`materialize_buffer`](@ref) that every thread in the scope can read. Devices
whose threads can read each other's registers specialize this for their scopes,
so that an argument read in lockstep can stay in the registers of the threads
that compute it (see [`register_values`](@ref)), where reading it can require
lockstep (see [`requires_lockstep`](@ref)).
"""
@inline cached_values(_, arg, _) = materialize_buffer(arg)

# Whether cached_arg evaluates its argument into a buffer or registers.
is_cached(_) = false
is_cached(field::Field) = DataLayouts.stored_in_registers(Fields.field_values(field))
is_cached(::LazyField) = true

# Field with the data shape of arg's slice of its space and arg's scope, used as
# a template for allocating values at the points of arg.
@inline point_template(arg) = DataLayouts.reassign(
    Fields.local_geometry_field(axes(arg)),
    DataLayouts.DataScope(arg),
)

"""
    register_values(arg)

Evaluate `arg` into the registers of the threads that own its points (see
[`register_similar`](@ref)), unless its values are already there.
"""
@inline register_values(field::Field) =
    DataLayouts.stored_in_scope_registers(Fields.field_values(field)) ? field :
    register_values(Base.broadcasted(identity, field))
@inline register_values(bc::LazyField) = constant_field(
    copyto!(register_similar(point_template(bc), eltype(bc)), bc),
)

"""
    requires_lockstep(arg)

Whether evaluating `arg` at a point reads values that require every thread of
its scope to read them at the same time (see
[`DataLayouts.requires_lockstep`](@ref)). An operator that reads such an
argument at neighboring points can only branch on the point after reading it.
"""
@inline requires_lockstep(_) = false
@inline requires_lockstep(field::Field) =
    DataLayouts.requires_lockstep(Fields.field_values(field))
@inline requires_lockstep(bc::Broadcast.Broadcasted) =
    unrolled_any(requires_lockstep, bc.args)

# Disable constant propagation to reduce heap allocations during type inference.
@drop_constprop register_similar, buffer_similar
@drop_constprop materialize_buffer, maybe_private_buffer, own_point_values, fused_buffer
