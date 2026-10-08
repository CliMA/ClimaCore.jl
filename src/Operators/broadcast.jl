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
    FT = Spaces.undertype(result_space(op, args...))
    args′ = unrolled_map(args) do arg
        is_auto_broadcastable(eltype(arg)) ?
        Broadcast.broadcasted(add_auto_broadcasters, arg) : arg
    end
    return Broadcast.Broadcasted(style, promote_bcs(op, FT), args′)
end

const NonPointwiseBroadcasted{F} = Broadcast.Broadcasted{<:OperatorStyle, <:Any, F}
const OperatorBroadcasted{Op <: AbstractOperator} = NonPointwiseBroadcasted{Op}

Utilities.unsafe_eltype(::OperatorBroadcasted) = return_eltype(bc.f, bc.args...)
@inline Fields.shared_space(::OperatorBroadcasted) = return_space(bc.f, bc.args...)

# The arguments of an operator can have spaces that differ from the result space
# (e.g., an F2C operator reads from a face space and writes to a center space),
# so they cannot be verified through check_broadcast_space.
@inline Fields.check_broadcast_space(space, bc::OperatorBroadcasted, only_check_size) =
    Fields.check_broadcast_space(
        space,
        Fields.local_geometry_field(axes(bc)),
        only_check_size,
    )

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
apply_operators(arg) = arg
apply_operators(bc::NonPointwiseBroadcasted) =
    Broadcast.broadcasted(bc.f, unrolled_map(apply_operators, bc.args)...)
apply_operators(bc::OperatorBroadcasted) =
    inlined_buffer_bytes(bc) <= MAX_INLINED_BUFFER_BYTES ?
    inline_apply_operators(bc) : noinline_apply_operators(bc)

@inline inline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_map(apply_operators, bc.args)...)
@noinline noinline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_map(apply_operators, bc.args)...)

function Base.copyto!(dest::Fields.Field, bc::NonPointwiseBroadcasted)
    copyto_slice!(dest_slice, bc_slice) = copyto!(dest_slice, apply_operators(bc_slice))
    DataLayouts.foreach_slice(slice_operator(bc.style), copyto_slice!, dest, bc)
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
registers; used for every buffer whose values cross a thread boundary.
"""
@inline buffer_similar(arg, ::Type{T}) where {T} =
    Field(DataLayouts.buffer_similar(Fields.field_values(arg), T), axes(arg))
@inline buffer_similar(arg) = buffer_similar(arg, drop_auto_broadcasters(eltype(arg)))

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
    fused_buffer(arg)

Like [`materialize_buffer`](@ref), but leaving `arg` lazy when it is a shallow
`Broadcasted` over materialized values and a single thread owns the data
(larger scopes have to materialize into shared memory).
"""
@inline fused_buffer(arg) = arg
@inline fused_buffer(bc::LazyField) =
    has_private_buffers(bc) && unrolled_all(arg -> !(arg isa LazyField), bc.args) ?
    bc : materialize_buffer(bc)

# Disable constant propagation to reduce heap allocations during type inference.
@drop_constprop register_similar, buffer_similar
@drop_constprop materialize_buffer, maybe_private_buffer, fused_buffer
