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
@inline function Broadcast.broadcasted(op::AbstractOperator, args...)
    args′ = unrolled_map(Broadcast.broadcastable, args)
    style = Broadcast.result_style(OperatorStyle(op), Broadcast.combine_styles(args′...))
    return Broadcast.broadcasted(style, op, args′...)
end
@inline function Fields.sliced_broadcasted(op::AbstractOperator, args, axes)
    style = Broadcast.result_style(OperatorStyle(op), Broadcast.combine_styles(args...))
    return Broadcast.Broadcasted(style, op, args, axes)
end

@inline function Broadcast.broadcasted(style::OperatorStyle, op::AbstractOperator, args...)
    FT = Spaces.undertype(return_space(op, args...))
    args′ = unrolled_map(args) do arg
        is_auto_broadcastable(eltype(arg)) ?
        Broadcast.broadcasted(add_auto_broadcasters, arg) : arg
    end
    return operator_broadcasted(style, promote_bcs(op, FT), args′)
end
# Methods for specific operators can rewrite their broadcasts (e.g., MatrixFields replaces
# linear finite difference operators with band matrix products).
operator_broadcasted(style, op, args) = Broadcast.Broadcasted(style, op, args)

const NonPointwiseBroadcasted{F} = Broadcast.Broadcasted{<:OperatorStyle, <:Any, F}
const OperatorBroadcasted{Op <: AbstractOperator} = NonPointwiseBroadcasted{Op}

# Base would flatten an operator in .&&/.|| into pointwise calls, so keep it as an operand.
for Op in (:AndAnd, :OrOr)
    @eval Broadcast.broadcasted(::Broadcast.$Op, a, bc::NonPointwiseBroadcasted) =
        Broadcast.broadcasted(Fields.ShortCircuit{Broadcast.$Op}(identity), a, bc)
end

Fields.sliced_function(slice, op::AbstractOperator, inds) = slice(op, inds...)
Utilities.unsafe_eltype(bc::OperatorBroadcasted) = return_eltype(bc.f, bc.args...)
# Like other instantiated broadcasts, an operator uses its stored axes.
@inline Fields.shared_space(bc::OperatorBroadcasted) =
    isnothing(bc.axes) ? return_space(bc.f, bc.args...) : bc.axes

# The destination of an operator, which becomes its stored axes, must have the shape and any
# staggering of its result.
@inline function Fields.check_broadcast_space(space, bc::OperatorBroadcasted, ::Val{true})
    result_space = return_space(bc.f, bc.args...)
    staggerings = (Spaces.staggering(space), Spaces.staggering(result_space))
    any(isnothing, staggerings) || check_staggering(space, typeof(staggerings[2]))
    Fields.check_broadcast_space(
        space,
        Fields.local_geometry_field(result_space),
        Val(true),
    )
end
@inline Fields.check_broadcast_space(space, bc::OperatorBroadcasted, ::Val{false}) =
    Fields.check_subspace(space, axes(bc))

# The arguments of an operator can have spaces that differ from the result space
# (e.g., an F2C operator reads from a face space and writes to a center space),
# so each argument must be checked against its own space (see also check_arg_spaces).
@inline function Broadcast._axes(bc::OperatorBroadcasted, ::Nothing)
    unrolled_foreach(bc.args) do arg
        @inline
        Fields.check_broadcast_space(Fields.shared_space(arg), arg, Val(false))
    end
    return Fields.shared_space(bc)
end
@drop_recursion_limits Broadcast._axes, Fields.check_broadcast_space

# Arguments are instantiated as well (unlike in Base), so that their slices keep their
# spaces: recomputing them in every slice of a loop makes deep expressions much slower.
# The arguments of each operator are compared here, once per top-level broadcast: slices
# need not have comparable spaces (e.g., a column field keeps its own column in every slice
# of a loop, including the bodies of slice loops, where broadcasts are also instantiated).
@inline function Broadcast.instantiate(bc::NonPointwiseBroadcasted)
    isnothing(bc.axes) || (
        bc isa OperatorBroadcasted ?
        Fields.check_broadcast_space(bc.axes, bc, Val(true)) :
        Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    )
    instantiated_args = unrolled_map(Broadcast.instantiate, bc.args)
    bc isa OperatorBroadcasted && length(bc.args) > 1 &&
        DataLayouts.needs_loop_setup(DataLayouts.DataScope(bc)) &&
        check_arg_spaces(bc.args)
    return Broadcast.Broadcasted(bc.style, bc.f, instantiated_args, axes(bc))
end

# The arguments of an operator must be on one space up to staggering, like the arguments of a
# pointwise broadcast without a destination (see Fields.shared_space), though a column or
# level of that space can be one of them; e.g., an upwind product's velocity must have its
# advected field's levels.
function check_arg_spaces(args)
    space = Fields.shared_space_ignoring_staggering(args)
    unrolled_foreach(args) do arg
        arg_space = Fields.staggered_like(Fields.shared_space(arg), space)
        isnothing(arg_space) || Fields.check_subspace(space, arg_space)
    end
end

# Reassign spatial data before using it to allocate destination fields, since
# the scope of a broadcast can be narrower than the scope of its spatial data.
@inline destination_template(arg) =
    DataLayouts.reassign(
        Fields.local_geometry_field(Fields.shared_space(arg)),
        DataLayouts.DataScope(arg),
    )

# A slice loop body inlines these, since a function that is not inlined would return the
# registers of its result (see register_similar) in an MArray on the heap.
@inline Base.similar(bc::NonPointwiseBroadcasted, ::Type{T}) where {T} =
    register_similar(bc, T)
@inline Base.copy(bc::NonPointwiseBroadcasted) = copyto!(similar(bc), bc)

single_field_buffer_bytes(arg, ::Type{T} = eltype(arg)) where {T} = sizeof(T) * length(arg)
# Bytes of a cached_arg buffer of T values at the points of arg (see cached_values).
cached_bytes(arg, ::Type{T} = eltype(arg)) where {T} =
    let scope = DataLayouts.DataScope(arg)
        caches_in_registers(scope, arg, Val(true)) ? 0 :
        DataLayouts.scoped_buffer_bytes(scope, single_field_buffer_bytes(arg, T))
    end

# Bytes of the shared buffers that an argument prepared for an operator reads (see
# cached_arg), from the types of their layouts, so that the bytes are known when the
# operator is compiled. A buffer that is live at the same time as these ones is placed
# after them (see buffer_similar). The bytes are Vals, whose sums are computed from type
# parameters (see add_bytes), since integer offsets do not constant-fold in kernels.
@inline shared_buffer_bytes(_) = Val(0)
@inline shared_buffer_bytes(field::Field) =
    DataLayouts.shared_buffer_bytes(Fields.field_values(field))
@inline shared_buffer_bytes(bc::Broadcast.Broadcasted) =
    unrolled_mapreduce(shared_buffer_bytes, add_bytes, bc.args; init = Val(0))
@inline add_bytes(::Val{a}, ::Val{b}) where {a, b} = Val(a + b)

# Whether the buffers of an expression fit in the buffer limit of its scope: by default, the
# buffers of the operator application in it that needs the most, since only the buffers of
# one application are live at a time (see the buffer reuse invariant in apply_operator). In
# the unlimited scopes of unsliced fields, any buffers fit and their bytes are not computed,
# since run-time sizes would make the decisions taken when broadcasts are built (e.g., in
# MatrixFields.projected_operand) type-unstable.
fits_buffers(bc, bytes = largest_application_bytes) =
    let limit = DataLayouts.total_buffer_limit(DataLayouts.DataScope(bc))
        limit == typemax(Int) || bytes(bc) <= limit
    end

# Bytes of the buffers that an operator application allocates (see buffer_bytes), for the
# application in an expression that needs the most. Buffers are in shared memory on GPUs,
# where a scope that exchanges values through registers allocates no buffer for them.
largest_application_bytes(arg) = 0
largest_application_bytes(bc::NonPointwiseBroadcasted) =
    unrolled_mapreduce(largest_application_bytes, max, bc.args; init = 0)
largest_application_bytes(bc::OperatorBroadcasted) = max(
    buffer_bytes(bc),
    unrolled_mapreduce(largest_application_bytes, max, bc.args; init = 0),
)

# Recursively replace all operator broadcasts with the result of apply_operator.
apply_operators(arg) = arg
apply_operators(bc::NonPointwiseBroadcasted) = apply_pointwise_operators(bc.style, bc)
apply_operators(bc::OperatorBroadcasted) = inline_apply_operators(bc)
@inline inline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_map(operator_arg, bc.args)...)

# The evaluation of pointwise functions of operators can depend on the style.
apply_pointwise_operators(_, bc) =
    Broadcast.broadcasted(bc.f, unrolled_map(apply_operators, bc.args)...)

# Whether the arguments can be passed to a function that is not inlined, which allocates
# an MArray argument on the heap.
can_compile_separately(args...) = !unrolled_any(has_mutable_registers, args)
has_mutable_registers(arg) = false
has_mutable_registers(field::Field) =
    has_mutable_registers(parent(Fields.field_values(field)))
has_mutable_registers(bc::Base.AbstractBroadcasted) =
    has_mutable_registers(bc.f) || unrolled_any(has_mutable_registers, bc.args)
# Storage that is not an SArray is an MArray or a field view of one.
has_mutable_registers(array::DataLayouts.RegisterArray) =
    !(array.array isa StaticArrays.SArray)
has_mutable_registers(view::DataLayouts.RegisterPointView) =
    has_mutable_registers(parent(view))
has_mutable_registers(::StaticArrays.MArray) = true
has_mutable_registers(array::SubArray) = has_mutable_registers(parent(array))
@drop_recursion_limits has_mutable_registers

# Copy the register data of an expression's Fields into SArrays (see constant_field).
@inline freeze_registers(arg) = arg
@inline freeze_registers(field::Field) =
    has_mutable_registers(field) ? constant_field(field) : field
@inline freeze_registers(bc::Broadcast.Broadcasted) =
    has_mutable_registers(bc) ?
    Broadcast.Broadcasted(
        bc.style,
        freeze_registers(bc.f),
        unrolled_map(freeze_registers, bc.args),
        bc.axes,
    ) : bc
@drop_recursion_limits freeze_registers

# A broadcast rebuilt over new arguments keeps the space it was instantiated with, since the
# spaces of the arguments of a slice need not be comparable (see Fields.shared_space).
@inline function rebroadcasted(bc, args)
    bc′ = Broadcast.broadcasted(bc.f, args...)
    return Broadcast.Broadcasted(bc′.style, bc′.f, bc′.args, bc.axes)
end

# Like apply_operators, for an operator argument, which may keep operator broadcasts lazy;
# reads is how the operator reads it (see arg_reads).
operator_arg(arg, reads = Val(false)) = apply_operators(arg)

# Copy a slice into dest (spaces can differ); broadcasts skip .=, which infers loops twice.
@inline copyto_slice!(dest, bc::LazyField) =
    copyto!(Fields.field_values(dest), Broadcast.instantiate(Fields.field_values(bc)))
@inline copyto_slice!(dest, arg) = (dest .= arg)

# Like apply_operators, but copies the result into dest (styles can write it directly). A
# style whose passes end with a barrier that publishes dest (see apply_stencil!) skips it with
# finish = Val(false), for a loop that synchronizes once per statement (as a fused solver
# does); the result is then not safe to read at other threads' points until the next barrier.
apply_operators!(dest, bc, finish = Val(true)) = copyto_slice!(dest, apply_operators(bc))
# The buffers of a broadcast are those of its operators, which decides the level of shared
# memory of a broadcast's slice loop (see DataLayouts.foreach_slice).
DataLayouts.required_shared_memory(::typeof(apply_operators!), dest, bc) =
    largest_application_bytes(bc)
DataLayouts.uses_shared_memory(::typeof(apply_operators!), dest, bc) =
    uses_shared_buffers(bc)

# Whether evaluating an expression allocates any shared buffer (see
# DataLayouts.uses_shared_memory): by default, those of its operator applications (see
# buffer_bytes); stencil applications allocate buffers for the arguments they cache (see
# allocates_shared_buffer), which their bytes do not account for.
uses_shared_buffers(arg) = false
uses_shared_buffers(bc::NonPointwiseBroadcasted) =
    unrolled_any(uses_shared_buffers, bc.args)
uses_shared_buffers(bc::OperatorBroadcasted) =
    buffer_bytes(bc) > 0 || unrolled_any(uses_shared_buffers, bc.args)

# Whether an argument is in registers once its operators are applied (see apply_operators),
# so that an operator reading it at other threads' points copies it into a buffer.
in_registers_when_applied(_) = false
in_registers_when_applied(field::Field) =
    DataLayouts.stored_in_registers(Fields.field_values(field))
in_registers_when_applied(::OperatorBroadcasted) = true
in_registers_when_applied(bc::Broadcast.Broadcasted) =
    unrolled_any(in_registers_when_applied, bc.args)
@drop_recursion_limits in_registers_when_applied

# The inliner leaves large statements out of line, so a statement of a slice loop body
# freezes the temporaries that it reads (in registers on GPUs or in a thread's private
# buffers, as on CPUs), and it is inlined if it writes to one, since a function that is not
# inlined would put them on the heap. Statements outside of slice loops have no temporaries,
# so they skip the check.
@inline function Base.copyto!(dest::Fields.Field, bc::NonPointwiseBroadcasted)
    has_registers =
        !DataLayouts.needs_loop_setup(DataLayouts.DataScope(dest, bc)) &&
        !can_compile_separately(dest, bc)
    has_registers || return copyto_by_slices!(dest, bc)
    frozen_bc = freeze_registers(bc)
    return can_compile_separately(dest) ? copyto_by_slices!(dest, frozen_bc) :
           @inline(copyto_by_slices!(dest, frozen_bc))
end

# Writing to register temporaries in a function that is not inlined would put them on the heap.
@inline inline_apply_operators!(dest, bc) = @inline apply_operators!(dest, bc, Val(true))

# Ignore masks that cannot skip this type of slice; for example, a column mask
# is ignored when looping over slabs, since every column in a slab is needed to
# compute the operator's result.
function copyto_by_slices!(dest, bc)
    op = slice_operator(bc.style)
    space_mask = Spaces.get_mask(axes(dest))
    mask =
        DataLayouts.is_valid_slice_mask(space_mask, op) ? space_mask : DataLayouts.NoMask()
    # Replicate _foreach_slice's entry: every layer around a slice loop is inferred with it.
    DataLayouts.each_slice_index(op, dest) === DataLayouts.each_slice_index(op, bc) ||
        throw(DimensionMismatch("Inputs to foreach_slice must have compatible dimensions"))
    scope = DataLayouts.DataScope(dest, bc)
    if DataLayouts.needs_loop_setup(scope)
        DataLayouts._foreach_slice(
            scope, op, apply_operators!, mask, Val(false), Val(false), dest, bc,
        )
    else
        # Every statement of a slice loop body is compiled into the same function, and the
        # buffers of different statements are never live at once, so they share memory. A
        # statement first waits for the statements before it when it reads their pointwise
        # results at other threads' points (see reads_fields_at_neighbors).
        !has_private_buffers(bc) && reads_fields_at_neighbors(bc, Val(false)) &&
            DataLayouts.synchronize(scope)
        DataLayouts.scoped_slice_loop(
            DataLayouts.slice_subscope(scope, op, dest, bc),
            scope,
            op,
            can_compile_separately(dest) ? apply_operators! : inline_apply_operators!,
            mask,
            Val(false),
            dest,
            bc,
        )
        DataLayouts.share_single_point!(Fields.field_values(dest))
        # A statement into registers ends without a barrier, so one that reads Fields at other
        # threads' points then waits for its slowest thread, before the statements after it can
        # overwrite them.
        !has_private_buffers(bc) &&
            DataLayouts.stored_in_registers(Fields.field_values(dest)) &&
            reads_fields_at_neighbors(bc, Val(false)) &&
            DataLayouts.synchronize(scope)
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

Private buffers are used on CPUs, and on GPUs when one thread evaluates a whole
slice of [`DataLayouts.foreach_slice`](@ref) (e.g., a slice with one point).
"""
@inline has_private_buffers(arg) = DataLayouts.DataScope(arg) == DataLayouts.ThisThread()

"""
    register_similar(arg, T)

Allocate a `Field` like `Base.similar(arg, T)`, but with its data in each
thread's registers; used for every [`apply_operator`](@ref) destination, which only its
own thread reads and writes. This lets two applications in one fused expression
be live at once (see the buffer reuse invariant in [`apply_operator`](@ref)).
"""
@inline register_similar(arg, ::Type{T}) where {T} =
    Field(
        DataLayouts.register_similar(Fields.field_values(arg), T),
        Fields.shared_space(arg),
    )

# An operator broadcast has no data of its own, so its destination's template is allocated.
@inline register_similar(bc::NonPointwiseBroadcasted, ::Type{T}) where {T} =
    register_similar(destination_template(bc), T)

"""
    buffer_similar(arg, [T], [Val(offset)])

Allocate a `Field` like `Base.similar(arg, T)`, but always through the argument's
[`DataLayouts.DataScope`](@ref) (shared memory on GPUs), never in per-thread
registers; used for every buffer whose values cross a thread boundary. A buffer that is
live at the same time as another one is placed after it, at a byte `offset` of at least
the other's [`shared_buffer_bytes`](@ref) (see the buffer reuse invariant in
[`apply_operator`](@ref)).
"""
@inline buffer_similar(arg, ::Type{T}, offset::Val = Val(0)) where {T} = Field(
    DataLayouts.buffer_similar(Fields.field_values(arg), T, offset),
    Fields.shared_space(arg),
)
@inline buffer_similar(
    bc::NonPointwiseBroadcasted,
    ::Type{T},
    offset::Val = Val(0),
) where {T} =
    buffer_similar(destination_template(bc), T, offset)
@inline buffer_similar(arg) = buffer_similar(arg, eltype(arg))

"""
    constant_field(arg)

When `arg` is a `Field` with thread-local data (i.e., data that is either owned
by a single thread or stored in registers), copy its data into an immutable
`StaticArrays.SArray`, which is always stack-allocated. The default
`StaticArrays.MArray` used to store thread-local data is heap-allocated unless
every read and write is inlined, but such excessive inlining generally blows up
compilation time. GPUs only have stack memory, so this enables GPU compilation
without full inlining.
"""
@inline constant_field(arg) =
    has_private_buffers(arg) || DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    Field(
        DataLayouts.unchecked_rebuild(
            Fields.field_values(arg),
            frozen_array(parent(Fields.field_values(arg))),
        ),
        Fields.shared_space(arg),
    ) :
    arg

# A view (like a level of a column) is frozen as a view of its frozen parent, whose size is static.
@inline frozen_array(array) = StaticArrays.SArray(array)
@inline frozen_array(array::SubArray) = view(frozen_array(parent(array)), array.indices...)

"""
    materialize_buffer(arg, [Val(offset)])

Materialize `arg` like `Base.materialize(arg)`, but store the result of a lazy
`Broadcasted` expression in a buffer from [`buffer_similar`](@ref) that every
thread in the argument's scope can read; register-resident `Field`s (see
[`register_similar`](@ref)) are also copied into such a buffer, and other
arguments are returned unchanged. On GPUs buffers share memory unless they are given
different offsets (see [`buffer_similar`](@ref)), so callers must keep their lifetimes
disjoint; see the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline materialize_buffer(arg, offset = Val(0)) = arg
@inline materialize_buffer(arg::MaybeLazyField, offset = Val(0)) =
    arg isa LazyField || DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    constant_field(copyto!(buffer_similar(arg, eltype(arg), offset), arg)) : arg

"""
    maybe_private_buffer(arg)

Like [`materialize_buffer`](@ref), but only applied to arguments that are never
read across a thread boundary (avoiding the latency of a shared memory buffer).
"""
@inline maybe_private_buffer(arg) = has_private_buffers(arg) ? materialize_buffer(arg) : arg

"""
    own_point_values(arg)

Like [`maybe_private_buffer`](@ref), for an argument that each thread reads only at its own
points but once per horizontal dimension, so that a scope with registers evaluates it there
once. Arguments read at other threads' points (like `Restrict`'s) must not use this.
"""
@inline own_point_values(arg) =
    has_private_buffers(arg) ? materialize_buffer(arg) :
    arg isa MaybeLazyField &&
    DataLayouts.has_scope_registers(Fields.field_values(destination_template(arg))) ?
    register_values(arg) : arg

"""
    fused_buffer(arg)

Like [`cached_arg`](@ref) for an argument read without branching, but leaving `arg` lazy when it
is a shallow `Broadcasted` over materialized values and a single thread owns the data.
"""
@inline fused_buffer(arg) = arg
@inline fused_buffer(bc::LazyField) =
    has_private_buffers(bc) && unrolled_all(arg -> !(arg isa LazyField), bc.args) ?
    bc : cached_arg(bc, Val(true))

"""
    cached_arg(arg, branch_free, [Val(offset)])

Prepare an argument that an operator reads at points other than the ones it evaluates, so
that each of its values is computed once (by the thread that owns its point) and can be read
by every thread in its scope. Lazy expressions and `Field`s in registers are evaluated by
[`cached_values`](@ref), into a buffer (see the buffer reuse invariant in
[`apply_operator`](@ref)) or, when `branch_free` is `Val(true)` (the operator reads it
without branching on the point), possibly into registers. A buffer is placed at the byte
`offset` of [`buffer_similar`](@ref).
"""
@inline cached_arg(arg, branch_free, offset = Val(0)) =
    is_cached(arg) ? cached_values(DataLayouts.DataScope(arg), arg, branch_free, offset) :
    arg

"""
    cached_values(scope, arg, branch_free, [Val(offset)])

Evaluate an argument for [`cached_arg`](@ref) in a [`DataLayouts.DataScope`](@ref), by
default into a buffer from [`materialize_buffer`](@ref); devices whose threads can read each
other's registers keep arguments read without branching there (see [`register_values`](@ref)
and [`caches_in_registers`](@ref)).
"""
@inline cached_values(scope, arg, branch_free, offset = Val(0)) =
    materialize_buffer(arg, offset)

"""
    caches_in_registers(scope, arg, branch_free)

Whether [`cached_values`](@ref) keeps `arg` in registers in `scope` (by default, it does not),
so that caching it takes no shared memory.
"""
@inline caches_in_registers(_, _, _) = false

# Whether cached_arg evaluates its argument into a buffer or registers.
is_cached(_) = false
is_cached(field::Field) = DataLayouts.stored_in_registers(Fields.field_values(field))
is_cached(::LazyField) = true

"""
    register_values(arg, [Val(offset)])

Evaluate `arg` into the registers of the threads owning its points (unless already there).
A pass that evaluates `arg` can cache values in buffers after the byte `offset` (see
[`materialize_buffer`](@ref)).
"""
@inline register_values(field::Field) =
    DataLayouts.stored_in_scope_registers(Fields.field_values(field)) ? field :
    register_values(Base.broadcasted(identity, field))
@inline register_values(bc::LazyField) =
    constant_field(copyto!(register_similar(destination_template(bc), eltype(bc)), bc))
@inline register_values(arg, _) = register_values(arg)

"""
    allows_branching(arg)

Whether a thread may skip reading `arg` at a point by branching on the point; `false` for
values that every thread in the scope must read together, like other threads' registers (see
[`cached_values`](@ref)), which stencils therefore read before branching.
"""
@inline allows_branching(_) = true
@inline allows_branching(bc::Broadcast.Broadcasted) =
    unrolled_all(allows_branching, bc.args)

# Disable constant propagation to reduce heap allocations during type inference.
@drop_constprop register_similar, buffer_similar
@drop_constprop materialize_buffer, maybe_private_buffer, own_point_values, fused_buffer
