"""
    OperatorStyle(slice_op)

The [`BroadcastStyle`](@ref) used by an [`AbstractOperator`](@ref) that runs
[`DataLayouts.foreach_slice`](@ref) with the given slice operator, which can
be either [`level`](@ref), [`slab`](@ref), or [`column`](@ref).
"""
struct OperatorStyle{S <: Union{typeof(level), typeof(slab), typeof(column)}} <:
       Fields.AbstractFieldStyle
end
OperatorStyle(slice_op) = OperatorStyle{typeof(slice_op)}()

slice_operator(::OperatorStyle{S}) where {S} = S.instance

Broadcast.BroadcastStyle(style::OperatorStyle, ::Fields.FieldStyle) = style

Broadcast.broadcasted(op::AbstractOperator, args...) = Broadcast.Broadcasted(
    Broadcast.BroadcastStyle(typeof(op)),
    op,
    unrolled_tuple_map(Broadcast.broadcastable, args),
)

@inline Fields.sliced_broadcasted(op::AbstractOperator, args, axes) =
    Broadcast.Broadcasted(Broadcast.BroadcastStyle(typeof(op)), op, args, axes)

const OperatorBroadcasted{F} = Broadcast.Broadcasted{<:OperatorStyle, <:Any, F}

Utilities.unsafe_eltype(bc::OperatorBroadcasted) = return_eltype(bc.f, bc.args...)
Broadcast._axes(bc::OperatorBroadcasted, ::Nothing) = return_space(bc.f, bc.args...)

# Replace every AbstractOperator broadcast with a pointwise broadcast that has
# the same eltype. Also drop the broadcast axes, since they aren't needed here.
drop_operators(arg) = arg
drop_operators(bc::OperatorBroadcasted) = Fields.sliced_broadcasted(
    bc.f, unrolled_tuple_map(drop_operators, bc.args), nothing,
)
drop_operators(bc::OperatorBroadcasted{<:AbstractOperator}) = Fields.sliced_broadcasted(
    Returns(new(eltype(bc))), unrolled_tuple_map(drop_operators, bc.args), nothing,
)

# Replace every AbstractOperator broadcast with the result of apply_operator.
apply_operators(arg) = arg
apply_operators(bc::OperatorBroadcasted) =
    Broadcast.broadcasted(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)
apply_operators(bc::OperatorBroadcasted{<:AbstractOperator}) =
    buffer_size(bc) <= MAX_BUFFER_SIZE ?
    inline_apply_operators(bc) : noinline_apply_operators(bc)

@inline inline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)
@noinline noinline_apply_operators(bc) =
    apply_operator(bc.f, unrolled_tuple_map(apply_operators, bc.args)...)

# Inline every operator unless its expression asks a block for more than CUDA's
# 48 KB of static shared memory (a compilation error). The budget is 192 bytes
# per thread of a 256-thread block; launched blocks hold at most 128 threads
# (MAX_SUBBLOCK_LAUNCH_THREADS in ext/cuda/scopes.jl), so a single expression
# reserves no more than half of the 48 KB. This allows two buffers to be live at
# the same time, but three or more can still overflow and cause a ptxas error.
const MAX_BUFFER_SIZE = 48 * 1024 ÷ 256

# Upper bound on the per-thread shared memory required to inline all operator
# applications in a broadcast expression, with each application charged two
# buffers of its largest element type. Assumes that each slice's scope has no
# fewer threads than points, which is guaranteed by DataLayouts.slice_subscope.
buffer_size(arg) = 0
buffer_size(bc::OperatorBroadcasted) = unrolled_sum(buffer_size, bc.args)
buffer_size(bc::OperatorBroadcasted{<:AbstractOperator}) =
    2 * unrolled_maximum(sizeof ∘ eltype, (bc, bc.args...)) +
    unrolled_sum(buffer_size, bc.args)

# Apply size/scope primitives to an operator-free pointwise equivalent of bc.
for f in (:size, :length, :ndims)
    @eval Base.$f(bc::OperatorBroadcasted) = $f(drop_operators(bc))
end
for f in (:DataScope, :shape_params, :inferred_size, :nelems)
    @eval DataLayouts.$f(bc::OperatorBroadcasted) = DataLayouts.$f(drop_operators(bc))
end

# Allocate materialized results from the broadcast's own data; the space-based
# LazyField fallback would allocate through coordinate data whose kernel-wide
# scope has no allocation method inside a fused slice loop.
Base.similar(bc::OperatorBroadcasted, ::Type{T}) where {T} = similar(drop_operators(bc), T)

# Drop duplicate pointers before sending bc to the GPU, then add them back in.
function Base.copyto!(
    dest::Fields.Field,
    bc::OperatorBroadcasted;
    mask = DataLayouts.NoMask(),
)
    slice_op = slice_operator(Broadcast.BroadcastStyle(bc))
    bc′ = toggle_placeholder_grids(bc, axes(dest))
    DataLayouts.foreach_slice(slice_op, dest, bc′; mask) do dest_slice, bc_slice′
        bc_slice = toggle_placeholder_grids(bc_slice′, axes(dest_slice))
        copyto!(dest_slice, apply_operators(bc_slice))
    end
    call_post_op_callback() && post_op_callback(dest, dest, bc; mask)
    return dest
end

"""
    PlaceholderGrid()

Singleton value that represents a [`Grids.AbstractGrid`](@ref). Replacing grids
with `PlaceholderGrid`s allows larger broadcasts to be passed into GPU kernels
without hitting a parameter memory limit, since the `Field`s in a broadcast
expression all contain pointers to the same grid data as the broadcast's
destination. Once inside a GPU kernel, every `PlaceholderGrid` is replaced by
its original grid, with redundant pointers stored in each thread's registers.
"""
struct PlaceholderGrid <: Grids.AbstractGrid end

for f in (:level, :slab, :column, :(Base.view))
    @eval $f(grid::PlaceholderGrid, inds...) = grid
end

struct PlaceholderGridAdaptor{G <: Grids.AbstractGrid}
    grid::G
end
Adapt.adapt_storage((; grid)::PlaceholderGridAdaptor, ::PlaceholderGrid) = grid
Adapt.adapt_storage(::PlaceholderGridAdaptor{G}, ::G) where {G <: Grids.AbstractGrid} =
    PlaceholderGrid()

"""
    toggle_placeholder_grids(bc, [space])

Replace every [`PlaceholderGrid`](@ref) in a broadcast expression with the grid
underlying the given `space`, or vice versa. The default `space` is `axes(bc)`.
"""
toggle_placeholder_grids(bc, space = axes(bc)) =
    Adapt.adapt(PlaceholderGridAdaptor(Spaces.grid(space)), bc)

"""
    has_private_buffers(arg)

Whether a buffer allocated for `arg` is private to the allocating thread, which
holds exactly when `arg`'s [`DataLayouts.DataScope`](@ref) is
[`DataLayouts.ThisThread`](@ref) (as on CPUs); otherwise buffers live in shared
memory and must obey the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline has_private_buffers(arg) = DataLayouts.DataScope(arg) == DataLayouts.ThisThread()

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
@inline function constant_field(arg)
    data = Fields.field_values(arg)
    has_private_buffers(arg) || DataLayouts.stored_in_registers(data) || return arg
    return Fields.Field(DataLayouts.rebuild(data, StaticArrays.SArray), axes(arg))
end

"""
    materialize_buffer(arg)

Like `Base.materialize(arg)`, but storing the result of a lazy `Broadcasted`
expression in a buffer from [`buffer_similar`](@ref) that every thread in the
argument's scope can read; register-resident `Field`s (see
[`register_similar`](@ref)) are also copied into such a buffer. On GPUs two
buffers of equal byte size can share memory, so callers must keep their
lifetimes disjoint; see the buffer reuse invariant in [`apply_operator`](@ref).
"""
@inline materialize_buffer(arg) = arg
@inline materialize_buffer(bc::Base.Broadcast.Broadcasted) = constant_field(
    copyto!(
        buffer_similar(bc, drop_auto_broadcasters(Utilities.safe_eltype(bc))),
        bc;
        mask = Spaces.get_mask(axes(bc)),
    ),
)
@inline materialize_buffer(arg::Fields.Field) =
    DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    materialize_buffer(Base.broadcasted(identity, arg)) : arg

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
@inline fused_buffer(bc::Base.Broadcast.Broadcasted) =
    has_private_buffers(bc) &&
    unrolled_all(arg -> !(arg isa Base.Broadcast.Broadcasted), bc.args) ? bc :
    materialize_buffer(bc)

"""
    register_similar(arg, T)

Like `Base.similar(arg, T)`, but with the new `Field`'s data in each thread's
registers; used for every [`apply_operator`](@ref) destination, which only its
own thread reads and writes. This lets two applications in one fused expression
be live at once (see the buffer reuse invariant in [`apply_operator`](@ref)).
"""
register_similar(arg, ::Type{T}) where {T} =
    Fields.Field(DataLayouts.register_similar(Fields.field_values(arg), T), axes(arg))

"""
    buffer_similar(arg, T)

Like `Base.similar(arg, T)`, but always allocated through the argument's
[`DataLayouts.DataScope`](@ref) (shared memory on GPUs), never in per-thread
registers; used for every buffer whose values cross a thread boundary.
"""
buffer_similar(arg, ::Type{T}) where {T} =
    Fields.Field(DataLayouts.buffer_similar(Fields.field_values(arg), T), axes(arg))

# Disable constant propagation to reduce heap allocations during type inference.
@drop_constprop materialize_buffer, maybe_private_buffer, fused_buffer
@drop_constprop register_similar, buffer_similar
