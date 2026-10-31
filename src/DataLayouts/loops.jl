macro simd_if(condition_expr, loop_expr)
    esc(:($condition_expr ? $(:(@simd $loop_expr)) : $loop_expr))
end

# Whether a point loop over these indices should run under @simd. This requires
# an indexable iterator, and it only helps when LLVM can vectorize across
# many contiguous points, like in CPU point loops. GPU scopes override this for
# their strided index subsets: each GPU thread only iterates a few points, and
# @simd's loop restructuring adds branches and index arithmetic to every kernel.
# The dispatch is on the index type rather than on the loop's scope, which would
# read as the more direct question, but the scope is a weaker discriminator than
# it looks: ThisThread is the scope of point loops on both hosts and devices.
@inline simd_over_indices(indices) = indices isa AbstractArray

@inline is_valid_slice_mask(::NoMask, _) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(view)) = true
@inline is_valid_slice_mask(::IJHMask, ::typeof(column)) = true
@inline is_valid_slice_mask(::IJHMask, _) = false

@inline all_indexables(arg) = (arg,)
@inline all_indexables(arg::Tuple) =
    unrolled_any(Base.Fix2(isa, Union{Tuple, NamedTuple}), arg) ?
    unrolled_flatmap(all_indexables, arg) : arg
@inline all_indexables(arg::NamedTuple) = unrolled_flatmap(all_indexables, arg)
@inline all_indexables(::NamedTuple{()}) = ()

@inline first_indexable(arg) = arg
@inline first_indexable(args::Union{Tuple, NamedTuple}) =
    first_indexable(first(all_indexables(args)))

# Unmasked point loops use linear indices when every argument supports them,
# avoiding the integer divisions of linear-to-Cartesian conversion. Static sizes
# need no divisions when only one axis is longer than 1, and only shifts when
# every axis is a power of two; loops over other static sizes (e.g., the slabs of
# spectral operators with 3 quadrature points) are faster with Cartesian indices
# on CPUs, and they compile to smaller GPU kernels.
@inline each_maskable_slice_index(_, op::O, args...) where {O} =
    each_slice_index(op, first_indexable(args))
@inline function each_maskable_slice_index(::NoMask, ::typeof(view), args...)
    arg = first_indexable(args)
    return has_inferred_size(arg) && !has_one_inferred_axis(arg) &&
           !unrolled_all(ispow2, inferred_size(arg)) ? each_slice_index(view, arg) :
           unrolled_eachindex(unrolled_map(first_indexable, all_indexables(args)))
end
@inline each_maskable_slice_index(mask::IJHMask, ::typeof(column), args...) =
    ActiveColumnIndices(mask)
@inline each_maskable_slice_index(mask::IJHMask, ::typeof(view), args...) =
    ActivePointIndices{nlevels(first_indexable(args))}(mask)

# Every valid mask and slice operator combination has indexable slice indices:
# NoMask uses the full index ranges, and IJHMask (which only supports column
# and view slices) uses compacted active indices, so no combination requires
# filtering the indices with a per-slice mask lookup.
@noinline throw_invalid_slice_mask(mask, op::O) where {O} =
    throw(ArgumentError(invalid_mask_string(mask, op)))
@generated invalid_mask_string(
    ::M,
    ::O,
) where {M, O} = "$M cannot be applied to $(O.instance) slices"

@inline function maskable_slice_indices(mask, op::O, args...) where {O}
    is_valid_slice_mask(mask, op) || throw_invalid_slice_mask(mask, op)
    return each_maskable_slice_index(mask, op, args...)
end

@inline subscope_slice_indices(subscope, scope, mask, op::O, args...) where {O} =
    @inbounds subscope_indices(
        subscope,
        scope,
        maskable_slice_indices(mask, op, args...),
    )

# Number of points in each slice that op generates from arg, measured from the
# slice via inferred_size so slice_subscope stays statically inferrable; a
# non-inferrable size is treated as unbounded, stopping slice_subscope's descent
# at the largest scope. A view slice contains one point by definition (point
# slices of fused broadcasts have inconsistent inferred_size across args).
@inline num_slice_points(::typeof(view), arg) = 1
@inline function num_slice_points(op::O, arg) where {O}
    slice = @inbounds op(arg, Tuple(first(each_slice_index(op, arg)))...)
    return has_inferred_size(slice) ? prod(inferred_size(slice)) : typemax(Int)
end

"""
    slice_subscope(scope, op, args...)

Return the [`DataScope`](@ref) that [`foreach_slice`](@ref) assigns to slices of
the given arguments when parallelizing over `scope`. By default, this is the
smallest scope, out of `scope` itself and its subsets, that does not require any
thread to process more than one point from the largest slice returned by `op`.
When no such scope is available, the largest of these scopes is used in order to minimize
the number of points per thread.

`scope` itself is only used when its thread count is a compile-time constant
(see [`static_num_threads`](@ref)), since `scoped_slice_loop` gives every
slice a scope with a statically known number of threads.
"""
@inline function slice_subscope(scope, op::O, args...) where {O}
    partition(scope) == ThisThread() && return ThisThread()
    # In a slice loop, a loop over one slice keeps the outer slice's scope: the subsets of
    # the scope that got no slice would skip every barrier of the slice's function. A point
    # has no barriers, and only one thread may update it.
    op !== view && is_nested_slice(scope, op, first_indexable(args)) && return scope
    max_slice_points =
        unrolled_maximum(Base.Fix1(num_slice_points, op), all_indexables(args))
    return points_subscope(scope, max_slice_points)
end
@inline is_nested_slice(scope, op, arg) =
    !isnothing(static_num_threads(scope)) && !needs_loop_setup(scope) &&
    has_inferred_size(arg) && isone(length(each_slice_index(op, arg)))

# Recursing on the precomputed point count (rather than on op and args) keeps
# the descent from re-materializing the argument tuple and re-slicing every
# argument at each scope level, which allocates once per kernel launch on hosts.
@inline function points_subscope(scope, max_slice_points::Int)
    subscope = partition(scope)
    subscope == ThisThread() && return subscope
    max_slice_points > num_threads(partition(subscope)) &&
        return slice_exceeds_subscope(scope, subscope, max_slice_points) ? scope : subscope
    return points_subscope(subscope, max_slice_points)
end

# Whether every subset of scope is too small for a slice (re-slicing in a slice loop): a
# subset's stride would not match outer register arrays, and shared buffers would double.
@inline slice_exceeds_subscope(scope, subscope, max_slice_points) =
    !isnothing(static_num_threads(scope)) &&
    something(static_num_threads(subscope), typemax(Int)) < max_slice_points

# The single definition of the slice-loop keyword defaults; every entry point
# slurps its keyword arguments and unpacks them here.
@inline function slice_loop_flags(;
    mask = NoMask(),
    enumerate = Val(false),
    shmem = Val(:auto),
    maxregs = Val(:auto),
)
    validate_shmem_level(shmem)
    validate_register_cap(maxregs)
    return (mask, enumerate, shmem, maxregs)
end

# The named levels of shared memory, whose bytes per thread the device sets (see
# with_shared_memory), from the least to the most.
const SHMEM_LEVELS = (:medium, :large, :max)

# Every level that foreach_slice accepts, checked at every entry point on every device
# (scopes without shared memory ignore the level), so that a misspelled level is not first
# found on a GPU. The check dispatches on the level's type, so it folds away for a valid level.
@inline validate_shmem_level(
    ::Union{Val{:auto}, Val{:none}, Val{:medium}, Val{:large}, Val{:max}},
) = nothing
@inline validate_shmem_level(level::Val{n}) where {n} =
    n isa Integer && n >= 0 ? nothing :
    throw(ArgumentError(invalid_shmem_level_string(level)))
validate_shmem_level(level) = throw(ArgumentError(invalid_shmem_level_string(level)))
function invalid_shmem_level_string(level)
    return "shmem must be Val(:auto), Val(:none), Val(:medium), Val(:large), Val(:max), or \
            Val(n) for n bytes per thread, got $(repr(level))"
end

# Every register cap that foreach_slice accepts (see maxregs), checked like the shmem level.
@inline validate_register_cap(cap::Val{R}) where {R} =
    (R isa Integer && R > 0) || R === :auto || R === :none ? nothing :
    throw(ArgumentError(invalid_register_cap_string(cap)))
validate_register_cap(cap) = throw(ArgumentError(invalid_register_cap_string(cap)))
function invalid_register_cap_string(cap)
    return "maxregs must be Val(:auto), Val(:none), or Val(n) for n registers per thread, \
            got $(repr(cap))"
end

"""
    with_shared_memory(scope, level)

Return a [`DataScope`](@ref) like `scope` whose slice loops give each slice the level of
shared memory `level` (see `shmem` in [`foreach_slice`](@ref)), or `scope` itself when
`level` is `nothing` or the scope has no shared memory (as on CPUs).
"""
with_shared_memory(scope, _) = scope

"""
    with_register_cap(scope, cap)

Return a [`DataScope`](@ref) like `scope` whose slice loops compile their kernels with the
register cap `cap` (see `maxregs` in [`foreach_slice`](@ref)), or `scope` itself when `cap`
is `nothing` or the scope has no register cap (as on CPUs).
"""
with_register_cap(scope, _) = scope

"""
    required_shared_memory(f, slices...)

Bytes of the shared memory of the slices' [`DataScope`](@ref) that the buffers of
`f(slices...)` need (see `total_buffer_limit`), found from the types of the slices alone,
or `nothing` when they cannot be known. Slice loops at the `:auto` level of `shmem` (see
[`foreach_slice`](@ref)) use this to choose the level of a loop; `nothing` by default,
since the buffers of an arbitrary function cannot be known, and extended for
`Operators.apply_operators!` (the function of a broadcast).
"""
required_shared_memory(f::F, slices...) where {F} = nothing

"""
    uses_shared_memory(f, slices...)

Whether `f(slices...)` allocates any buffer in the shared memory of the slices'
[`DataScope`](@ref) (see `scoped_static_array`), found from the types of the slices alone;
only read when [`required_shared_memory`](@ref) is known. Slice loops at the `:auto` level
of `shmem` (see [`foreach_slice`](@ref)) give a loop that allocates none no shared memory;
`true` by default, and extended for `Operators.apply_operators!` (the function of a
broadcast), whose buffers can fit in no bytes when the values that its operators cache stay
in registers.
"""
uses_shared_memory(f::F, slices...) where {F} = true

"""
    foreach_slice(op, f, args...; mask = NoMask(), enumerate = Val(false), shmem = Val(:auto), maxregs = Val(:auto))
    foreach_slice(scope, op, f, args...; mask = NoMask(), enumerate = Val(false), shmem = Val(:auto), maxregs = Val(:auto))

Apply `f` to slices of every [`DataLayout`](@ref) or similarly indexable argument,
as a generalization of `eachslice`/`mapslices`. The slice operator `op` can be any
of the following:

  - [`level`](@ref), but only when [`nelems`](@ref) is statically inferrable.
  - [`slab`](@ref) or [`column`](@ref).
  - `view` (for single-point slices).

Each slice is assigned to a [`slice_subscope`](@ref) of `scope`, by default the
largest available [`DataScope`](@ref) that can access every argument, and a
[`DataMask`](@ref) may be used to skip a particular subset of slices. On CUDA
devices, `foreach_slice` supports at most `2^31 - 1` slices.

Statements in `f` run at the slice's scope, and threads only wait for each other where
values cross them: same-shaped pointwise broadcasts assign every point to the same thread
in each statement, spectral operators publish values that cross threads through
synchronized buffers, finite difference operators that read Fields at neighboring levels
wait for the statements before them, and the statements after them wait for those reads,
a level of a temporary in registers is shared by the thread that owns it, and a statement
that reads or writes a single point outside of registers (e.g., `Fields.level` of a Field
of the loop), or whose operators read a Field as a boundary value, waits for the
statements before it, and the statements after it wait for it.

By default, `f` is called as `f(slices...)`. Setting `enumerate` to `Val(true)`
makes this `f(index, slices...)`, like in a loop over `Base.enumerate(arg)`.

An argument can also be a `Tuple` or `NamedTuple` of arguments, or a nested container of
them, whose slice is the same container of their slices. Every argument in a container is
scoped, masked, and checked for compatible dimensions like the other arguments. Containers
can be empty, as long as some argument is not. For example, this sets every column of `a`
to the sum of the columns of `b` and `c`:

```julia
foreach_column(a, (; b, c)) do a_column, (; b, c)
    @. a_column = b + c
end
```

Julia does not infer calls that splat more than 32 values, which would leave dynamic calls
in GPU kernels, so a loop over more than 31 arguments should group them into containers.

On GPUs, `shmem` sets how much of a block's shared memory each slice can use for the
buffers of the operators in `f` (see `Operators.apply_operator`), per thread of the slice.
This keyword is experimental, and may change in future:

  - `Val(:max)`: the most that a kernel can be launched with (163 KB per block on an A100,
    divided among the threads of a slice), at the cost of occupancy.
  - `Val(:large)`: the most that keeps shared memory from limiting the occupancy of a
    kernel below half (148 bytes per thread for slices of 64 threads on an A100).
  - `Val(:medium)`: the most that keeps shared memory from limiting the occupancy of a
    kernel (66 bytes per thread for slices of 64 threads on an A100).
  - `Val(n)`: `n` bytes.
  - `Val(:none)`: no shared memory, with each slice evaluated by one thread.
  - `Val(:auto)` (the default): the first level that holds the buffers of `f`. When they
    are known from the types of the slices (see [`required_shared_memory`](@ref), as for a
    broadcast), this is `Val(0)` if `f` allocates none (see [`uses_shared_memory`](@ref)),
    otherwise `Val(:medium)` if it holds them, or else `Val(n)` for exactly their bytes,
    and `Val(:none)` beyond `Val(:max)`. When they are not known (as for a loop body), the
    loop is launched at `Val(:medium)` if its slices are wider than a warp, and with no
    shared memory otherwise (slices of at most a warp cache values in registers).

Every kernel is launched with the shared memory of its level, which the L1 cache of a GPU
cannot use. A level is the most shared memory that a slice is given: a buffer that does
not fit is recomputed where it is read if it can be (see the buffer reuse invariant in
`Operators.apply_operator`), and otherwise the loop is compiled again at the next level
(`Val(:medium)`, `Val(:large)`, `Val(:max)`, then `Val(:none)`), with a warning that names
the bytes it needs the first time, which is remembered for the loop's later launches. At
`Val(:auto)`, a loop whose buffers are known is launched at a level that holds them, and
is never compiled again. The level is ignored on CPUs.

On GPUs, `similar` or `zero` of a slice inside `f` allocates a buffer at the start of the
slice's shared memory (see `scoped_static_array`), as does every statement that caches
values, which retires the buffer: it is only valid until then, so values that outlive the
next statement belong in a `Field` of the loop, or in a temporary `@.` broadcast, which
stays in registers.

On GPUs, `maxregs` caps the registers per thread of the kernel of a loop, so that more of
its blocks can be resident on a multiprocessor. This keyword is experimental, and may
change in future:

  - `Val(:auto)` (the default) and `Val(:none)`: no cap, leaving the count to the compiler.
  - `Val(n)`: `n` registers per thread, for every kind of slice.

Values that do not fit under the cap are spilled to local memory, so a cap suits some loop
bodies and not others: a fused body of several finite difference statements over columns
can run faster under a cap that keeps more of its blocks resident, while a pointwise
expression that needs many registers runs slower under any cap. The cap is ignored on CPUs.
"""
@inline function foreach_slice(op::O, f::F, args...; kwargs...) where {O, F}
    (mask, enumerate, shmem, maxregs) = slice_loop_flags(; kwargs...)
    return foreach_slice_with_options(shmem, maxregs, op, f, mask, enumerate, args...)
end

@inline _foreach_slice(op::O, f::F, mask, enumerate, args...) where {O, F} =
    foreach_slice_with_options(nothing, nothing, op, f, mask, enumerate, args...)

@inline function foreach_slice_with_options(
    shmem,
    maxregs,
    op::O,
    f::F,
    mask,
    enumerate,
    args...,
) where {O, F}
    indexables = all_indexables(args)
    isempty(indexables) &&
        throw(ArgumentError("Inputs to foreach_slice cannot all be empty containers"))
    unrolled_allequal(Base.Fix1(each_slice_index, op), indexables) ||
        throw(DimensionMismatch("Inputs to foreach_slice must have compatible dimensions"))
    data_scope =
        isone(length(indexables)) ? DataScope(indexables[1]) :
        unrolled_mapreduce(DataScope, DataScope, indexables)
    scope = with_register_cap(with_shared_memory(data_scope, shmem), maxregs)
    # Go straight onto the loop barrier for scopes without setup: every
    # forwarding layer adds a method instance per loop, inferred with the whole
    # loop inlined into it.
    return needs_loop_setup(scope) ?
           _foreach_slice(scope, op, f, mask, enumerate, args...) :
           scoped_slice_loop(
        slice_subscope(scope, op, args...), scope, op, f, mask, enumerate, args...,
    )
end

for (name, op, ref) in (
    (:foreach_point, :view, "`view`"),
    (:foreach_level, :level, "[`level`](@ref)"),
    (:foreach_slab, :slab, "[`slab`](@ref)"),
    (:foreach_column, :column, "[`column`](@ref)"),
)
    # These wrappers unpack the public keyword arguments once, at the entry
    # point, and pass them on positionally. Everything below them is positional
    # because each keyword-forwarding layer adds a Core.kwcall method instance
    # and a hidden body method instance per loop, each inferred with the whole
    # loop inlined into it, whether or not it is @inline.
    @eval begin
        """
            $($name)(f, args...; mask = NoMask(), enumerate = Val(false), shmem = Val(:auto), maxregs = Val(:auto))

        Run [`foreach_slice`](@ref) with $($ref) as the slice operator. The `shmem` and
        `maxregs` keywords are experimental, and may change in future.
        """
        @inline function $name(f::F, args...; kwargs...) where {F}
            (mask, enumerate, shmem, maxregs) = slice_loop_flags(; kwargs...)
            return foreach_slice_with_options(
                shmem, maxregs, $op, f, mask, enumerate, args...,
            )
        end
    end
end

# Whether looping over a DataScope requires work before and after the loop,
# done by the scope's own _foreach_slice method; extend alongside every method.
@inline needs_loop_setup(::DataScope) = false

# A thread pool has to be resolved before looping over it, and given back afterward.
@inline needs_loop_setup(::ThisThreadPool) = true
@inline function _foreach_slice(
    scope::ThisThreadPool,
    op::O,
    f::F,
    mask,
    enumerate,
    args...,
) where {O, F}
    # A single-threaded pool takes a shortcut around the whole claim/release
    # protocol. This is exactly equivalent to running it: claim_pool_threads
    # returns 0 without touching POOL_THREADS_IN_USE when the pool has one
    # thread, and PENDING_POOL_LOOPS is only incremented to be decremented
    # again, so no shared state would change. Folding one_thread into in_pool
    # is what skips the release in the finally block, and it cannot mask a real
    # nested loop: pool_thread_info is only set by launch_pool_threads and by a
    # resolve_pool_threads call that claims threads, and a one-thread pool
    # reaches neither.
    one_thread = isone(default_pool_size())
    in_pool = one_thread || pool_thread_info() != (0, 0)
    # The thread count stays an Int: a run-time-typed scope is a union in every call below.
    threads =
        one_thread ? 1 : (in_pool ? num_threads(scope) : resolve_pool_threads())
    try
        return isone(threads) ?
               scoped_slice_loop(
            ThisThread(), ThisThread(), op, f, mask, enumerate, args...,
        ) :
               parallelize_over(
            () -> scoped_slice_loop(
                slice_subscope(scope, op, args...), scope, op, f, mask, enumerate,
                args...,
            ),
            scope,
        )
    finally
        in_pool || release_pool_threads()
    end
end

@inline _foreach_slice(
    scope::DataScope, op::O, f::F, mask, enumerate, args...,
) where {O, F} =
    scoped_slice_loop(
        slice_subscope(scope, op, args...), scope, op, f, mask, enumerate, args...,
    )

@inline function foreach_slice(
    scope::DataScope, op::O, f::F, args...; kwargs...,
) where {O, F}
    (mask, enumerate, shmem, maxregs) = slice_loop_flags(; kwargs...)
    scope = with_register_cap(with_shared_memory(scope, shmem), maxregs)
    return _foreach_slice(scope, op, f, mask, enumerate, args...)
end

# The loop body lives behind a function barrier so that the subscope is a
# dispatch argument: a run-time slice size would then split a Union of scopes
# once instead of forcing correlated unions of all the argument slices at the
# call to f. The mask and enumerate flags are positional: another keyword layer
# would add a Core.kwcall method and a hidden body method per loop, each
# re-optimized with the whole loop inlined.
#
# Point loops need @simd and an inlined call to f for vectorization (the
# inliner gives up on point closures over large broadcasts); without them they
# are several times slower than ordinary Array broadcasts. Non-point slices do
# too much work per slice for vectorization to matter.
#
# NOTE: a function barrier here is not an option: a non-inlined call boxes the
# broadcast expression at least once per slab, allocating inside the tendency
# (68 kB per hyperdiffusion tendency for slab loops alone, 21 MB with point
# loops too).
@inline function scoped_slice_loop(
    subscope, scope, op::O, f::F, mask, enumerate, args...,
) where {O, F}
    indices = subscope_slice_indices(subscope, scope, mask, op, args...)
    scoped_args = unrolled_map(@inline(arg -> reassign(arg, subscope)), args)
    @simd_if (op == view && simd_over_indices(indices)) for i in index_positions(indices)
        index = @inbounds slice_index(indices, i)
        slices = @inbounds slice_every_arg(op, index, scoped_args...)
        @inline enumerate isa Val{true} ? f(index, slices...) : f(slices...)
    end
end

# Strided subsets (see StridedRange) are iterated by value, in their own integer type, so
# that GPUs neither divide out their length nor emulate a 64-bit loop counter.
@inline index_positions(indices) = 1:length(indices)
@inline index_positions(range::StridedRange) = range
@inline index_positions(strided::StridedCartesianIndices) = strided.view_range
const StridedActiveIndices = Union{
    ActiveColumnIndices{<:Any, <:StridedRange},
    ActivePointIndices{<:Any, <:Any, <:StridedRange},
}
@inline index_positions(indices::StridedActiveIndices) = indices.indices
Base.@propagate_inbounds slice_index(::StridedRange, i) = i
Base.@propagate_inbounds slice_index(strided::StridedCartesianIndices, i) =
    single_axis_cartesian_index(strided.indices, i)
Base.@propagate_inbounds slice_index(indices::StridedActiveIndices, i) =
    active_index(indices, i)

# When only the first axis of CartesianIndices is longer than 1, i is the position on it:
# LLVM does not always remove Base's division, which keeps CPU loops from vectorizing.
Base.@propagate_inbounds slice_index(indices, i) = indices[i]
Base.@propagate_inbounds slice_index(indices::CartesianIndices, i) =
    ndims(indices) > 1 && unrolled_all(isone, Base.tail(size(indices))) ?
    CartesianIndex(
        indices.indices[1][i], unrolled_map(first, Base.tail(indices.indices))...,
    ) : indices[i]

# One slice of every argument, and of every argument in a container (see foreach_slice). A
# generated function rather than an unrolled_map over a closure (two method instances per
# slice loop per argument-type combination); level/slab/column take a splatted Cartesian
# index, view takes the index itself, and the singleton operator leaves one inferred branch.
# The expression of the slice of an argument of type T is defined before the generator that
# calls it, which only sees methods defined before it.
function slice_expr(O, T, arg)
    T <: Union{Tuple, NamedTuple} || return O === typeof(view) ? :(view($arg, index)) :
           :(slice_arg(op, $arg, Tuple(index)...))
    slices = Expr(:tuple)
    for n in 1:fieldcount(T)
        push!(slices.args, slice_expr(O, fieldtype(T, n), :(getfield($arg, $n))))
    end
    return T <: Tuple ? slices : :(NamedTuple{$(fieldnames(T))}($slices))
end
@generated slice_every_arg(op::O, index, args...) where {O} = quote
    Base.@_propagate_inbounds_meta
    return $(slice_expr(O, Tuple{args...}, :args))
end

# Slices at a slice loop's own (valid) indices, so that types can skip their own checks
# (see Fields.slice_arg): Julia 1.10 keeps checks that @inbounds makes dead in the IR.
Base.@propagate_inbounds slice_arg(op::O, arg, indices...) where {O} = op(arg, indices...)

"""
    update_points!(f, op, data, slice_op)

Replace each value `x` at a point `index` of `data` with `op(x, f(index))` (or `f(index)`
if `op` is `nothing`), where the points are the slices that `slice_op` returns (e.g.,
`view`, or `column` for the single-point columns of a slab). Unlike [`foreach_point`](@ref),
this can be specialized for a [`DataScope`](@ref) to call `f` with all of its threads in
rounds, in which a thread without a point calls `f` at the last point and discards the
result, so that `f` can read values that every thread of the scope must read together (see
`Operators.allows_branching`).
"""
@inline function update_points!(f::F, op::O, data, slice_op::S) where {F, O, S}
    update_point!(index, point) =
        (@inbounds point[] = updated_value(op, point, @inline f(index)))
    scope = DataScope(data)
    needs_loop_setup(scope) ?
    _foreach_slice(slice_op, update_point!, NoMask(), Val(true), data) :
    scoped_slice_loop(
        slice_subscope(scope, slice_op, data),
        scope,
        slice_op,
        update_point!,
        NoMask(),
        Val(true),
        data,
    )
    return data
end
@inline updated_value(::Nothing, _, value) = value
@inline updated_value(op::O, point, value) where {O} = op(point[], value)

# Alternative to scoped_slice_loop that generates ordinary, unfused for loops,
# with every slice processed by the full scope instead of a slice_subscope.
@inline function unfused_slice_loop(scope, op::O, f::F, args...; kwargs...) where {O, F}
    (mask, enumerate, _, _) = slice_loop_flags(; kwargs...)
    for index in subscope_slice_indices(scope, scope, mask, op, args...)
        slices = @inbounds slice_every_arg(op, index, args...)
        enumerate isa Val{true} ? f(index, slices...) : f(slices...)
    end
end

"""
    reduce_points(op, arg; mask = NoMask(), [init])
    reduce_points(scope, op, arg; mask = NoMask(), [init])

Combine all values of a [`DataLayout`](@ref) or similarly indexable argument with
`op`, as a generalization of `reduce`, using the threads of `scope`, by default the
largest available [`DataScope`](@ref) that can access the argument. A
[`DataMask`](@ref) may be used to skip a particular subset of points. The
`init` value must be specified if the `mask` disables every point, if there are
no points in `arg` to begin with, or when reducing with any mask on a GPU,
where a mask can leave whole blocks without active points.
"""
@inline reduce_points(op::O, arg; mask = NoMask(), init...) where {O} =
    reduce_points(DataScope(arg), op, arg; mask, init...)

@inline function reduce_points(scope::ThisThreadPool, op::O, arg; kwargs...) where {O}
    # Reduce on this thread when the pool has one thread, or when there are too few points
    # to divide up, without resolving the pool or claiming any of its threads.
    (isone(default_pool_size()) || length(arg) <= default_pool_size()) &&
        return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
    T = thread_fold_type(op, arg; kwargs...)
    if pool_thread_info() != (0, 0)
        # A reduction nested in another loop gets a fresh results array, since the task's
        # reduction buffer may still be read by an outer reduction that is applying op.
        results = Array{T}(undef, num_threads(scope))
        return reduce_pool_points(results, scope, op, arg; kwargs...)
    end
    threads = resolve_pool_threads()
    try
        isone(threads) &&
            return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
        results = scoped_array(scope, T, threads; buffer = true)
        return reduce_pool_points(results, scope, op, arg; kwargs...)
    finally
        release_pool_threads()
    end
end

# Reduce over the threads of the current pool loop, storing each thread's share of the
# reduction in results. The results array is allocated by the caller, so that this method
# stays concretely typed whether it is given a view of the task's reduction buffer or a
# fresh array.
@inline function reduce_pool_points(
    results,
    scope::ThisThreadPool,
    op::O,
    arg;
    kwargs...,
) where {O}
    parallelize_over(scope) do
        @inbounds results[thread_rank(scope)] =
            reduce_points(ThisThread(), op, arg; kwargs...)
    end
    return reduce(op, results)
end

# Change the scope to ThisThread when given only one thread or a small argument.
# Otherwise, reduce each thread's values, then reduce the results in one thread.
@inline function reduce_points(scope::DataScope, op::O, arg; kwargs...) where {O}
    (isone(num_threads(scope)) || length(arg) <= num_threads(scope)) &&
        return reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
    T = return_type(op, NTuple{2, eltype(arg)})
    results = scoped_array(scope, T, num_threads(scope))
    parallelize_over(scope) do
        @inbounds results[thread_rank(scope)] =
            reduce_points(ThisThread(), op, arg; kwargs...)
    end
    return reduce(op, results)
end

# All indices a reduction over arg combines, before division among a
# DataScope's threads. Unmasked reductions use linear indices whenever they need
# no divisions (eachindex, or positions along the only axis of a slice like a
# column): CPU reductions vectorize an order of magnitude better over linear than
# Cartesian indices, and GPU reductions over linear indices need no divisions.
@inline reduced_point_indices(arg, mask) =
    mask != NoMask() ? maskable_slice_indices(mask, view, arg) :
    has_inferred_size(arg) && has_one_inferred_axis(arg) ? Base.OneTo(length(arg)) :
    eachindex(arg)

# Reduce all points assigned to this thread with safe_mapreduce (pairwise
# splitting for logarithmic roundoff, @simd for vectorization); Base's pairwise
# mapreduce has an empty-collection error path that cannot compile in GPU
# kernels. The @simd loops only vectorize sums and products, whose
# floating-point operations LLVM may reassociate, and only when the point
# values are read by inlined code. Inlining into other reductions (e.g., with
# the NaN-aware comparisons of min and max) can double their compilation time,
# while making them at most 20% faster.
#
# A thread can be assigned no points (slice_subscope rounds a slice up to a
# power of two, or a mask leaves fewer active points than threads). Such a
# thread folds a duplicate of the first point's value; every scope that can
# leave a thread empty excludes it from its own fold (see num_reduced_threads
# in the CUDA extension). A mask can also leave a whole kernel block empty,
# which the grid-level fold cannot exclude, so masked kernel reductions
# require an init value (enforced in the CUDA extension). With no points at
# all, only an init value can define the result, so empty reductions still
# require one.
@inline function reduce_points(::ThisThread, op::O, arg; mask, init...) where {O}
    all_indices = reduced_point_indices(arg, mask)
    indices = @inbounds subscope_indices(ThisThread(), DataScope(arg), all_indices)
    value =
        op isa Union{typeof(+), typeof(*), typeof(Base.add_sum), typeof(Base.mul_prod)} ?
        @inline(index -> @inbounds arg[index]) : (index -> @inbounds arg[index])
    isempty(init) &&
        isempty(indices) &&
        !isempty(all_indices) &&
        return value(@inbounds all_indices[firstindex(all_indices)])
    return safe_mapreduce(value, op, indices; init...)
end

# Type of the fold of op over a thread's points of arg (see reduce_points(::ThisThread)
# above), which an init value of a wider type than the points widens (e.g., a Float64 init
# over Float32 points), found by inference on arg at a thread's scope: the results of the
# threads of a multi-thread reduction are stored with this type, so that none is converted.
@inline thread_fold_type(op::O, arg; kwargs...) where {O} =
    return_type(Tuple{typeof(values(kwargs))}) do kwargs
        reduce_points(ThisThread(), op, reassign(arg, ThisThread()); kwargs...)
    end

"""
    column_reduce!(op, dest, arg; mask = NoMask(), flip = Val(false), [init])

Use [`foreach_column`](@ref) to combine the levels of each column of `arg` with
`op`, storing the results in the corresponding columns of `dest`. Setting `flip`
to `Val(true)` changes the order of reduction from left-associative (default) to
right-associative, and `init` seeds the fold when it is given.
"""
@inline column_reduce!(
    op::O,
    dest,
    arg;
    mask = NoMask(),
    flip = Val(false),
    init...,
) where {O} =
    foreach_column(dest, arg; mask) do dest_column, arg_column
        maybe_reverse = flip isa Val{true} ? reverse : identity
        fill!(dest_column, reduce(op, maybe_reverse(arg_column); init...))
    end

# TODO: Extend this to column_accumulate!

# A single point that every thread of a scope stores in registers (see register_similar) is
# written by every thread, once the statements before it are done.
@inline function written_layout(dest)
    stored_in_registers(dest) && isone(register_stride(parent(dest))) || return dest
    synchronize(DataScope(dest))
    return reassign(dest, ThisThread())
end

# After a writer that only writes the copy of the thread that owns a single point in registers
# (like an operator), every thread copies the owner's value.
@inline function share_single_point!(data)
    stored_in_registers(data) && isone(register_stride(parent(data))) || return nothing
    value = shared_level_value(DataScope(data), data, 1)
    @inbounds data[CartesianIndex(1, 1, 1, 1)] = value
    return nothing
end

# Whether the point at a linear index is in the calling thread's strided subset of points.
@inline owns_point(scope, index) =
    rem(index - 1, Int(num_threads(scope))) == Int(thread_rank(scope)) - 1

# Whether a layout or broadcast has a single point that is not in registers, like a level of a
# column of a slice loop, which one thread of a statement writes or every thread reads, and
# which the thread that owns it may have written in a statement before it or overwrite in a
# statement after it.
@inline reads_shared_points(_) = false
@inline reads_shared_points(data::DataLayout) =
    iszero(ndims(data)) && !stored_in_registers(data) &&
    !(parent(data) isa Union{StaticArrays.StaticArray, RegisterPointView})
@inline reads_shared_points(bc::LazyDataLayout) =
    unrolled_any(reads_shared_points, layout_args(bc))
@inline reads_shared_points(bc::FusedMultiBroadcast) =
    unrolled_any(
        pair -> reads_shared_points(pair.first) || reads_shared_points(pair.second),
        bc.pairs,
    )

# A statement of a slice loop that writes or reads such a point waits for the statements before
# it, and the statements after it wait for it.
@inline synchronize_shared_points(scope, args...) =
    unrolled_any(reads_shared_points, args) && !needs_loop_setup(scope) &&
    something(static_num_threads(scope), 1) > 1 && synchronize(scope)

# Convert the value before the fill! loop. Even though setindex! converts at
# every point, the compiler does not hoist the conversion, and filling a Float64
# layout with an Int is measurably slower if the conversion isn't done first.
# The converted value is passed to GPU kernels as parent array entries because
# Int128 and UInt128 fields in kernel arguments crash LLVM's NVPTX backend prior
# to LLVM 20 (llvm/llvm-project#49221). 128-bit integers are only safe in
# registers, like the ones bitcast_struct uses to reconstruct the value.
@inline function Base.fill!(dest::DataLayout, value; kwargs...)
    B = eltype(parent(dest))
    converted_value = convert(eltype(dest), value)
    entries = bitcast_struct(NTuple{num_basetypes(B, eltype(dest)), B}, converted_value)
    synchronize_shared_points(DataScope(dest), dest)
    foreach_point(written_layout(dest); kwargs...) do dest_point
        @inbounds dest_point[] = bitcast_struct(eltype(dest_point), entries)
    end
    synchronize_shared_points(DataScope(dest), dest)
    call_post_op_callback() && post_op_callback(dest, dest, value; kwargs...)
    return dest
end

# Every copyto! into a DataLayout takes its mask as an optional positional
# argument rather than a keyword, so that callers in Fields and Operators reach
# the loop without a Core.kwcall layer. Each keyword layer is a separate method
# instance with the whole loop inlined into it, and costs inference time even
# when it is @inline and has no recursion limit.
#
# Replicate Base's scalar broadcast copyto!, where data .= value becomes fill!,
# and any other scalar broadcast becomes a pointwise loop. Since materialize!
# attaches dest's axes, but foreach_point strips dest of its axes, the scalar
# broadcast must also have its axes dropped, mirroring how Base's instantiate
# drops scalar broadcast axes.
@inline function _copyto_scalar_broadcast!(
    dest::DataLayout,
    bc,
    mask::DataMask = NoMask(),
)
    if bc.f === identity && isone(length(bc.args)) && Broadcast.isflat(bc)
        @inbounds arg = first(bc.args)
        return @inbounds fill!(dest, arg isa Tuple ? first(arg) : arg[]; mask)
    end
    bc_without_axes = Broadcast.Broadcasted(bc.style, bc.f, bc.args)
    synchronize_shared_points(DataScope(dest), dest)
    _foreach_slice(
        view,
        dest_point -> (@inbounds dest_point[] = first(bc_without_axes)),
        mask,
        Val(false),
        written_layout(dest),
    )
    synchronize_shared_points(DataScope(dest), dest)
    call_post_op_callback() && post_op_callback(dest, dest, bc; mask)
    return dest
end

# The StaticArrayStyle{0} and AbstractBlockStyle{0} methods avoid ambiguities
# with StaticArrays and BlockArrays. ClimaCoreCUDAExt adds the corresponding
# AbstractGPUArrayStyle{0} method, which cannot be defined here because
# GPUArrays is not a dependency of ClimaCore.
for S in (
    :(<:Broadcast.AbstractArrayStyle{0}),
    :(<:StaticArrays.StaticArrayStyle{0}),
    :(<:BlockArrays.AbstractBlockStyle{0}),
)
    @eval @inline Base.copyto!(
        dest::DataLayout,
        bc::Broadcast.Broadcasted{$S},
        mask::DataMask = NoMask(),
    ) = _copyto_scalar_broadcast!(dest, bc, mask)
end

@inline is_scalar_or_length_one(arg) = true
@inline is_scalar_or_length_one(arg::Tuple) = isone(length(arg))
@inline is_scalar_or_length_one(bc::Broadcast.Broadcasted) =
    unrolled_all(is_scalar_or_length_one, bc.args)

# Single-element tuples are handled the same way as Refs. A multi-element tuple
# is indexed along the first dimension of dest, as in Base, and each active
# point of dest reads the broadcast at its own Cartesian index.
@inline function Base.copyto!(
    dest::DataLayout,
    bc::Broadcast.Broadcasted{Broadcast.Style{Tuple}},
    mask::DataMask = NoMask(),
)
    is_scalar_or_length_one(bc) && return copyto!(
        dest,
        convert(Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}}, bc),
        mask,
    )
    bc_with_axes = Broadcast.instantiate(
        Broadcast.Broadcasted{Nothing}(bc.f, bc.args, map(Base.OneTo, size(dest))),
    )
    _foreach_slice(
        view,
        (index, dest_point) ->
            (@inbounds dest_point[] = bc_with_axes[CartesianIndices(dest)[index]]),
        mask,
        Val(true),
        dest,
    )
    call_post_op_callback() && post_op_callback(dest, dest, bc; mask)
    return dest
end

@inline function Base.copyto!(
    dest::DataLayout,
    arg::MaybeLazyDataLayout,
    mask::DataMask = NoMask(),
)
    scope = DataScope(dest, arg)
    dest_and_arg = Broadcast.broadcasted(tuple, written_layout(dest), arg)
    size(dest) == size(dest_and_arg) ||
        throw(DimensionMismatch("DataLayout broadcast result exceeds destination size"))
    synchronize_shared_points(scope, dest, arg)
    _foreach_slice(
        view,
        dest_and_arg_point -> begin
            @inbounds (dest_point, arg_point) = dest_and_arg_point.args
            @inbounds dest_point[] = arg_point[]
        end,
        mask,
        Val(false),
        dest_and_arg,
    )
    synchronize_shared_points(scope, dest, arg)
    call_post_op_callback() && post_op_callback(dest, dest, arg; mask)
    return dest
end

@inline function Base.copyto!(bc::FusedMultiBroadcast; kwargs...)
    synchronize_shared_points(DataScope(bc), bc)
    foreach_point(bc; kwargs...) do bc_point
        unrolled_foreach(bc_point.pairs) do (dest_point, arg_point)
            @inbounds dest_point[] = arg_point[]
        end
    end
    synchronize_shared_points(DataScope(bc), bc)
    unrolled_foreach(pair -> share_single_point!(pair.first), bc.pairs)
    call_post_op_callback() && post_op_callback(bc, bc; kwargs...)
    return bc
end

@inline Base.copy(arg::MaybeLazyDataLayout) = copyto!(similar(arg), arg)

# Add axes to LazyDataLayouts and AutoBroadcaster wrappers to DataLayouts before
# reducing them. Remove all AutoBroadcaster wrappers after obtaining the result.
@inline function Base.reduce(
    op::O,
    arg::MaybeLazyDataLayout;
    mask = NoMask(),
    init...,
) where {O}
    # The containers in a reducible argument are AutoBroadcasters, and so is the container
    # that init may be, which keeps the result of a reduction with init concrete.
    reducible = arg isa LazyDataLayout ? Broadcast.instantiate : Broadcast.broadcastable
    reducible_init = map(add_auto_broadcasters, values(init))
    result =
        drop_auto_broadcasters(reduce_points(op, reducible(arg); mask, reducible_init...))
    call_post_op_callback() && post_op_callback(result, op, arg; mask, init...)
    return result
end

# Combine arguments for map!, map, and mapreduce into LazyDataLayouts.
@inline Base.map!(
    f::F,
    dest::DataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F} = copyto!(dest, Broadcast.broadcasted(f, args...); kwargs...)
@inline Base.map(
    f::F,
    arg::MaybeLazyDataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F} = copy(Broadcast.broadcasted(f, arg, args...); kwargs...)
@inline Base.mapreduce(
    f::F,
    op::O,
    arg::MaybeLazyDataLayout,
    args::MaybeLazyDataLayout...;
    kwargs...,
) where {F, O} = reduce(op, Broadcast.broadcasted(f, arg, args...); kwargs...)

# Avoid constructing a LazyDataLayout if the broadcast operation does nothing.
@inline Base.mapreduce(
    ::typeof(identity),
    op::O,
    arg::MaybeLazyDataLayout;
    kwargs...,
) where {O} = reduce(op, arg; kwargs...)

# Optimize unmasked equality checks for similar layouts with the same packed
# (un-padded) element types by deferring to their parent arrays. Padded values
# should not be compared in this way, since equality must not depend on padding.
# Parent arrays with mismatched dimensions are flattened before being compared.
@inline Base.:(==)(arg1::DataLayout, arg2::DataLayout; mask = NoMask()) =
    size(arg1) == size(arg2) && (
        mask != NoMask() ||
        !(eltype(arg1) == eltype(arg2) && Base.ispacked(eltype(arg1))) ||
        !(layout_type(arg1) == layout_type(arg2) && f_dim(arg1) == f_dim(arg2)) ?
        mapreduce(all ∘ ==, &, arg1, arg2; mask, init = true) :
        ndims(parent(arg1)) == ndims(parent(arg2)) ? parent(arg1) == parent(arg2) :
        stable_view(parent(arg1), :) == stable_view(parent(arg2), :)
    )
@inline Base.:(==)(arg1::MaybeLazyDataLayout, arg2::MaybeLazyDataLayout; kwargs...) =
    size(arg1) == size(arg2) &&
    mapreduce(all ∘ ==, &, arg1, arg2; init = true, kwargs...)
