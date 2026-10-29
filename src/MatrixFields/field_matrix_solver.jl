"""
    FieldMatrixSolverAlgorithm

Abstract type for algorithms that solve an equation of the form `A * x = b` for
`x`, where `A` is a `FieldMatrix` and where `x` and `b` are both `FieldVector`s.
Different algorithms can be nested inside each other, enabling the construction of
specialized linear solvers that use the sparsity pattern of `A`. Subtypes:
[`BlockDiagonalSolve`](@ref), [`BlockLowerTriangularSolve`](@ref),
[`BlockArrowheadSolve`](@ref), [`SchurComplementReductionSolve`](@ref), and
[`LazyFieldMatrixSolverAlgorithm`](@ref).

Every subtype of `FieldMatrixSolverAlgorithm` must implement methods for the
following functions:

  - [`check_field_matrix_solver`](@ref).
  - [`run_field_matrix_solver!`](@ref).

Subtypes can also implement methods for [`previous_solution`](@ref) and
[`solver_bytes`](@ref).
"""
abstract type FieldMatrixSolverAlgorithm end

"""
    check_field_matrix_solver(alg, A, b)

Check that the sparsity structure of `A` is supported by the
`FieldMatrixSolverAlgorithm` `alg`, and that `A` is compatible with `b` in the
equation `A * x = b`; throw an error otherwise.
"""
function check_field_matrix_solver end

"""
    run_field_matrix_solver!(alg, x, A, b, [previous_x, pool])

Set `x` to the value that solves the equation `A * x = b` using the
`FieldMatrixSolverAlgorithm` `alg`, where `previous_x` holds the solution of the previous
solve for correlated solves (see [`previous_solution`](@ref)); `b` and the entries of `A` can
be lazy (see [`lazy_mul`](@ref)). Temporary vectors and matrices are new `Field`s; in a loop
over columns, they are in the registers of the threads that compute them, or in a shared
buffer when other threads read them; in a loop of their own, they are taken from the `pool`
of the [`FieldMatrixSolver`](@ref) when there is one (see `pooled`). Return the entries of `x`
that are not zero by structure (see `is_zero_entry`), so that callers can skip products with
the others.

Methods should be inlined: when the columns of a GPU kernel are solved in one loop, a call
of a method that is too large to inline copies its arguments to local memory.
"""
function run_field_matrix_solver! end
@inline run_field_matrix_solver!(alg, x, A, b) =
    run_field_matrix_solver!(alg, x, A, b, empty_dict(b), nothing)

"""
    previous_solution(alg, A, b)

The entries of the solution of `A * x = b` that the correlated solves of the
`FieldMatrixSolverAlgorithm` `alg` (see [`StationaryIterativeSolve`](@ref)) keep between
calls of [`field_matrix_solve!`](@ref), as a `FieldNameDict` of zeros; empty by default.
"""
previous_solution(alg, A, b) = empty_dict(b)

# A FieldVectorView without entries, with the name tree of another one, and the
# FieldVectorView with the entries of two with different keys.
empty_dict(b) = FieldNameDict(FieldVectorKeys((), keys(b).name_tree), ())
function union_dicts(dict1, dict2)
    all_keys = union(keys(dict1), keys(dict2))
    entries = map(key -> key in keys(dict1) ? dict1[key] : dict2[key], all_keys)
    return FieldNameDict(all_keys, entries)
end

"""
    solver_bytes(alg, x, A, b)

Bytes of the shared buffers of the band matrix solves that [`run_field_matrix_solver!`](@ref)
fuses in the columns of a GPU kernel (the most of any solve, since every solve retires the
buffers of the one before it; see `shared_buffers`) and of the temporary vectors that other
threads read, found from the types of column slices of the states; `nothing` by default.
The loop of [`field_matrix_solve!`](@ref) is then compiled once, at the level of shared
memory that holds the buffers (see `DataLayouts.required_shared_memory`), rather than at
every level.
"""
solver_bytes(alg, x, A, b) = nothing

# The temporaries of solves in a loop of their own (see run_field_matrix_solver!), which the
# first solve allocates and every later solve takes in the same order.
mutable struct FieldPool
    const fields::Vector{Any}
    n::Int
end
FieldPool() = FieldPool([], 0)

# The next value of the pool, or a new one without a pool, from a constructor like similar
# and its arguments, whose types give the type of the value.
@inline pooled(::Nothing, f::F, args...) where {F} = f(args...)
@inline function pooled(pool::FieldPool, f::F, args...) where {F}
    n = pool.n += 1
    n > length(pool.fields) && push!(pool.fields, f(args...))
    return pool.fields[n]::return_type(f, typeof(args))
end

"""
    FieldMatrixSolver(alg, A, b)

Combination of a [`FieldMatrixSolverAlgorithm`](@ref) `alg`, the solution of the previous
solve that its correlated solves start from (see [`previous_solution`](@ref)), and the pool
of its temporaries in a loop of their own (see [`run_field_matrix_solver!`](@ref)). The
values of `A` and `b` passed to this constructor must be `similar` to the ones passed to
[`field_matrix_solve!`](@ref).
"""
struct FieldMatrixSolver{A <: FieldMatrixSolverAlgorithm, X}
    alg::A
    previous_x::X
    pool::FieldPool
end
function FieldMatrixSolver(
    alg::FieldMatrixSolverAlgorithm,
    A::FieldMatrix,
    b::Fields.FieldVector,
)
    b_view = field_vector_view(b)
    A_with_tree = replace_name_tree(A, keys(b_view).name_tree)
    check_field_matrix_solver(alg, A_with_tree, b_view)
    previous_x = previous_solution(alg, A_with_tree, b_view)
    return FieldMatrixSolver(alg, previous_x, FieldPool())
end

"""
    field_matrix_solve!(solver, x, A, b)

Solve the equation `A * x = b` for `x` in place using the [`FieldMatrixSolver`](@ref)
`solver`, and return `x`.
"""
NVTX.@annotate function field_matrix_solve!(
    solver::FieldMatrixSolver,
    x::Fields.FieldVector,
    A::FieldMatrix,
    b::Fields.FieldVector,
)
    (; alg, previous_x, pool) = solver
    x_view = field_vector_view(x)
    b_view = field_vector_view(b)
    keys(x_view) == keys(b_view) || error(
        "The linear system cannot be solved because x and b have incompatible \
         keys: $(set_string(keys(x_view))) vs. $(set_string(keys(b_view)))",
    )
    A_with_tree = replace_name_tree(A, keys(b_view).name_tree)
    check_field_matrix_solver(alg, A_with_tree, b_view)
    # On GPUs, every column is solved in one loop, unless the algorithm itself stores Fields
    # or logs norms of whole vectors, a Field has another mask or column shape than x₁ (e.g.,
    # a level Field with a masked extruded one, or a point Field with a spectral element one),
    # or the loop does not fit its arguments. On CPUs, setting up every statement of a solver
    # in each column takes longer than looping over the columns of each statement.
    x₁ = first(state_fields(x_view))
    states = (x_view, A_with_tree, b_view, previous_x)
    scope = DataLayouts.DataScope(x₁)
    if isbits(alg) && !is_debugging(alg) && !(scope isa DataLayouts.ThisThreadPool)
        # States without Fields (e.g., an empty previous solution) are captured by the function.
        constants = unrolled_map(
            state -> isempty(state_fields(state)) ? Some(state) : nothing,
            states,
        )
        sliced_states = unrolled_filter(state -> !isempty(state_fields(state)), states)
        f = ColumnwiseSolve(alg, constants)
        args = (x₁, sliced_states...)
        mask = Spaces.get_mask(axes(x₁))
        is_fusable = unrolled_all(unrolled_flatmap(state_fields, sliced_states)) do field
            Spaces.get_mask(axes(field)) === mask && column_shape(field) == column_shape(x₁)
        end
        if is_fusable && DataLayouts.fits_slice_loop(scope, column, f, mask, args)
            # Registers are capped: compilers keep values in registers across the statements
            # of a solve, so without a cap fewer columns are solved at once, which costs more
            # than spills.
            DataLayouts.foreach_slice_with_options(
                nothing,
                Val(80),
                column,
                f,
                mask,
                Val(false),
                Val(false),
                args,
            )
            return x
        end
    end
    pool.n = 0
    run_field_matrix_solver!(alg, states..., pool)
    return x
end

# Fill in every nothing in constants with the next slice, and unwrap the other constants.
@inline with_slices(constants, slices) =
    isempty(constants) ? () :
    isnothing(first(constants)) ?
    (first(slices), with_slices(Base.tail(constants), Base.tail(slices))...) :
    (something(first(constants)), with_slices(Base.tail(constants), slices)...)

# The function of the loop of field_matrix_solve!, which solves the slices after that of x₁
# (see with_slices). The band matrix solves and temporaries of the loop have buffers of known
# bytes (see solver_bytes), and its products cache arguments in shared memory within the
# level of the loop (see Operators.apply_operator), unless the scope of the slices keeps them
# in registers. Products read the solution at other threads' points, so it is solved in
# shared buffers after those of the solves when they fit, each entry after the one before it
# from the offset O that the loop finds from the types of its slices (see slice_parameter and
# staged_solution_offset), and then copied to x, whose global memory is slower to read.
struct ColumnwiseSolve{A, C, O}
    alg::A
    constants::C
end
ColumnwiseSolve(alg, constants, offset = nothing) =
    ColumnwiseSolve{typeof(alg), typeof(constants), offset}(alg, constants)
@inline_types_only function (solve::ColumnwiseSolve{<:Any, <:Any, O})(slices) where {O}
    (x, A, b, previous_x) = with_slices(solve.constants, Base.tail(slices))
    isnothing(O) && return run_field_matrix_solver!(solve.alg, x, A, b, previous_x, nothing)
    (entries, _) = unrolled_reduce(values(x), ((), Val(O))) do (entries, offset), entry
        buffer = Operators.buffer_similar(entry, eltype(entry), offset)
        next_offset = Operators.add_bytes(offset, Operators.shared_buffer_bytes(buffer))
        ((entries..., buffer), next_offset)
    end
    staged_x = FieldNameDict(keys(x), entries)
    run_field_matrix_solver!(solve.alg, staged_x, A, b, previous_x, nothing)
    copy_values!(unrolled_map(name -> copy_arg(x[name], staged_x[name]), keys(x).values))
end
DataLayouts.slice_parameter(solve::ColumnwiseSolve, slices) = staged_solution_offset(
    solve.alg,
    Base.front(with_slices(solve.constants, Base.tail(slices)))...,
)
DataLayouts.with_slice_parameter(solve::ColumnwiseSolve, ::Val{offset}) where {offset} =
    ColumnwiseSolve(solve.alg, solve.constants, offset)

# The bytes of a loop with a staged solution, at the level of its slices or else at their
# optional level (see DataLayouts.optional_buffer_scope).
function DataLayouts.required_shared_memory(solve::ColumnwiseSolve, slices)
    states = Base.front(with_slices(solve.constants, Base.tail(slices)))
    offset = staged_solution_offset(solve.alg, states...)
    isnothing(offset) || return offset + staged_solution_bytes(states[1])
    scope = DataLayouts.optional_buffer_scope(DataLayouts.DataScope(states[1]))
    if !isnothing(scope)
        optional_states = unrolled_map(Base.Fix2(DataLayouts.reassign, scope), states)
        offset = staged_solution_offset(solve.alg, optional_states...)
        isnothing(offset) || return offset + staged_solution_bytes(optional_states[1])
    end
    return solver_bytes(solve.alg, states...)
end
DataLayouts.uses_shared_memory(solve::ColumnwiseSolve, slices) =
    DataLayouts.required_shared_memory(solve, slices) > 0 ||
    !Operators.caches_in_registers(DataLayouts.DataScope(slices[1]), slices[1], Val(true))

# The bytes of the buffers of the solves, after which the solution is staged (see
# ColumnwiseSolve), or nothing if it is not: in a block diagonal solve, which does not read
# it, and when the buffers of the solves and of the solution do not fit in the scope or at
# its optional level.
function staged_solution_offset(alg, x, A, b)
    scope = DataLayouts.DataScope(x)
    optional_scope = DataLayouts.optional_buffer_scope(scope)
    (isnothing(optional_scope) || alg isa BlockDiagonalSolve) && return nothing
    limit =
        min(
            DataLayouts.total_buffer_limit(scope),
            DataLayouts.total_buffer_limit(optional_scope),
        ) - staged_solution_bytes(x)
    limit < 0 && return nothing
    bytes = solver_bytes(alg, x, A, b)
    return !isnothing(bytes) && bytes <= limit ? bytes : nothing
end

# The bytes of shared buffers for the entries of x (see ColumnwiseSolve).
staged_solution_bytes(x) =
    unrolled_sum(values(x)) do entry
        bytes = Operators.single_field_buffer_bytes(entry)
        DataLayouts.scoped_buffer_bytes(DataLayouts.DataScope(entry), bytes)
    end

# The bytes of several solves, or nothing if any are unknown.
max_bytes(bytes...) = unrolled_any(isnothing, bytes) ? nothing : max(bytes...)

is_debugging(alg) =
    unrolled_any(
        is_debugging,
        ntuple(Base.Fix1(getfield, alg), Val(fieldcount(typeof(alg)))),
    )

# A FieldNameDict passed to a slice loop is sliced, scoped, and reassigned like every Field
# in it (including those in its NamedTuple entries), which lets a solver's state be split
# into columns; NamedTuples passed to a slice loop are containers of arguments (see
# DataLayouts.foreach_slice).
const SolverState = Union{FieldNameDict, NamedTuple}
@inline state_fields(_) = ()
@inline state_fields(field::Fields.Field) = (field,)
@inline state_fields(state::SolverState) = unrolled_flatmap(state_fields, values(state))

@inline map_state_fields(_, entry) = entry
@inline map_state_fields(f::F, field::Fields.Field) where {F} = f(field)
@inline map_state_fields(f::F, state::NamedTuple) where {F} =
    NamedTuple{keys(state)}(map_entries(f, values(state)))
@inline map_state_fields(f::F, state::FieldNameDict) where {F} =
    FieldNameDict(keys(state), map_entries(f, values(state)))

# The recursion only passes through functions without recursion limits (no closures), and
# Base's map is avoided because it cannot infer results for more than 31 entries.
@inline map_entries(f::F, entries) where {F} =
    unrolled_map(map_state_fields, unrolled_map(Returns(f), entries), entries)
@drop_recursion_limits state_fields map_state_fields map_entries

@inline DataLayouts.DataScope(state::FieldNameDict) = unrolled_mapreduce(
    DataLayouts.DataScope,
    DataLayouts.DataScope,
    state_fields(state),
)
@inline DataLayouts.reassign(state::FieldNameDict, scope) =
    map_state_fields(Base.Fix2(DataLayouts.reassign, scope), state)
@inline DataLayouts.has_inferred_size(state::FieldNameDict) =
    unrolled_all(DataLayouts.has_inferred_size, state_fields(state))
@inline function DataLayouts.each_slice_index(::typeof(column), state::FieldNameDict)
    fields = state_fields(state)
    unrolled_allequal(Base.Fix1(DataLayouts.each_slice_index, column), fields) ||
        throw(DimensionMismatch("Fields of a solver state must have the same columns"))
    return DataLayouts.each_slice_index(column, first(fields))
end
@inline DataLayouts.num_slice_points(::typeof(column), state::FieldNameDict) =
    unrolled_maximum(Base.Fix1(DataLayouts.num_slice_points, column), state_fields(state))
Base.@propagate_inbounds DataLayouts.slice_arg(
    ::typeof(column),
    state::FieldNameDict,
    indices...,
) = map_state_fields(field -> DataLayouts.slice_arg(column, field, indices...), state)

# Toggling every Field separately avoids Adapt's recursion over nested NamedTuples, which
# leaves dynamic calls in kernels.
@inline Grids.toggle_placeholder_grid(state::FieldNameDict, grid::Grids.AbstractGrid) =
    map_state_fields(Base.Fix2(Grids.toggle_placeholder_grid, grid), state)

# Whether a FieldNameDict holds slices in a loop, like the column loop of field_matrix_solve!.
# A statement in such a loop ends with a barrier when it writes to the dict, since later
# statements can read values that other threads wrote.
@inline in_slice_loop(dict) = unrolled_any(is_sliced_field, values(dict))
@inline is_sliced_field(entry) =
    entry isa Fields.Field && !DataLayouts.needs_loop_setup(DataLayouts.DataScope(entry))
@inline synchronize_in_loop(dict) =
    in_slice_loop(dict) ?
    DataLayouts.synchronize(DataLayouts.DataScope(first(state_fields(dict)))) : nothing

function check_block_diagonal_matrix_has_no_missing_blocks(A, b)
    rows_with_missing_blocks =
        setdiff(keys(b), matrix_row_keys(matrix_diagonal_keys(keys(A))))
    missing_keys = corresponding_matrix_keys(rows_with_missing_blocks)
    # The missing keys correspond to zeros, and det(A) = 0 when A is a block
    # diagonal matrix with zeros along its diagonal. We can only solve A * x = b
    # if det(A) != 0, so we throw an error whenever there are missing keys.
    # Although it might still be the case that det(A) = 0 even if there are no
    # missing keys, this cannot be inferred during compilation.
    isempty(missing_keys) ||
        error("The linear system cannot be solved because A does not have any \
               entries at the following keys: $(set_string(missing_keys))")
end

function partition_blocks(names₁, A, b = nothing, x = nothing)
    keys₁ = FieldVectorKeys(names₁, keys(A).name_tree)
    keys₂ = static_call(set_complement, keys₁)
    A₁₁ = A[static_call(cartesian_product, keys₁, keys₁)]
    A₁₂ = A[static_call(cartesian_product, keys₁, keys₂)]
    A₂₁ = A[static_call(cartesian_product, keys₂, keys₁)]
    A₂₂ = A[static_call(cartesian_product, keys₂, keys₂)]
    b_blocks = isnothing(b) ? () : (b[keys₁], b[keys₂])
    x_blocks = isnothing(x) ? () : (x[keys₁], x[keys₂])
    return (A₁₁, A₁₂, A₂₁, A₂₂, b_blocks..., x_blocks...)
end

# A new Field for every entry of the solution of A * x = b (see x_eltype): in a slice loop, in
# registers, or in shared buffers that other threads can read when offset is a Val, with each
# entry placed after the one before it, starting at the offset (see Operators.buffer_similar).
function similar_to_x(A, b, offset = nothing)
    names = matrix_row_keys(keys(A))
    (entries, _) = unrolled_reduce(names.values, ((), offset)) do (entries, offset), name
        T = x_eltype(A[name, name], b[name])
        entry =
            isnothing(offset) ? Operators.register_similar(b[name], T) :
            Operators.buffer_similar(b[name], T, offset)
        next_offset =
            isnothing(offset) ? nothing :
            Operators.add_bytes(offset, Operators.shared_buffer_bytes(entry))
        ((entries..., entry), next_offset)
    end
    return FieldNameDict(names, entries)
end

# Bytes of the shared buffers of similar_to_x.
solution_bytes(A, b) =
    unrolled_sum(matrix_row_keys(keys(A)).values) do name
        bytes =
            Operators.single_field_buffer_bytes(b[name], x_eltype(A[name, name], b[name]))
        DataLayouts.scoped_buffer_bytes(DataLayouts.DataScope(b[name]), bytes)
    end

################################################################################

# Lazy (i.e., as matrix-free as possible) operations for FieldMatrix and
# analogues of FieldMatrix

import LazyBroadcast: lazy
lazy_inv(A) = Base.Broadcast.broadcasted(inv, A)
lazy_add(As...) = Base.Broadcast.broadcasted(+, As...)
lazy_sub(As...) = Base.Broadcast.broadcasted(-, As...)

# Whether an entry of a lazy FieldNameDict is zero by structure (a lazy zero of a Field, like
# the rows of a right-hand side that its matrix does not have; see schur_complement_rhs), and
# the keys of the other entries. Solves skip such entries, and products skip absent keys.
is_zero_entry(_) = false
is_zero_entry(::Base.Broadcast.Broadcasted{<:Any, <:Any, typeof(zero)}) = true

nonzero_keys(dict) = FieldVectorKeys(
    unrolled_filter(key -> !is_zero_entry(dict[key]), keys(dict).values),
    keys(dict).name_tree,
)

# A lazy Field evaluated into a new Field (from the pool in a loop of its own), or a lazy
# FieldNameDict with its lazy entries that are not zero by structure so evaluated (zeros take
# no Fields); in a slice loop, the new Fields are in
# registers (an immutable array on the stack of a thread that owns its whole slice, as on
# CPUs), whose points the statement updates directly (as in copyto_foreach!) instead of
# through a nested slice loop.
@inline function evaluated(bc::Fields.LazyField, pool = nothing)
    DataLayouts.needs_loop_setup(DataLayouts.DataScope(bc)) &&
        return evaluate!(pooled(pool, similar, bc), bc)
    Operators.has_private_buffers(bc) &&
        return Operators.materialize_buffer(Operators.apply_operators(bc))
    dest = similar(bc)
    Operators.apply_operators!(dest, bc, Val(false))
    return Operators.constant_field(dest)
end
@inline function evaluated(dict::FieldNameDict, pool = nothing)
    # In a slice loop, the lazy entries that are stencil expressions (e.g., products of band
    # matrices) over the same columns are evaluated in one loop over points (see
    # Operators.apply_stencils!), so that the loads of all of them are issued together.
    is_batched(entry) =
        entry isa Operators.StencilBroadcasted &&
        !DataLayouts.needs_loop_setup(DataLayouts.DataScope(entry)) &&
        !Operators.has_private_buffers(entry) &&
        Operators.num_caching_operators(entry) == 0
    batched = unrolled_map(
        entry -> (similar(entry), entry),
        unrolled_filter(is_batched, values(dict)),
    )
    Operators.apply_stencils!(batched)
    (entries, _) = unrolled_reduce(values(dict), ((), batched)) do (entries, batched), entry
        is_batched(entry) ?
        ((entries..., Operators.constant_field(batched[1][1])), Base.tail(batched)) :
        entry isa Fields.LazyField && !is_zero_entry(entry) ?
        ((entries..., evaluated(entry, pool)), batched) : ((entries..., entry), batched)
    end
    return FieldNameDict(keys(dict), entries)
end

# A lazy Field evaluated into a Field in a loop of its own; on CPUs, the children of a
# combined entry (see combined_entry) are evaluated in statements of their own, which are
# faster than one statement for all of them.
@inline evaluate!(dest, entry) = Base.materialize!(dest, entry)
@inline function evaluate!(dest, bc::Fields.LazyField)
    is_combined = bc.f === tuple || bc.f isa NamedTupleOf{fieldnames(eltype(dest))}
    is_combined && DataLayouts.DataScope(dest) isa DataLayouts.ThisThreadPool ||
        return Base.materialize!(dest, bc)
    unrolled_foreach(evaluate!, subfields(dest), bc.args)
    return dest
end

# A Field for the right-hand side of solves in a loop of their own, which evaluate_rhs!
# evaluates once for all of them (each solve would otherwise evaluate its entries into Fields
# of its own); in a slice loop, the solves evaluate the lazy right-hand side (into registers).
new_rhs(x, b, pool) = in_slice_loop(x) ? nothing : pooled(pool, similar, b)

evaluate_rhs!(::Nothing, lazy_b) = lazy_b
function evaluate_rhs!(b, lazy_b)
    keys_to_evaluate = nonzero_keys(lazy_b)
    foreach(key -> evaluate!(b[key], lazy_b[key]), keys_to_evaluate)
    return union_dicts(b[keys_to_evaluate], lazy_b)
end

# The lazy difference b - Ax with the keys of b, whose entries that the product Ax splits into
# children (e.g., a Field of NamedTuples whose components have their own matrix rows) are
# recombined (see Base.getindex), so that the solves of the result read whole entries once.
function lazy_residual(b, Ax)
    lazy_b = lazy_sub(b, Ax)
    keys(lazy_b) == keys(b) && return lazy_b
    return FieldNameDict(keys(b), map(key -> lazy_b[key], keys(b)))
end

"""
    lazy_mul(A, args...)

Construct a lazy `FieldMatrix` that represents the product `@. *(A, args...)`.
This involves regular broadcasting when `A` is a `FieldMatrix`, but it has more
complex behavior for other objects like the [`LazySchurComplement`](@ref).
"""
lazy_mul(A, args...) = Base.Broadcast.broadcasted(*, A, args...)

"""
    LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, [alg₁, x₁, b₁′, pool])

Analogue of a `FieldMatrix` that represents the Schur complement of `A₁₁` in `A`,
`A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`. Since `inv(A₁₁)` is generally a dense matrix,
computing the Schur complement directly is inefficient, so this object only
supports the "lazy" functions [`lazy_mul`](@ref), which multiplies it by the vector
`x₂`, and [`lazy_preconditioner`](@ref), which approximates it with a
`FieldMatrix`.

The values `alg₁`, `x₁`, `b₁′`, and `pool` must be specified for `lazy_mul` to compute
`inv(A₁₁) * A₁₂ * x₂`, which it stores in `x₁`, with `A₁₂ * x₂` evaluated into `b₁′` (see
`new_rhs`) and temporaries from the `pool` (see [`run_field_matrix_solver!`](@ref)); `x₁`
and `A₁₁` can be restricted to the entries that the product `A₂₁ * x₁` reads and the ones
they depend on (see `needed_keys`), the rows of `A₁₂ * x₂` that `A₁₂` does not have are
zero by structure, so `alg₁` skips their solves, and the product skips the entries of `x₁`
that are zero by structure. When a `LazySchurComplement` is not passed to `lazy_mul`, these
values can be omitted.
"""
struct LazySchurComplement{M11, M12, M21, M22, A1, V1, V2, P}
    A₁₁::M11
    A₁₂::M12
    A₂₁::M21
    A₂₂::M22
    alg₁::A1
    x₁::V1
    b₁′::V2
    pool::P
end
LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂) =
    LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, nothing, nothing, nothing, nothing)

Base.zero(lsc::LazySchurComplement) =
    LazySchurComplement(map(fn -> zero(getfield(lsc, fn)), fieldnames(lsc))...)

function lazy_mul(A₂₂′::LazySchurComplement, x₂)
    (; A₁₁, A₁₂, alg₁, x₁, b₁′, pool) = A₂₂′
    rhs = evaluate_rhs!(b₁′, schur_complement_rhs(A₁₂, x₂, x₁))
    nonzero_x₁ = run_field_matrix_solver!(alg₁, x₁, A₁₁, rhs, empty_dict(rhs), pool)
    return lazy_mul_template(A₂₂′, x₂, nonzero_x₁)
end

# The lazy product of lazy_mul without the solve of a LazySchurComplement, whose types are
# those of the product (with every entry of x₁, before the solve finds the zero ones).
lazy_mul_template(A, x) = lazy_mul(A, x)
lazy_mul_template(A₂₂′::LazySchurComplement, x₂, x₁ = A₂₂′.x₁) =
    lazy_sub(lazy_mul(A₂₂′.A₂₂, x₂), lazy_mul(A₂₂′.A₂₁, x₁))

# The right-hand side of the solve in lazy_mul, A₁₂ * x₂, with zeros at the rows of x₁ that
# A₁₂ does not have.
function schur_complement_rhs(A₁₂, x₂, x₁)
    zero_rows = static_call(setdiff, keys(x₁), matrix_row_keys(keys(A₁₂)))
    lazy_b = @. lazy(A₁₂ * x₂ + zero(x₁[zero_rows]))
    keys(lazy_b) == keys(x₁) && return lazy_b
    return FieldNameDict(keys(x₁), map(key -> lazy_b[key], keys(x₁)))
end

# The keys of the entries of x₁ that a Schur complement reads (the columns of A₂₁) and of the
# entries that they depend on through the off-diagonal entries of A₁₁ (closed by adding the
# columns of the rows in the set as many times as there are rows); the solves of inv(A₁₁) * b
# in lazy_mul and in the first step of SchurComplementReductionSolve skip the other entries.
function needed_keys(A₁₁, A₂₁)
    names = matrix_row_keys(keys(A₁₁))
    return unrolled_reduce(
        StaticOneTo(length(names)),
        matrix_col_keys(keys(A₂₁)),
    ) do keys₁, _
        coupling_keys = keys(A₁₁[cartesian_product(keys₁, setdiff(names, keys₁))])
        union(keys₁, matrix_col_keys(coupling_keys))
    end
end

"""
    LazyFieldMatrixSolverAlgorithm

A `FieldMatrixSolverAlgorithm` that does not require `A` to be a `FieldMatrix`,
i.e., a "matrix-free" algorithm. Internally, a `FieldMatrixSolverAlgorithm`
(for example, [`SchurComplementReductionSolve`](@ref)) might run a
`LazyFieldMatrixSolverAlgorithm` on a "lazy" representation of a `FieldMatrix`
(like a [`LazySchurComplement`](@ref)).

The only operations used by a `LazyFieldMatrixSolverAlgorithm` that depend on
`A` are [`lazy_mul`](@ref) and, when required, [`lazy_preconditioner`](@ref).
These and other lazy operations are used to minimize the number of calls to
`Base.materialize!`, since each call comes with a small performance penalty.
"""
abstract type LazyFieldMatrixSolverAlgorithm <: FieldMatrixSolverAlgorithm end

################################################################################

"""
    BlockDiagonalSolve()

A `FieldMatrixSolverAlgorithm` for a block diagonal matrix:

```math
A = \\begin{bmatrix}
     A_{11} & \\mathbf{0} & \\mathbf{0} & \\cdots & \\mathbf{0} \\\\
\\mathbf{0} &      A_{22} & \\mathbf{0} & \\cdots & \\mathbf{0} \\\\
\\mathbf{0} & \\mathbf{0} &      A_{33} & \\cdots & \\mathbf{0} \\\\
    \\vdots &     \\vdots &     \\vdots & \\ddots &     \\vdots \\\\
\\mathbf{0} & \\mathbf{0} & \\mathbf{0} & \\cdots &      A_{NN}
\\end{bmatrix}
```

This algorithm solves the `N` block equations `Aₙₙ * xₙ = bₙ` in sequence, setting
`xₙ` to zero without a solve when `bₙ` is zero by structure (see `is_zero_entry`).

If `Aₙₙ` is a diagonal matrix, the equation `Aₙₙ * xₙ = bₙ` is solved by making a
single pass over the data, setting each `xₙ[i]` to `inv(Aₙₙ[i, i]) * bₙ[i]`.

Otherwise, on a CPU, the equation `Aₙₙ * xₙ = bₙ` is solved using Gaussian elimination
(without pivoting), which makes two passes over the data. This is implemented for
band matrices `Aₙₙ` with any numbers of diagonals, and it has a specialized version for
tridiagonal matrices that compiles faster. In Gaussian
elimination, `Aₙₙ` is effectively factorized into the product `Lₙ * Dₙ * Uₙ`,
where `Dₙ` is a diagonal matrix, and where `Lₙ` and `Uₙ` are unit lower and upper
triangular matrices, respectively. The first pass multiplies both sides of the
equation by `inv(Lₙ * Dₙ)`, replacing `Aₙₙ` with `Uₙ` and `bₙ` with `Uₙxₙ`, which
is referred to as putting `Aₙₙ` into "reduced row echelon form". The second pass
solves `Uₙ * xₙ = Uₙxₙ` for `xₙ` with a unit upper triangular matrix solver, which
is referred to as "back substitution". These operations can become numerically
unstable when `Aₙₙ` has entries with large disparities in magnitude, but avoiding
this would require swapping the rows of `Aₙₙ` (i.e., replacing `Dₙ` with a
partial pivoting matrix).

On a GPU, where the points of each column are divided among multiple threads,
tridiagonal systems are instead solved with the parallel cyclic reduction (PCR)
method. PCR recursively decomposes each tridiagonal system into two systems of
half the size, until it is left with only scalar equations. The rows of each
column are exchanged through either shared memory or warp shuffles, depending on
whether the column fits in a warp. Other band matrices would require a more
complex implementation of PCR, so they are solved by performing Gaussian
elimination in one thread per column.
"""
struct BlockDiagonalSolve <: FieldMatrixSolverAlgorithm end

Base.zero(alg::BlockDiagonalSolve) = alg

function check_field_matrix_solver(::BlockDiagonalSolve, A, b)
    check_block_diagonal_matrix(
        A,
        "BlockDiagonalSolve cannot be used because A",
    )
    check_block_diagonal_matrix_has_no_missing_blocks(A, b)
    foreach(matrix_row_keys(keys(A))) do name
        check_single_field_solver(A[name, name], b[name])
    end
end

# Scalars and diagonal matrices can be inverted pointwise, so they do not
# benefit from using a fused solver that loops over columns. On CPUs, every
# matrix is solved in its own loop over columns, which is faster than one loop.
avoid_fusing_solver(::ScalingFieldMatrixEntry) = true
avoid_fusing_solver(A) =
    eltype(A) <: DiagonalMatrixRow ||
    DataLayouts.DataScope(A) isa DataLayouts.ThisThreadPool

# The argument of a column solve in a slice loop: the data of a Field, since the solves need
# no spaces, or a Field in registers (which a solve may have to copy into a buffer; see
# shared_values) or a lazy Field (only in the types of a solve; see solver_bytes). In a loop
# of its own, a solve evaluates lazy arguments into Fields first, since evaluating them in
# every column is slower than a loop over the columns of each.
solve_arg(arg) =
    arg isa Fields.Field && !DataLayouts.stored_in_registers(Fields.field_values(arg)) ?
    Fields.field_values(arg) : arg
materialized_values(arg, pool) =
    Fields.field_values(arg isa Fields.Field ? arg : evaluated(arg, pool))

# Solve every column directly, since a nested loop over one column could split
# the threads of the outer column, and only some of them would reach barriers.
# Recursion over the arguments compiles in GPU kernels where unrolled_foreach
# over partitions of them leaves dynamic calls. The rows of every parallel solve
# after the first are loaded before the solve that precedes it, whose barriers
# the loads would otherwise wait for (see solve_rows).
solve_columns!(_, ::Tuple{}) = nothing
solve_columns!(_, ::Tuple{}, _) = nothing
solve_columns!(shared, args) =
    solve_columns!(shared, args, solve_rows(shared, unrolled_take(args, Val(3))))
function solve_columns!(shared, args, rows)
    next_args = unrolled_drop(args, Val(3))
    next_rows =
        isempty(next_args) ? nothing :
        solve_rows(shared, unrolled_take(next_args, Val(3)))
    single_column_solve!(shared, solve_values(shared, unrolled_take(args, Val(3))), rows)
    return solve_columns!(shared, next_args, next_rows)
end

# The rows argument, which does not shrink like args, would widen the recursion into a
# dynamic call.
@drop_recursion_limits solve_columns!

band_matrices(::Tuple{}) = ()
band_matrices(args) = (args[2], band_matrices(unrolled_drop(args, Val(3)))...)

# The column shape of a Field's layout type (or of a lazy Field's space), with Nh = nothing
# when the number of elements is only known at run time (the loop checks it), so comparing
# shapes needs no run-time branch.
@inline function column_shape(arg)
    field = arg isa Fields.Field ? arg : Fields.local_geometry_field(axes(arg))
    (; Ni, Nj, Nh) = DataLayouts.vijh_params(Fields.field_values(field))
    return (Ni, Nj, Nh)
end

# The band matrix solves that a block diagonal solve fuses into one loop over columns, as
# one Tuple of their arguments (x, A, b); the other blocks are solved pointwise, or set to
# zero when their right-hand sides are zero by structure.
is_fused_solve(A, b, name) =
    !avoid_fusing_solver(A[name, name]) && !is_zero_entry(b[name])
fused_solves(x, A, b) =
    unrolled_flatmap(matrix_row_keys(keys(A)).values) do name
        is_fused_solve(A, b, name) ? (x[name], A[name, name], b[name]) : ()
    end

@inline function run_field_matrix_solver!(::BlockDiagonalSolve, x, A, b, _, pool)
    # In a slice loop, lazy entries of A and b are evaluated into registers before the solves,
    # which are then compiled once for all right-hand sides of a type.
    (A, b) = in_slice_loop(x) ? (evaluated(A), evaluated(b)) : (A, b)
    if in_slice_loop(x)
        names =
            split_by(name -> is_fused_solve(A, b, name), matrix_row_keys(keys(A)).values)[2]
        copy_values!(
            unrolled_map(names) do name
                value =
                    is_zero_entry(b[name]) ? b[name] :
                    pointwise_solution(A[name, name], b[name])
                copy_arg(x[name], value)
            end,
        )
    else
        foreach(matrix_row_keys(keys(A))) do name
            if is_zero_entry(b[name])
                x[name] .= b[name]
            elseif avoid_fusing_solver(A[name, name])
                single_field_solve!(x[name], A[name, name], b[name], pool)
            end
        end
    end
    args = fused_solves(x, A, b)
    if isempty(args) || !unrolled_allequal(column_shape, args)
        # Fields with different column shapes (e.g., on column and extruded spaces)
        # cannot share a loop, so every matrix is then solved in its own loop.
        foreach(matrix_row_keys(keys(A))) do name
            is_fused_solve(A, b, name) &&
                single_field_solve!(x[name], A[name, name], b[name], pool)
        end
    elseif in_slice_loop(x)
        # The previous statement ends with a barrier, so the columns are solved without a
        # nested loop, whose barriers would be redundant (see _foreach_slice).
        data_args = unrolled_map(solve_arg, args)
        solve_columns!(Val(shared_buffers(data_args)), data_args)
    else
        # The mask comes from a solution that is fused.
        mask = Spaces.get_mask(axes(args[1]))
        data_args = unrolled_map(Base.Fix2(materialized_values, pool), args)
        solve_columns_in_loops!(mask, data_args)
    end
    # In a slice loop, other threads can read the solutions in later statements.
    synchronize_in_loop(x)
    return x[nonzero_keys(b)]
end

# Lazy pairs of Fields and the values to copy into them in a slice loop (see copy_arg), copied
# in one statement per column size that reads every value before it writes any of them, so
# that the reads are issued together rather than one statement after another.
@inline function copy_values!(args)
    isempty(args) && return nothing
    size_of(arg) = DataLayouts.inferred_size(arg)
    (loop_args, other_args) = split_by(arg -> size_of(arg) == size_of(args[1]), args)
    DataLayouts._foreach_slice(
        view,
        points -> unrolled_foreach(
            (point, value) -> (@inbounds point.args[1][] = value),
            points,
            unrolled_map(point -> (@inbounds point.args[2][]), points),
        ),
        DataLayouts.NoMask(),
        Val(false),
        Val(false),
        loop_args,
    )
    return copy_values!(other_args)
end

# A single point in registers is written by every thread (see DataLayouts.written_layout).
@inline copy_arg(dest, value) = Base.Broadcast.broadcasted(
    tuple,
    DataLayouts.written_layout(Fields.field_values(dest)),
    Fields.field_values(value),
)

# The items of a Tuple for which f is true and the other items, like unrolled_split, by a
# recursion over their indices I and J with one recursive call, whose inference grows
# linearly with the number of items. Tuples of items are not splatted, since that leaves
# dynamic calls for over 32 items.
@inline split_by(f::F, itr) where {F} = split_by(f, itr, Val(((), ())))
@inline function split_by(f::F, itr, ::Val{indices}) where {F, indices}
    (I, J) = indices
    n = length(I) + length(J) + 1
    n > length(itr) && return isempty(J) ? (itr, ()) :
           (unrolled_map(m -> itr[m], I), unrolled_map(m -> itr[m], J))
    return split_by(f, itr, Val(f(itr[n]) ? ((I..., n), J) : (I, (J..., n))))
end
@drop_recursion_limits split_by

function solver_bytes(::BlockDiagonalSolve, x, A, b)
    data_args = unrolled_map(solve_arg, fused_solves(x, A, b))
    return isempty(data_args) ? 0 : shared_solve_bytes(data_args)
end

# Every solve is in one loop, unless the loop does not fit its arguments (e.g., in the
# parameter memory of a GPU kernel), in which case the solves are split in half.
function solve_columns_in_loops!(mask, data_args)
    slice_op = DataLayouts.column
    (solve_function, serial) = column_solve_function(solve_columns!, data_args)
    scope = DataLayouts.DataScope(data_args[1])
    fits = DataLayouts.fits_slice_loop(scope, slice_op, solve_function, mask, data_args)
    if fits || length(data_args) == 3
        DataLayouts._foreach_slice(
            slice_op,
            solve_function,
            mask,
            Val(false),
            serial,
            data_args,
        )
    else
        n = 3 * (length(data_args) ÷ 6)
        solve_columns_in_loops!(mask, unrolled_take(data_args, Val(n)))
        solve_columns_in_loops!(mask, unrolled_drop(data_args, Val(n)))
    end
end

"""
    BlockLowerTriangularSolve(names₁...; alg₁ = BlockDiagonalSolve(), alg₂ = BlockDiagonalSolve())

A `FieldMatrixSolverAlgorithm` for a 2×2 block lower triangular matrix:

```math
A = \\begin{bmatrix} A_{11} & \\mathbf{0} \\\\ A_{21} & A_{22} \\end{bmatrix}
```

The `FieldName`s in `names₁` correspond to the subscript `₁`, while all other
`FieldName`s correspond to the subscript `₂`. This algorithm has 2 steps:

 1. Solve `A₁₁ * x₁ = b₁` for `x₁` using the algorithm `alg₁`, which is set to a
    [`BlockDiagonalSolve`](@ref) by default.
 2. Solve `A₂₂ * x₂ = b₂ - A₂₁ * x₁` for `x₂` using the algorithm `alg₂`, which
    is also set to a `BlockDiagonalSolve` by default.
"""
struct BlockLowerTriangularSolve{
    N <: NTuple{<:Any, FieldName},
    A1 <: FieldMatrixSolverAlgorithm,
    A2 <: FieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    names₁::N
    alg₁::A1
    alg₂::A2
end
BlockLowerTriangularSolve(
    names₁...;
    alg₁ = BlockDiagonalSolve(),
    alg₂ = BlockDiagonalSolve(),
) = BlockLowerTriangularSolve(names₁, alg₁, alg₂)

Base.zero(alg::BlockLowerTriangularSolve) =
    BlockLowerTriangularSolve(alg.names₁, zero(alg.alg₁), zero(alg.alg₂))

function previous_solution(alg::BlockLowerTriangularSolve, A, b)
    A₁₁, _, _, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    return union_dicts(
        previous_solution(alg.alg₁, A₁₁, b₁),
        previous_solution(alg.alg₂, A₂₂, b₂),
    )
end

function check_field_matrix_solver(alg::BlockLowerTriangularSolve, A, b)
    A₁₁, A₁₂, _, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    isempty(keys(A₁₂)) || error(
        "BlockLowerTriangularSolve cannot be used because A has entries at the \
         following upper triangular keys: $(set_string(keys(A₁₂)))",
    )
    check_field_matrix_solver(alg.alg₁, A₁₁, b₁)
    check_field_matrix_solver(alg.alg₂, A₂₂, b₂)
end

@inline function run_field_matrix_solver!(
    alg::BlockLowerTriangularSolve,
    x,
    A,
    b,
    previous_x,
    pool,
)
    A₁₁, _, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    nonzero_x₁ = run_field_matrix_solver!(alg.alg₁, x₁, A₁₁, b₁, previous_x[keys(b₁)], pool)
    # In a slice loop, the solves of A₂₂ evaluate their right-hand side once, so it is lazy;
    # in a loop of their own, they can read it several times (e.g., a Schur complement
    # reduction reads its first block twice), so it is evaluated once.
    b₂′ = lazy_residual(b₂, lazy_mul(A₂₁, nonzero_x₁))
    b₂′ = in_slice_loop(x) ? b₂′ : evaluated(b₂′, pool)
    nonzero_x₂ =
        run_field_matrix_solver!(alg.alg₂, x₂, A₂₂, b₂′, previous_x[keys(b₂)], pool)
    return union_dicts(nonzero_x₁, nonzero_x₂)
end

function solver_bytes(alg::BlockLowerTriangularSolve, x, A, b)
    A₁₁, _, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    return max_bytes(
        solver_bytes(alg.alg₁, x₁, A₁₁, b₁),
        solver_bytes(alg.alg₂, x₂, A₂₂, lazy_residual(b₂, lazy_mul(A₂₁, x₁))),
    )
end

"""
    BlockArrowheadSolve(names₁...; alg₂ = BlockDiagonalSolve())

A `FieldMatrixSolverAlgorithm` for a 2×2 block arrowhead matrix:

```math
A = \\begin{bmatrix} A_{11} & A_{12} \\\\ A_{21} & A_{22} \\end{bmatrix}, \\quad
\\text{where } A_{11} \\text{ is a diagonal matrix}
```

The `FieldName`s in `names₁` correspond to the subscript `₁`, while all other
`FieldName`s correspond to the subscript `₂`. This algorithm has only 1 step:

 1. Solve `(A₂₂ - A₂₁ * inv(A₁₁) * A₁₂) * x₂ = b₂ - A₂₁ * inv(A₁₁) * b₁` for `x₂`
    using the algorithm `alg₂`, which is set to a [`BlockDiagonalSolve`](@ref) by
    default, and set `x₁` to `inv(A₁₁) * (b₁ - A₁₂ * x₂)`.

Since `A₁₁` is a diagonal matrix, `inv(A₁₁)` is also diagonal, so the Schur
complement of `A₁₁` in `A`, `A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`, as well as the vectors
`b₂ - A₂₁ * inv(A₁₁) * b₁` and `inv(A₁₁) * (b₁ - A₁₂ * x₂)`, can be computed
directly with band matrix operations.

This algorithm is equivalent to block Gaussian elimination with all operations
inlined into a single step.
"""
struct BlockArrowheadSolve{
    N <: NTuple{<:Any, FieldName},
    A <: FieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    names₁::N
    alg₂::A
end
BlockArrowheadSolve(names₁...; alg₂ = BlockDiagonalSolve()) =
    BlockArrowheadSolve(names₁, alg₂)

function previous_solution(alg::BlockArrowheadSolve, A, b)
    _, _, _, A₂₂, _, b₂ = partition_blocks(alg.names₁, A, b)
    return previous_solution(alg.alg₂, A₂₂, b₂)
end

function check_field_matrix_solver(alg::BlockArrowheadSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    check_diagonal_matrix(A₁₁, "BlockArrowheadSolve cannot be used because A")
    check_block_diagonal_matrix_has_no_missing_blocks(A₁₁, b₁)
    A₂₂′, _ = arrowhead_system(lazy_mul(A₂₁, lazy_inv(A₁₁)), A₁₂, A₂₂, b₁, b₂)
    check_field_matrix_solver(alg.alg₂, A₂₂′, b₂)
end

# The Schur complement of A₁₁ in A and the right-hand side of its equation (see
# BlockArrowheadSolve), given A₂₁ * inv(A₁₁), as lazy expressions that the solves of A₂₂′
# evaluate once.
arrowhead_system(A₂₁_invA₁₁, A₁₂, A₂₂, b₁, b₂) =
    (lazy_sub(A₂₂, lazy_mul(A₂₁_invA₁₁, A₁₂)), lazy_residual(b₂, lazy_mul(A₂₁_invA₁₁, b₁)))

@inline function run_field_matrix_solver!(
    alg::BlockArrowheadSolve,
    x,
    A,
    b,
    previous_x,
    pool,
)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    # Both products read A₂₁ * inv(A₁₁) only at their own points, so a scope with registers
    # evaluates it once (see Operators.own_point_values).
    A₂₁_invA₁₁ = lazy_mul(A₂₁, lazy_inv(A₁₁))
    A₂₁_invA₁₁ = FieldNameDict(
        keys(A₂₁_invA₁₁),
        unrolled_map(Operators.own_point_values, values(A₂₁_invA₁₁)),
    )
    A₂₂′, b₂′ = arrowhead_system(A₂₁_invA₁₁, A₁₂, A₂₂, b₁, b₂)
    run_field_matrix_solver!(alg.alg₂, x₂, A₂₂′, b₂′, previous_x[keys(b₂)], pool)
    @. x₁ = inv(A₁₁) * (b₁ - A₁₂ * x₂)
    return x
end

# The bytes are found from the types of lazy expressions, which are not evaluated.
function solver_bytes(alg::BlockArrowheadSolve, x, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, _, x₂ = partition_blocks(alg.names₁, A, b, x)
    A₂₂′, b₂′ = arrowhead_system(lazy_mul(A₂₁, lazy_inv(A₁₁)), A₁₂, A₂₂, b₁, b₂)
    return solver_bytes(alg.alg₂, x₂, A₂₂′, b₂′)
end

"""
    SchurComplementReductionSolve(names₁...; alg₁ = BlockDiagonalSolve(), alg₂)

A `FieldMatrixSolverAlgorithm` for any 2×2 block matrix:

```math
A = \\begin{bmatrix} A_{11} & A_{12} \\\\ A_{21} & A_{22} \\end{bmatrix}
```

The `FieldName`s in `names₁` correspond to the subscript `₁`, while all other
`FieldName`s correspond to the subscript `₂`. This algorithm has 3 steps:

 1. Solve `A₁₁ * x₁′ = b₁` for `x₁′` using the algorithm `alg₁`, which is set to
    a [`BlockDiagonalSolve`](@ref) by default (only for the entries of `x₁′` that
    step 2 reads, i.e., the columns of `A₂₁`, and the entries they depend on, unless
    `alg₁` has correlated solves).
 2. Solve `(A₂₂ - A₂₁ * inv(A₁₁) * A₁₂) * x₂ = b₂ - A₂₁ * x₁′` for `x₂`
    using the algorithm `alg₂`.
 3. Solve `A₁₁ * x₁ = b₁ - A₁₂ * x₂` for `x₁` using the algorithm `alg₁`.

Since `A₁₁` is not necessarily a diagonal matrix, `inv(A₁₁)` is generally a dense
matrix, which means that the Schur complement of `A₁₁` in `A`,
`A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`, cannot be computed efficiently. So, `alg₂` must be
a [`LazyFieldMatrixSolverAlgorithm`](@ref), which can evaluate the matrix-vector
product `(A₂₂ - A₂₁ * inv(A₁₁) * A₁₂) * x₂` without computing the Schur
complement matrix. This involves representing the Schur complement matrix by a
[`LazySchurComplement`](@ref), which uses `alg₁` to invert `A₁₁` when computing
the matrix-vector product.

This algorithm is equivalent to block Gaussian elimination, where steps 1 and 2
put `A` into reduced row echelon form, and step 3 performs back substitution.
For more information on this algorithm, see Section 5 of [Numerical solution of
saddle point problems](@cite Benzi2005).
"""
struct SchurComplementReductionSolve{
    N <: NTuple{<:Any, FieldName},
    A1 <: FieldMatrixSolverAlgorithm,
    A2 <: LazyFieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    names₁::N
    alg₁::A1
    alg₂::A2
end
SchurComplementReductionSolve(names₁...; alg₁ = BlockDiagonalSolve(), alg₂) =
    SchurComplementReductionSolve(names₁, alg₁, alg₂)

Base.zero(alg::SchurComplementReductionSolve) =
    SchurComplementReductionSolve(alg.names₁, zero(alg.alg₁), zero(alg.alg₂))

function previous_solution(alg::SchurComplementReductionSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    return union_dicts(
        previous_solution(alg.alg₁, A₁₁, b₁),
        previous_solution(alg.alg₂, LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂), b₂),
    )
end

function check_field_matrix_solver(alg::SchurComplementReductionSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    check_field_matrix_solver(alg.alg₁, A₁₁, b₁)
    check_field_matrix_solver(alg.alg₂, LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂), b₂)
end

@inline function run_field_matrix_solver!(
    alg::SchurComplementReductionSolve,
    x,
    A,
    b,
    previous_x,
    pool,
)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    # Only the entries of x₁′ = inv(A₁₁) * b₁ that the Schur complement reads are solved for
    # (see needed_keys), in the entries of x₁ that the last solve overwrites, unless alg₁ has
    # correlated solves, which keep previous solutions of every entry.
    keys₁′ =
        isempty(keys(previous_x[keys(b₁)])) ? needed_keys(A₁₁, A₂₁) : keys(b₁)
    A₁₁′ = A₁₁[cartesian_product(keys₁′, keys₁′)]
    x₁′ = x₁[keys₁′]
    b₁′ = new_rhs(x, b₁, pool)
    A₂₂′ = LazySchurComplement(A₁₁′, A₁₂, A₂₁, A₂₂, alg.alg₁, x₁′, b₁′, pool)
    run_field_matrix_solver!(alg.alg₁, x₁′, A₁₁′, b₁[keys₁′], previous_x[keys₁′], pool)
    # Every iteration of alg₂ reads its right-hand side, so it is evaluated once (in a slice
    # loop, the barrier keeps the solves of alg₂ from updating x₁ before it is evaluated).
    b₂′ = evaluated(lazy_residual(b₂, lazy_mul(A₂₁, x₁′)), pool)
    synchronize_in_loop(x)
    run_field_matrix_solver!(alg.alg₂, x₂, A₂₂′, b₂′, previous_x[keys(b₂)], pool)
    rhs = evaluate_rhs!(b₁′, lazy_residual(b₁, lazy_mul(A₁₂, x₂)))
    run_field_matrix_solver!(alg.alg₁, x₁, A₁₁, rhs, previous_x[keys(b₁)], pool)
    return x
end

# The solves of the Schur complement (see lazy_mul) have the types of the solves of A₁₁.
function solver_bytes(alg::SchurComplementReductionSolve, x, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    A₂₂′ = LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, alg.alg₁, x₁, nothing, nothing)
    return max_bytes(
        solver_bytes(alg.alg₁, x₁, A₁₁, b₁),
        solver_bytes(alg.alg₁, x₁, A₁₁, schur_complement_rhs(A₁₂, x₂, x₁)),
        solver_bytes(alg.alg₁, x₁, A₁₁, lazy_residual(b₁, lazy_mul(A₁₂, x₂))),
        solver_bytes(alg.alg₂, x₂, A₂₂′, lazy_residual(b₂, lazy_mul(A₂₁, x₁))),
    )
end
