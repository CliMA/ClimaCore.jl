import ..DataLayouts

# Each value of x is a product of a value of b with the inverse of a value of A.
inv_mul(a, b) = inv(a) * b
x_eltype(A::ScalingFieldMatrixEntry, b) =
    x_type(eltype(A), eltype(Base.broadcastable(b)))
x_eltype(A, b) = x_type(eltype(eltype(A)), eltype(Base.broadcastable(b)))

x_type(::Type{T_A}, ::Type{T_b}) where {T_A, T_b} = return_type(inv_mul, Tuple{T_A, T_b})

################################################################################

check_single_field_solver(::ScalingFieldMatrixEntry, _) = nothing
function check_single_field_solver(A, b)
    matrix_shape(A) == Square() || error(
        "Cannot solve linear system because a diagonal entry in A is not a \
         square matrix",
    )
    axes(A) === axes(b) || error(
        "Cannot solve linear system because a diagonal entry in A is not on \
         the same space as the corresponding entry in b",
    )
end

# Tridiagonal systems are solved using parallel cyclic reduction when multiple
# threads are available (and their rows fit in shared memory; see column_solve_function).
# Otherwise, one thread performs Gaussian elimination.
use_parallel_solver(A) =
    eltype(A) <: TridiagonalMatrixRow &&
    !(
        DataLayouts.DataScope(A) isa
        Union{DataLayouts.ThisThread, DataLayouts.ThisThreadPool}
    )

# Gaussian elimination factors a band matrix with rows of type R into L * D * U, and it
# stores U * x and the entries of U above its diagonal in every row (see band_matrix_solve!).
factors_type(::Type{R}, ::Type{T_b}) where {R, T_b} = Tuple{
    x_type(eltype(R), T_b),
    Vararg{x_type(eltype(R), eltype(R)), outer_diagonals(R)[2]},
}

# Bytes of the buffers of a solve with arguments (x, A, b). A parallel solve buffers the rows
# that its threads exchange (see pcr_neighbors). The one thread of any other solve buffers
# the arguments in registers, or lazy ones in band_solve_steps (see shared_values and
# register_values_buffer), and the factors that it computes. A loop needs the most that one
# of its solves needs, since every buffer of a solve is retired before the next solve
# allocates its own (see the buffer reuse invariant in Operators.apply_operator). Only data
# in registers counts for x: a Field in registers from similar_to_x stands for a shared
# buffer in band_solve_steps.
column_rows_bytes(x, A, b) =
    use_parallel_solver(A) ?
    exchanged_rows_bytes(
        DataLayouts.DataScope(x),
        x,
        sizeof(eltype(A)) + sizeof(eltype(Base.broadcastable(b))),
    ) : register_values_bytes(x) + shared_values_bytes(A) + shared_values_bytes(b)
column_factors_bytes(x, A, b) =
    use_parallel_solver(A) ? 0 : column_rows_bytes(x, A, b) + factors_bytes(x, A, b)
factors_bytes(x, A, b) = DataLayouts.scoped_buffer_bytes(
    DataLayouts.DataScope(x),
    DataLayouts.nlevels(x) * sizeof(factors_type(eltype(A), eltype(Base.broadcastable(b)))),
)
shared_values_bytes(arg) =
    Operators.is_cached(arg) ?
    DataLayouts.scoped_buffer_bytes(
        DataLayouts.DataScope(arg),
        Operators.single_field_buffer_bytes(arg, eltype(arg)),
    ) : register_values_bytes(arg)

register_values_bytes(_) = 0
register_values_bytes(data::DataLayouts.MaybeLazyDataLayout) =
    DataLayouts.stored_in_registers(data) ?
    DataLayouts.scoped_buffer_bytes(
        DataLayouts.DataScope(data),
        length(data) * sizeof(eltype(data)),
    ) : 0

loop_bytes(_, ::Tuple{}) = 0
loop_bytes(bytes, args) =
    max(bytes(args[1], args[2], args[3]), loop_bytes(bytes, unrolled_drop(args, Val(3))))

# Bytes of the buffer in which the threads of a column exchange rows of row_bytes bytes (see
# pcr_neighbors), which DataScopes that exchange them otherwise (e.g., through warp shuffles)
# can override.
exchanged_rows_bytes(scope, x, row_bytes) =
    DataLayouts.scoped_buffer_bytes(scope, DataLayouts.nlevels(x) * row_bytes)

# A loop that solves band matrices gives every column one thread, which keeps the factors in
# its own memory, unless a solve is parallel or has factors that take more than
# max_bytes_per_thread. A GPU kernel then gives every column the threads of a block. Their
# shared memory holds the rows of the parallel solves if they fit at the level of the loop
# (otherwise, one thread solves those systems by Gaussian elimination), and the factors if
# they also fit. Every solve is passed the buffers that it keeps in shared memory, or :copies
# when the thread of each column first copies the bands of A and the values of b into its
# own memory (see DataLayouts.copies_column_values).
function column_solve_function(f::F, args) where {F}
    scope = DataLayouts.DataScope(args[1])
    parallel =
        unrolled_any(use_parallel_solver, band_matrices(args)) ||
        loop_bytes(column_factors_bytes, args) > DataLayouts.max_bytes_per_thread(scope)
    return (
        ColumnSolves{!parallel && DataLayouts.copies_column_values(scope), F}(f),
        Val(!parallel),
    )
end

# The function of a loop that solves band matrices with f, which passes f the buffers that
# the solves of each column keep in shared memory (or :copies; see column_solve_function).
# These buffers are known from the types of the columns, so a GPU loop is only compiled at
# the level of shared memory that holds them (see DataLayouts.foreach_slice).
struct ColumnSolves{C, F}
    f::F
end
@inline (solves::ColumnSolves{C})(columns) where {C} =
    @inline solves.f(Val(C ? :copies : shared_buffers(columns)), columns)
DataLayouts.required_shared_memory(::ColumnSolves, columns) = shared_solve_bytes(columns)
DataLayouts.uses_shared_memory(solves::ColumnSolves, columns) =
    DataLayouts.required_shared_memory(solves, columns) > 0

# Bytes of shared memory that the solves of a loop need (see shared_buffers): the rows that
# parallel solves exchange, so that the loop is launched at a level that holds them (and
# with one thread per column beyond the largest level), and the factors of the other
# solves when they fit at the level of the columns' scope.
function shared_solve_bytes(args)
    factors = loop_bytes(column_factors_bytes, args)
    limit = DataLayouts.total_buffer_limit(DataLayouts.DataScope(args[1]))
    return max(loop_bytes(column_rows_bytes, args), factors <= limit ? factors : 0)
end

@inline function shared_buffers(args)
    scope = DataLayouts.DataScope(args[1])
    DataLayouts.static_num_threads(scope) == 1 && return :none
    limit = DataLayouts.total_buffer_limit(scope)
    loop_bytes(column_rows_bytes, args) <= limit || return :none
    return loop_bytes(column_factors_bytes, args) <= limit ? :all : :rows
end

# The solution of a scaling or diagonal block, lazily.
pointwise_solution(A::ScalingFieldMatrixEntry, b) =
    Base.broadcasted(*, (inv(scaling_value(A)),), b)
pointwise_solution(A, b) = @. lazy(inv(getindex(A, 0)) * b)

# A and b can be lazy, and temporaries come from the pool (see run_field_matrix_solver!).
single_field_solve!(x, A::ScalingFieldMatrixEntry, b, _ = nothing) =
    Base.materialize!(x, pointwise_solution(A, b))
single_field_solve!(x, A, b, pool = nothing) =
    if eltype(A) <: DiagonalMatrixRow
        Base.materialize!(x, pointwise_solution(A, b))
    else
        slice_op = DataLayouts.column
        mask = Spaces.get_mask(axes(x))
        args = map(Base.Fix2(materialized_values, pool), (x, A, b))
        (solve_function, serial) = column_solve_function(single_column_solve!, args)
        DataLayouts._foreach_slice(slice_op, solve_function, mask, Val(false), serial, args)
    end

# Solves need no cache, so the cache that callers of the four-argument method (e.g.,
# ClimaAtmos) get from single_field_solver_cache is nothing.
single_field_solver_cache(A, b) = nothing

single_field_solve!(::Nothing, x::Fields.Field, A, b) = single_field_solve!(x, A, b)

@inline subfields(field) =
    ntuple(@inline(n -> getproperty(field, n)), Val(fieldcount(eltype(field))))

# The arguments of a solve (see single_column_solve!), with Fields in registers (see
# solve_arg) made readable where the solve reads them: at each thread's own rows for the
# parallel solver, and at every row for the one thread of the other solvers, through
# shared buffers after the factors (see shared_values).
function solve_values(::Val{shared}, (x, A, b)) where {shared}
    shared != :none && use_parallel_solver(A) &&
        return (x, register_values(A), register_values(b))
    A_offset = shared == :all ? factors_bytes(x, A, b) : 0
    b_offset = A_offset + shared_values_bytes(A)
    return (x, shared_values(A, Val(A_offset)), shared_values(b, Val(b_offset)))
end

@inline register_values(arg) = arg
@inline register_values(field::Fields.Field) = Fields.field_values(field)

@inline function shared_values(arg, offset)
    Operators.is_cached(arg) || return arg
    # One thread (as on a CPU) copies the whole column into a buffer of its own.
    Operators.has_private_buffers(arg) &&
        return Fields.field_values(Operators.materialize_buffer(arg))
    buffer = Operators.buffer_similar(arg, eltype(arg), offset)
    if arg isa Fields.Field
        # Each thread copies its own points of a Field in registers.
        data = Fields.field_values(arg)
        DataLayouts._foreach_slice(
            view,
            (index, point) -> (@inbounds point[] = data[index]),
            DataLayouts.NoMask(),
            Val(true),
            Val(false),
            Fields.field_values(buffer),
        )
    else
        Operators.apply_operators!(buffer, arg, Val(false))
    end
    DataLayouts.synchronize(DataLayouts.DataScope(arg))
    return Fields.field_values(buffer)
end

# The parallel solver reads whole rows of A, so only the other solvers view its bands. One
# thread computes the factors of the other solvers. It keeps them in arrays of its own (on
# the stack of a CPU thread, or in the local memory of a GPU thread), or in shared memory
# before the buffers of the arguments in registers (see column_solve_function and
# register_values_buffer). The rows of a parallel solve can be loaded by the caller (see
# solve_rows).
single_column_solve!(shared, args) =
    single_column_solve!(shared, args, solve_rows(shared, args))
@inline function single_column_solve!(::Val{shared}, (x, A, b), rows) where {shared}
    shared != :none && use_parallel_solver(A) &&
        return parallel_band_matrix_solve!(Base.broadcastable(x), rows)
    scope = DataLayouts.DataScope(x)
    A_offset = Val(shared == :all ? factors_bytes(x, A, b) : 0)
    (A′, b_offset) = register_values_buffer(scope, A, A_offset)
    (b′, x_offset) = register_values_buffer(scope, b, b_offset)
    (x′, _) = register_values_buffer(scope, x, x_offset)
    T_U = factors_type(eltype(A′), eltype(Base.broadcastable(b′)))
    Uⱼs =
        shared == :all ? subfields(DataLayouts.buffer_similar(x, T_U)) :
        thread_arrays(T_U, Val(DataLayouts.nlevels(x)))
    DataLayouts.owns_point(scope, 1) && band_matrix_solve!(
        Uⱼs,
        Base.broadcastable(x′),
        solve_operands(Val(shared), x, subfields(A′.entries), Base.broadcastable(b′))...,
    )
    DataLayouts.stored_in_registers(x) || return nothing
    # Every thread copies its own rows of the solution into its registers, once the one
    # thread has written them.
    DataLayouts.synchronize(scope)
    for index in @inbounds DataLayouts.subscope_indices(
        DataLayouts.ThisThread(),
        scope,
        Base.OneTo(length(x)),
    )
        @inbounds x[index] = x′[index]
    end
end

@inline thread_arrays(::Type{T}, ::Val{n}) where {T, n} =
    ntuple(@inline(i -> MArray{Tuple{n}, fieldtype(T, i)}(undef)), Val(fieldcount(T)))

# The bands of A and the values of b that a solve reads at every row, copied into arrays of
# the solving thread by one loop over the rows when the solves of a column copy them (see
# column_solve_function).
@inline solve_operands(_, x, Aⱼs, b) = (Aⱼs, b)
@inline function solve_operands(::Val{:copies}, x, Aⱼs, b)
    operands = (Aⱼs..., b)
    copies = unrolled_map(operands) do operand
        @inline
        MArray{Tuple{DataLayouts.nlevels(x)}, eltype(operand)}(undef)
    end
    @inbounds for i in 1:DataLayouts.nlevels(x)
        unrolled_foreach(
            @inline(n -> @inbounds(copies[n][i] = operands[n][i])),
            StaticOneTo(length(operands)),
        )
    end
    return (Base.front(copies), last(copies))
end

# An argument of a solve in the registers of the threads of a scope (like a temporary of an
# enclosing loop; see DataLayouts.RegisterArray) can only be read and written by the threads
# that own its rows. Its values are copied into a buffer at a byte offset of shared memory,
# which the one thread of a solve can read and write at every row. Returns the buffer and
# the offset after it, or the argument and the given offset if it is not in registers.
@inline register_values_buffer(scope, arg, offset::Val{start}) where {start} =
    DataLayouts.stored_in_registers(arg) ?
    (
        DataLayouts.shared_values(scope, arg, offset),
        Val(start + register_values_bytes(arg)),
    ) :
    (arg, offset)

# The rows of a parallel solve, loaded from its values (see solve_values), or nothing for a
# serial solve, which copies its values into buffers that an earlier solve may still read.
solve_rows(::Val{:none}, _) = nothing
function solve_rows(shared, args)
    use_parallel_solver(args[2]) || return nothing
    (x, A, b) = solve_values(shared, args)
    # The indices of the calling thread's rows and the rows, which can be loaded before an
    # earlier solve of the same thread (see solve_columns!).
    x′ = Base.broadcastable(x)
    b′ = Base.broadcastable(b)
    scope = DataLayouts.DataScope(x′)
    n = length(x′) % Int32
    num_threads = DataLayouts.num_threads(scope) % Int32
    rank = DataLayouts.thread_rank(scope) % Int32
    num_rows = cld(length(x′), DataLayouts.static_num_threads(scope))
    indices = ntuple(Val(num_rows)) do row_number
        (row_number - 1) % Int32 * num_threads + rank
    end
    zero_A = zero(eltype(eltype(A)))
    rows = unrolled_map(indices) do i
        i = min(i, n)
        (A₋₁_value, A₀_value, A₊₁_value) = (@inbounds A[i]).entries
        (i > 1 ? A₋₁_value : zero_A, A₀_value, i < n ? A₊₁_value : zero_A, @inbounds b′[i])
    end
    return (indices, rows)
end

#=
The Thomas algorithm, as presented in
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Method,
but with the following variable name changes:
    - a → A₋₁
    - b → A₀
    - c → A₊₁
    - d → b
    - c′ → U₊₁
    - d′ → Ux
Transforms the tri-diagonal matrix into a unit upper bi-diagonal matrix, then
solves the resulting system using back substitution. The order of
multiplications has been modified in order to handle block vectors/matrices.
=#
@inline function band_matrix_solve!(Uⱼs::NTuple{2, Any}, x, Aⱼs::NTuple{3}, b)
    A₋₁, A₀, A₊₁ = Aⱼs
    Ux, U₊₁ = Uⱼs
    n = length(x)
    @inbounds begin
        # The last row also stores an unused entry of U, which avoids a branch in the loop.
        inv_D₀_value = inv(A₀[1])
        Ux_value = Ux[1] = inv_D₀_value * b[1]
        U₊₁_value = U₊₁[1] = inv_D₀_value * A₊₁[1]
        for i in 2:n
            inv_D₀_value = inv(A₀[i] - A₋₁[i] * U₊₁_value)
            Ux_value = Ux[i] = inv_D₀_value * (b[i] - A₋₁[i] * Ux_value)
            U₊₁_value = U₊₁[i] = inv_D₀_value * A₊₁[i]
        end

        x_value = x[n] = Ux_value
        for i′ in 1:(n - 1)
            i = n - i′
            x_value = x[i] = Ux[i] - U₊₁[i] * x_value
        end
    end
end

#=
Gaussian elimination without pivoting for a band matrix with any numbers of lower and
upper diagonals, ld and ud, which generalizes the Thomas algorithm above (used instead for
tridiagonal matrices, since it compiles faster). A is factored into L * D * U, where L is
unit lower triangular with ld diagonals below the main one, D is diagonal, and U is unit
upper triangular with ud diagonals above it. The forward sweep keeps the last ld rows of U
and U * x in registers, and the back substitution keeps the last ud rows of x; these
windows start as copies of the first row, whose terms are dropped near the ends of the
matrix. The order of multiplications handles block matrices.
=#
@inline Base.@constprop :none function band_matrix_solve!(Uⱼs, x, Aⱼs, b)
    (Ux, U₊s...) = Uⱼs
    ud = length(U₊s)
    ld = length(Aⱼs) - ud - 1
    n = length(x)
    @inbounds begin
        A_values = unrolled_map(@inline(A -> @inbounds(A[1])), Aⱼs)
        inv_D₀_value = inv(A_values[ld + 1])
        row = (
            inv_D₀_value * b[1],
            ntuple(@inline(m -> inv_D₀_value * A_values[ld + 1 + m]), Val(ud)),
        )
        store_factored_row!(Ux, U₊s, row, 1)
        rows = factor_rows!(Ux, U₊s, ntuple(Returns(row), Val(ld)), Aⱼs, b, 2:min(ld, n))
        factor_rows!(Ux, U₊s, rows, Aⱼs, b, max(ld + 1, 2):n)

        x_value = x[n] = Ux[n]
        xs = back_substitute!(x, Ux, U₊s, ntuple(Returns(x_value), Val(ud)), 2:min(ud, n))
        back_substitute!(x, Ux, U₊s, xs, max(ud + 1, 2):n)
    end
end

# Factors the rows is of A, given the window of the ld rows before them, and returns the
# window of the last ld rows.
@inline Base.@constprop :none function factor_rows!(Ux, U₊s, rows, Aⱼs, b, is)
    @inbounds for i in is
        row = factored_row(rows, Aⱼs, b, i)
        store_factored_row!(Ux, U₊s, row, i)
        rows = Base.tail((rows..., row))
    end
    return rows
end

# The values of U * x and U in row i, given the same values in the previous ld rows
# (farthest first), where the row at distance d is only used if d < i.
Base.@propagate_inbounds function factored_row(rows, Aⱼs, b, i)
    ld = length(rows)
    ud = length(Aⱼs) - ld - 1
    A_values = unrolled_map(@inline(A -> @inbounds(A[i])), Aⱼs)
    # Each previous row is eliminated with its lower entry after the elimination of the
    # rows before it; the nearest rows are subtracted first.
    L = unrolled_reduce(StaticOneTo(ld), ()) do L, _
        @inline
        (L..., eliminated_entry(A_values, L, rows, i, Val(length(L) - length(rows))))
    end
    inv_D₀_value = inv(eliminated_entry(A_values, L, rows, i, Val(0)))
    Ux_value = unrolled_reduce(StaticOneTo(ld), b[i]) do Ux_value, d
        @inline
        ifelse(d < i, Ux_value - L[ld + 1 - d] * rows[ld + 1 - d][1], Ux_value)
    end
    U₊_values = unrolled_map(ntuple(Val, Val(ud))) do val_m
        @inline
        inv_D₀_value * eliminated_entry(A_values, L, rows, i, val_m)
    end
    return (inv_D₀_value * Ux_value, U₊_values)
end

# The entry of row i of A in diagonal m after the elimination of the previous rows whose
# lower entries are in L.
Base.@propagate_inbounds function eliminated_entry(A_values, L, rows, i, ::Val{m}) where {m}
    ld = length(rows)
    ud = length(A_values) - ld - 1
    d_min = max(1, 1 - m)
    n_terms = max(min(ld, ud - m) + 1 - d_min, 0)
    return unrolled_reduce(StaticOneTo(n_terms), A_values[ld + 1 + m]) do A_value, d′
        @inline
        d = d_min - 1 + d′
        ifelse(d < i, A_value - L[ld + 1 - d] * rows[ld + 1 - d][2][m + d], A_value)
    end
end

# Stores the values of U * x and U in row i.
Base.@propagate_inbounds function store_factored_row!(Ux, U₊s, (Ux_value, U₊_values), i)
    Ux[i] = Ux_value
    unrolled_foreach(
        @inline(m -> @inbounds(U₊s[m][i] = U₊_values[m])),
        StaticOneTo(length(U₊s)),
    )
end

# Solves for x in rows n + 1 - i′ for i′ in i′s, given the window of the ud rows after them,
# and returns the window of the first ud rows.
@inline Base.@constprop :none function back_substitute!(x, Ux, U₊s, xs, i′s)
    @inbounds for i′ in i′s
        i = length(x) + 1 - i′
        x_value = x[i] = back_substituted_value(Ux, U₊s, xs, i, i′)
        xs = Base.front((x_value, xs...))
    end
    return xs
end

# The value of x in row i, given the values of x in the next ud rows (nearest first), where
# the row at distance m is only used if it is in the matrix (m < i′ = n + 1 - i).
Base.@propagate_inbounds back_substituted_value(Ux, U₊s, xs, i, i′) =
    unrolled_reduce(StaticOneTo(length(xs)), Ux[i]) do x_value, m
        @inline
        ifelse(m < i′, x_value - @inbounds(U₊s[m][i]) * xs[m], x_value)
    end

#=
Each method for band_matrix_solve! above has an order of operations that is
correct when x, A, and b are block vectors/matrices (i.e., when multiplication
is not necessarily commutative). So, the following are all valid combinations of
eltype(x), eltype(A), and eltype(b):
- Number, Number, and Number
- SVector{N}, SMatrix{N, N}, and SVector{N}
- Tensor{1} with Components B1, Tensor{2} (or its adjoint) with bases (B2, dual(B1)),
  and Tensor{1} with Components B2
- nested type (Tuple or NamedTuple), scalar type (Number, SMatrix, or
  Tensor{2}/adjoint thereof), nested type (Tuple or NamedTuple)
=#

#=
The parallel cyclic reduction (PCR) algorithm, as presented in
    https://doi.org/10.1145/1693453.1693472,
but with the following variable name changes:
    - a → A₋₁
    - b → A₀
    - c → A₊₁
    - d → b
    - k₁ → k₋
    - k₂ → k₊
Solves the tridiagonal system A * x = b through log2(n) steps, where n is the
number of rows. On each step, row i is decoupled from rows i - stride and
i + stride, and instead it is coupled to rows i - 2 * stride and i + 2 * stride,
until all rows are decoupled from one another when the stride length exceeds n.
Every thread updates the rows at its own indices, reading other rows by calling
pcr_neighbors. Threads past the last row repeat it, and rows outside of the matrix
are read as the nearest row in it, so that every load and exchange is uniform; their
couplings are then dropped by zeroing k₋ or k₊ instead of loading zeros. The order
of multiplications has been modified to handle block vectors/matrices.
=#
@inline function parallel_band_matrix_solve!(x, (indices, rows))
    scope = DataLayouts.DataScope(x)
    n = length(x) % Int32
    stride = one(n)
    while stride < n
        s = stride
        # Rows are exchanged with inverted diagonal entries, so each one is inverted once.
        inverted_rows = map(((A₋₁′, A₀′, A₊₁′, b′),) -> (A₋₁′, inv(A₀′), A₊₁′, b′), rows)
        neighboring_rows = pcr_neighbors(scope, inverted_rows, indices, n, s)
        rows = map(indices, rows, neighboring_rows) do i, row, (row⁻, row⁺)
            i = min(i, n)
            (A₋₁′, A₀′, A₊₁′, b′) = row
            ((A₋₁′⁻, inv_A₀′⁻, A₊₁′⁻, b′⁻), (A₋₁′⁺, inv_A₀′⁺, A₊₁′⁺, b′⁺)) =
                (row⁻, row⁺)
            k₋ = A₋₁′ * inv_A₀′⁻
            k₊ = A₊₁′ * inv_A₀′⁺
            k₋ = i > s ? k₋ : zero(k₋)
            k₊ = i + s <= n ? k₊ : zero(k₊)
            (
                -k₋ * A₋₁′⁻, A₀′ - k₋ * A₊₁′⁻ - k₊ * A₋₁′⁺, -k₊ * A₊₁′⁺,
                b′ - k₋ * b′⁻ - k₊ * b′⁺,
            )
        end
        stride += stride
    end
    unrolled_foreach(indices, rows) do i, (_, A₀′, _, b′)
        i <= n && (@inbounds x[i] = inv(A₀′) * b′)
    end
end

# Rows at a distance of s below and above each of the calling thread's rows,
# which are located at the specified indices in an array with at most n rows.
# The default method uses a buffer to exchange rows, but DataScopes can override
# this to use other forms of inter-thread communication (e.g. warp shuffles). Rows
# whose entries share a type and take more than 16 bytes (like rows of Float64s) are
# stored with each entry in a plane of its own, so that the threads of a warp access
# consecutive entries, without the bank conflicts of wider rows in shared memory.
@inline function pcr_neighbors(scope, rows, indices, n, s)
    T = row_plane_type(eltype(rows))
    isnothing(T) || return pcr_neighbors_by_entry(T, scope, rows, indices, n, s)
    buffer = DataLayouts.scoped_static_array(scope, eltype(rows), n)
    unrolled_foreach(indices, rows) do i, row
        i <= n && (@inbounds buffer[i] = row)
    end
    DataLayouts.synchronize(scope)
    return unrolled_map(indices) do i
        i = min(i, n)
        @inbounds (buffer[max(i - s, one(i))], buffer[min(i + s, n)])
    end
end

@inline row_plane_type(::Type{Tuple{T, Vararg{T, N}}}) where {N, T} =
    (N + 1) * sizeof(T) > 16 ? T : nothing
@inline row_plane_type(_) = nothing

@inline function pcr_neighbors_by_entry(::Type{T}, scope, rows, indices, n, s) where {T}
    N = fieldcount(eltype(rows))
    buffer = DataLayouts.scoped_static_array(scope, T, (n, N))
    unrolled_foreach(indices, rows) do i, row
        i <= n && unrolled_foreach(StaticOneTo(N)) do k
            @inline
            @inbounds buffer[i, k] = row[k]
        end
    end
    DataLayouts.synchronize(scope)
    return unrolled_map(indices) do i
        i = min(i, n)
        row_at(j) = ntuple(@inline(k -> @inbounds(buffer[j, k])), Val(N))
        (row_at(max(i - s, one(i))), row_at(min(i + s, n)))
    end
end
