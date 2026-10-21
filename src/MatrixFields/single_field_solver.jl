import ..DataLayouts

# Each value of x is a product of a value of b with the inverse of a value of A.
inv_mul(a, b) = inv(a) * b
x_eltype(A::ScalingFieldMatrixEntry, b) =
    x_type(eltype(A), eltype(Base.broadcastable(b)))
x_eltype(A::ColumnwiseBandMatrixField, b) =
    x_type(eltype(eltype(A)), eltype(Base.broadcastable(b)))
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
# threads are available. Otherwise, one thread performs Gaussian elimination.
# Every column of a thread pool is solved by one thread, so its caches are needed.
use_parallel_solver(A) =
    eltype(A) <: TridiagonalMatrixRow &&
    !(
        DataLayouts.DataScope(A) isa
        Union{DataLayouts.ThisThread, DataLayouts.ThisThreadPool}
    )

single_field_solver_cache(::ScalingFieldMatrixEntry, b) = similar(b, Tuple{})
function single_field_solver_cache(A::ColumnwiseBandMatrixField, b)
    # A column with only one level is also solved by one thread on a GPU.
    (
        eltype(A) <: DiagonalMatrixRow ||
        use_parallel_solver(A) && Spaces.nlevels(axes(A)) > 1
    ) &&
        return similar(b, Tuple{})
    ud = outer_diagonals(eltype(A))[2]
    T_U = x_type(eltype(eltype(A)), eltype(eltype(A)))
    return similar(b, Tuple{x_eltype(A, b), ntuple(Returns(T_U), Val(ud))...})
end

single_field_solve!(_, x, A::ScalingFieldMatrixEntry, b) =
    x .= (inv(scaling_value(A)),) .* b
single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b) =
    if eltype(A) <: DiagonalMatrixRow
        A₀ = A.entries.:1
        @. x = inv(A₀) * b
    else
        slice_op = DataLayouts.column
        mask = Spaces.get_mask(axes(x))
        # Solves need no spaces, so the loop slices the data of each argument.
        args = map(Fields.field_values, (cache, x, A, b))
        DataLayouts._foreach_slice(
            slice_op,
            single_column_solve!,
            mask,
            Val(false),
            args...,
        )
    end

# ntuple passes literal indices to the inlined function, which keeps every view
# type-stable in large loops, unlike the values passed by unrolled_map.
@inline subfields(field) =
    ntuple(@inline(n -> getproperty(field, n)), Val(fieldcount(eltype(field))))

function single_column_solve!(cache, x, A, b)
    Aⱼs = subfields(A.entries)
    x′ = Base.broadcastable(x)
    b′ = Base.broadcastable(b)
    scope = DataLayouts.DataScope(x)
    # In a slice loop, other threads can read the solution in later statements.
    if use_parallel_solver(A)
        parallel_band_matrix_solve!(x′, Aⱼs, b′)
    else
        DataLayouts.owns_point(scope, 1) &&
            band_matrix_solve!(subfields(cache), x′, Aⱼs, b′)
    end
    DataLayouts.synchronize(scope)
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
Solves the tridiagonal system A * x = b by factoring A into L * D * U, where L
is unit lower bi-diagonal, D is diagonal, and U is unit upper bi-diagonal. One
step computes U and U * x, and another step solves for x with back substitution.
The order of multiplications has been modified to handle block vectors/matrices.
=#
@inline function band_matrix_solve!(Uⱼs, x, Aⱼs::NTuple{3}, b)
    A₋₁, A₀, A₊₁ = Aⱼs
    Ux, U₊₁ = Uⱼs
    n = length(x)
    @inbounds begin
        # The previous row is kept in registers, since reading it from the cache
        # would make every step wait for the store of the step before it.
        inv_D₀ = inv(A₀[1])
        Uxᵢ₋₁ = Ux[1] = inv_D₀ * b[1]
        U₊₁ᵢ₋₁ = U₊₁[1] = inv_D₀ * A₊₁[1]
        for i in 2:n
            inv_D₀ = inv(A₀[i] - A₋₁[i] * U₊₁ᵢ₋₁)
            Uxᵢ₋₁ = Ux[i] = inv_D₀ * (b[i] - A₋₁[i] * Uxᵢ₋₁)
            i < n && (U₊₁ᵢ₋₁ = U₊₁[i] = inv_D₀ * A₊₁[i])
        end

        # As above, the next row of x is kept in registers instead of reread.
        xᵢ₊₁ = x[n] = Ux[n]
        for i′ in 1:(n - 1)
            i = n - i′
            xᵢ₊₁ = x[i] = Ux[i] - U₊₁[i] * xᵢ₊₁
        end
    end
end

#=
The PTRANS-I algorithm, as presented in
    https://www.hindawi.com/journals/mpe/2015/232456/alg1,
but with the following variable name changes:
    - e → A₋₂
    - c → A₋₁
    - d → A₀
    - a → A₊₁
    - b → A₊₂
    - y → b
    - α → U₊₁
    - β → U₊₂
    - z → Ux
    - γ → A₋₁′
    - μ → D₀
Solves the pentadiagonal system A * x = b by factoring A into L * D * U, where L
is unit lower tridiagonal, D is diagonal, and U is unit upper tridiagonal. One
step computes U and U * x, and another step solves for x with back substitution.
The order of multiplications has been modified to handle block vectors/matrices.
=#
@inline function band_matrix_solve!(Uⱼs, x, Aⱼs::NTuple{5}, b)
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs
    Ux, U₊₁, U₊₂ = Uⱼs
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[1])
        Ux[1] = inv_D₀ * b[1]
        U₊₁[1] = inv_D₀ * A₊₁[1]
        U₊₂[1] = inv_D₀ * A₊₂[1]
        inv_D₀ = inv(A₀[2] - A₋₁[2] * U₊₁[1])
        Ux[2] = inv_D₀ * (b[2] - A₋₁[2] * Ux[1])
        U₊₁[2] = inv_D₀ * (A₊₁[2] - A₋₁[2] * U₊₂[1])
        U₊₂[2] = inv_D₀ * A₊₂[2]
        for i in 3:n
            A₋₁′ = A₋₁[i] - A₋₂[i] * U₊₁[i - 2]
            inv_D₀ = inv(A₀[i] - A₋₁′ * U₊₁[i - 1] - A₋₂[i] * U₊₂[i - 2])
            Ux[i] = inv_D₀ * (b[i] - A₋₁′ * Ux[i - 1] - A₋₂[i] * Ux[i - 2])
            i < n && (U₊₁[i] = inv_D₀ * (A₊₁[i] - A₋₁′ * U₊₂[i - 1]))
            i < n - 1 && (U₊₂[i] = inv_D₀ * A₊₂[i])
        end

        xᵢ₊₂ = x[n] = Ux[n]
        xᵢ₊₁ = x[n - 1] = Ux[n - 1] - U₊₁[n - 1] * xᵢ₊₂
        for i′ in 2:(n - 1)
            i = n - i′
            xᵢ = x[i] = Ux[i] - U₊₁[i] * xᵢ₊₁ - U₊₂[i] * xᵢ₊₂
            (xᵢ₊₁, xᵢ₊₂) = (xᵢ, xᵢ₊₁)
        end
    end
end

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
@inline function parallel_band_matrix_solve!(x, Aⱼs::NTuple{3}, b)
    A₋₁, A₀, A₊₁ = Aⱼs
    n = length(x)
    scope = DataLayouts.DataScope(x)
    indices = ntuple(Val(cld(n, DataLayouts.static_num_threads(scope)))) do row_number
        (row_number - 1) * DataLayouts.num_threads(scope) + DataLayouts.thread_rank(scope)
    end
    zero_A = zero(eltype(A₀))
    rows = unrolled_map(indices) do i
        i = min(i, n)
        @inbounds (i > 1 ? A₋₁[i] : zero_A, A₀[i], i < n ? A₊₁[i] : zero_A, b[i])
    end
    stride = 1
    while stride < n
        s = stride # Define a new local variable to avoid boxing in map.
        neighboring_rows = pcr_neighbors(scope, rows, indices, n, s)
        rows = map(indices, rows, neighboring_rows) do i, row, (row⁻, row⁺)
            i = min(i, n)
            (A₋₁′, A₀′, A₊₁′, b′) = row
            ((A₋₁′⁻, A₀′⁻, A₊₁′⁻, b′⁻), (A₋₁′⁺, A₀′⁺, A₊₁′⁺, b′⁺)) = (row⁻, row⁺)
            k₋ = A₋₁′ * inv(A₀′⁻)
            k₊ = A₊₁′ * inv(A₀′⁺)
            k₋ = i > s ? k₋ : zero(k₋)
            k₊ = i + s <= n ? k₊ : zero(k₊)
            (
                -k₋ * A₋₁′⁻, A₀′ - k₋ * A₊₁′⁻ - k₊ * A₋₁′⁺, -k₊ * A₊₁′⁺,
                b′ - k₋ * b′⁻ - k₊ * b′⁺,
            )
        end
        stride *= 2
    end
    unrolled_foreach(indices, rows) do i, (_, A₀′, _, b′)
        i <= n && (@inbounds x[i] = inv(A₀′) * b′)
    end
end

# Rows at a distance of s below and above each of the calling thread's rows,
# which are located at the specified indices in an array with at most n rows.
# The default method uses a buffer to exchange rows, but DataScopes can override
# this to use other forms of inter-thread communication (e.g. warp shuffles).
@inline function pcr_neighbors(scope, rows, indices, n, s)
    buffer = DataLayouts.scoped_static_array(scope, eltype(rows), n)
    DataLayouts.synchronize(scope) # Wait for every thread to read the previous rows.
    unrolled_foreach(indices, rows) do i, row
        i <= n && (@inbounds buffer[i] = row)
    end
    DataLayouts.synchronize(scope)
    return unrolled_map(indices) do i
        i = min(i, n)
        @inbounds (buffer[max(i - s, 1)], buffer[min(i + s, n)])
    end
end
