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

single_field_solver_cache(::ScalingFieldMatrixEntry, b) = similar(b, Tuple{})
function single_field_solver_cache(A::ColumnwiseBandMatrixField, b)
    ud = outer_diagonals(eltype(A))[2]
    ud == 0 && return similar(b, Tuple{})
    T_U = x_type(eltype(eltype(A)), eltype(eltype(A)))
    return similar(b, Tuple{x_eltype(A, b), ntuple(Returns(T_U), Val(ud))...})
end

single_field_solve!(_, x, A::ScalingFieldMatrixEntry, b) =
    x .= (inv(scaling_value(A)),) .* b
function single_field_solve!(cache, x, A::ColumnwiseBandMatrixField, b)
    if eltype(A) <: DiagonalMatrixRow
        A₀ = A.entries.:1
        @. x = inv(A₀) * b
    else
        # Solves need no spaces, so the loop slices the data of each argument.
        DataLayouts._foreach_slice(
            DataLayouts.column,
            single_field_solve_column!,
            Spaces.get_mask(axes(x)),
            Val(false),
            Fields.field_values(cache),
            Base.broadcastable(Fields.field_values(x)),
            Fields.field_values(A),
            Base.broadcastable(Fields.field_values(b)),
        )
        call_post_op_callback() && post_op_callback(x, cache, x, A, b)
    end
    return nothing
end

# A tri-diagonal system in a column divided among multiple threads is solved by parallel
# cyclic reduction. Other systems are solved by Gaussian elimination in one thread.
@inline function single_field_solve_column!(cache, x, A, b)
    Aⱼs = unzip_tuple_field_values(A.entries)
    scope = DataLayouts.DataScope(x)
    if eltype(A) <: TridiagonalMatrixRow && scope != DataLayouts.ThisThread()
        pcr_solve!(scope, x, Aⱼs, b)
    else
        DataLayouts.owns_point(scope, 1) &&
            band_matrix_solve!(eltype(A), unzip_tuple_field_values(cache), x, Aⱼs, b)
    end
    DataLayouts.synchronize(scope)
    return nothing
end

# Generate the tuple directly instead of calling ntuple, since the views made
# by getproperty are only free of allocations when they do not escape through
# the non-inlined ntuple closure.
@generated function unzip_tuple_field_values(data)
    property_exprs =
        (:(getproperty(data, $i)) for i in 1:fieldcount(eltype(data)))
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, property_exprs...))
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
@inline function band_matrix_solve!(::Type{<:TridiagonalMatrixRow}, cache, x, Aⱼs, b)
    A₋₁, A₀, A₊₁ = Aⱼs
    Ux, U₊₁ = cache
    n = length(x)
    @inbounds begin
        inv_D₀ = inv(A₀[1])
        U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[1]
        Uxᵢ₋₁ = inv_D₀ * b[1]
        Ux[1] = Uxᵢ₋₁
        U₊₁[1] = U₊₁ᵢ₋₁

        for i in 2:n
            A₋₁ᵢ = A₋₁[i]
            inv_D₀ = inv(A₀[i] - A₋₁ᵢ * U₊₁ᵢ₋₁)
            Uxᵢ₋₁ = inv_D₀ * (b[i] - A₋₁ᵢ * Uxᵢ₋₁)
            Ux[i] = Uxᵢ₋₁
            if i < n
                U₊₁ᵢ₋₁ = inv_D₀ * A₊₁[i] # U₊₁[n] is outside the matrix.
                U₊₁[i] = U₊₁ᵢ₋₁
            end
        end

        x[n] = Ux[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        i = (n - 1)
        # for i in (n - 1):-1:1
        while i ≥ 1
            x[i] = Ux[i] - U₊₁[i] * x[i + 1]
            i -= 1
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
    - γ → L₋₁
    - μ → D₀
Transforms the penta-diagonal matrix into a unit upper tri-diagonal matrix, then
solves the resulting system using back substitution. The order of
multiplications has been modified in order to handle block vectors/matrices.
=#
@inline function band_matrix_solve!(::Type{<:PentadiagonalMatrixRow}, cache, x, Aⱼs, b)
    A₋₂, A₋₁, A₀, A₊₁, A₊₂ = Aⱼs
    Ux, U₊₁, U₊₂ = cache
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
            L₋₁ = A₋₁[i] - A₋₂[i] * U₊₁[i - 2]
            inv_D₀ = inv(A₀[i] - L₋₁ * U₊₁[i - 1] - A₋₂[i] * U₊₂[i - 2])
            Ux[i] = inv_D₀ * (b[i] - L₋₁ * Ux[i - 1] - A₋₂[i] * Ux[i - 2])
            i < n && (U₊₁[i] = inv_D₀ * (A₊₁[i] - L₋₁ * U₊₂[i - 1]))
            i < n - 1 && (U₊₂[i] = inv_D₀ * A₊₂[i])
        end

        x[n] = Ux[n]
        x[n - 1] = Ux[n - 1] - U₊₁[n - 1] * x[n]
        # Avoid steprange on GPU: https://cuda.juliagpu.org/stable/tutorials/performance/#Avoiding-StepRange
        # for i in (n - 2):-1:1
        i = (n - 2)
        while i ≥ 1
            x[i] = Ux[i] - U₊₁[i] * x[i + 1] - U₊₂[i] * x[i + 2]
            i -= 1
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
Each step eliminates the couplings of every row to the rows at a distance of stride, using
the equations of those rows to couple it to the rows at a distance of 2 * stride instead, so
that every row is decoupled from all others after log2(n) steps. A thread updates the rows
at its own points (see DataLayouts.owns_point), reading the other rows it needs through
pcr_neighbors. Entries outside of the matrix are replaced with zeros, so that they cannot
make any coefficient non-finite. As in the Thomas algorithm, the order of multiplications
handles block vectors/matrices.
=#
@inline function pcr_solve!(scope, x, (A₋₁, A₀, A₊₁), b)
    n = prod(DataLayouts.inferred_size(x))
    num_threads = DataLayouts.static_num_threads(scope)
    positions = ntuple(Val(cld(n, num_threads))) do k
        (k - 1) * num_threads + Int(DataLayouts.thread_rank(scope))
    end
    coefficients = map(positions) do position
        i = min(position, n)
        @inbounds (
            i > 1 ? A₋₁[i] : zero(eltype(A₋₁)),
            A₀[i],
            i < n ? A₊₁[i] : zero(eltype(A₊₁)),
            b[i],
        )
    end
    stride = 1
    while stride < n
        neighbors = pcr_neighbors(scope, coefficients, positions, stride, n)
        coefficients = let stride = stride
            map(positions, coefficients, neighbors) do position, row, (row₋, row₊)
                pcr_row(min(position, n), row, row₋, row₊, stride, n)
            end
        end
        stride *= 2
    end
    unrolled_foreach(positions, coefficients) do position, (_, bᵢ, _, dᵢ)
        position <= n && (@inbounds x[position] = inv(bᵢ) * dᵢ)
    end
end

# Eliminate the couplings of row i to the rows at a distance of stride, whose coefficients
# are row₋ and row₊ (or those of a boundary row, for rows outside of the matrix).
@inline function pcr_row(i, row, row₋, row₊, stride, n)
    ((aᵢ, bᵢ, cᵢ, dᵢ), (a₋, b₋, c₋, d₋), (a₊, b₊, c₊, d₊)) = (row, row₋, row₊)
    k₋ = -aᵢ * inv(b₋)
    k₊ = -cᵢ * inv(b₊)
    k₋ = i > stride ? k₋ : zero(k₋)
    k₊ = i + stride <= n ? k₊ : zero(k₊)
    new_row = (k₋ * a₋, bᵢ + k₋ * c₋ + k₊ * a₊, k₊ * c₊, dᵢ + k₋ * d₋ + k₊ * d₊)
    return map(oftype, row, new_row)
end

# Coefficients of the rows at a distance of stride below and above each of the calling
# thread's rows, published in a buffer that every thread of the scope can read; a row that
# is not in the matrix is replaced by the boundary row closest to it. Rows are published in
# chunks of 8 words, so that every solve over as many threads shares one scratch buffer.
@inline function pcr_neighbors(scope, coefficients, positions, stride, n)
    words = map(coefficients) do row
        DataLayouts.bitcast_struct(DataLayouts.words_type(typeof(row)), row)
    end
    num_words = Val(fieldcount(eltype(words)))
    slots = length(positions) * DataLayouts.static_num_threads(scope)
    buffer =
        DataLayouts.scoped_scratch_array(scope, NTuple{8, eltype(eltype(words))}, slots)
    chunks = ntuple(Val(cld(fieldcount(eltype(words)), 8))) do chunk
        DataLayouts.synchronize(scope)
        unrolled_foreach(positions, words) do position, row_words
            chunk_words = ntuple(Val(8)) do k
                word = 8 * (chunk - 1) + k
                word <= length(row_words) ? row_words[word] : zero(eltype(row_words))
            end
            position <= n && (@inbounds buffer[position] = chunk_words)
        end
        DataLayouts.synchronize(scope)
        map(positions) do position
            i = min(position, n)
            @inbounds (buffer[max(i - stride, 1)], buffer[min(i + stride, n)])
        end
    end
    return ntuple(Val(length(positions))) do p
        ntuple(Val(2)) do side
            row_words = ntuple(k -> chunks[cld(k, 8)][p][side][mod1(k, 8)], num_words)
            DataLayouts.bitcast_struct(eltype(coefficients), row_words)
        end
    end
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

We might eventually want a single general method for band_matrix_solve!, similar
to the BLAS.gbsv function. For now, though, the methods above should be enough.
=#
