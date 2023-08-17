"""
    FieldMatrixSolverAlgorithm

Description of how to solve an equation of the form `A * x = b` for `x`, where
`A` is a `FieldMatrix` and where `x` and `b` are both `FieldVector`s. Different
algorithms can be nested inside each other, enabling the construction of
specialized linear solvers that fully utilize the sparsity pattern of `A`.
"""
abstract type FieldMatrixSolverAlgorithm end

"""
    FieldMatrixSolver(alg, A, b)

Combination of a `FieldMatrixSolverAlgorithm` and the cache that it requires to
solve the equation `A * x = b` for `x`. The values of `A` and `b` that get
passed to this constructor should be `similar` to the ones that get passed to
`field_matrix_solve!` in order to ensure that the cache gets allocated
correctly.
"""
struct FieldMatrixSolver{A <: FieldMatrixSolverAlgorithm, C}
    alg::A
    cache::C
end
function FieldMatrixSolver(
    alg::FieldMatrixSolverAlgorithm,
    A::FieldMatrix,
    b::Fields.FieldVector,
)
    b_view = field_vector_view(b)
    cache = field_matrix_solver_cache(alg, A, b_view)
    check_field_matrix_solver(alg, cache, A, b_view)
    return FieldMatrixSolver(alg, cache)
end

"""
    field_matrix_solve!(solver, x, A, b)

Solves the equation `A * x = b` for `x` using the given `FieldMatrixSolver`.
"""
function field_matrix_solve!(
    solver::FieldMatrixSolver,
    x::Fields.FieldVector,
    A::FieldMatrix,
    b::Fields.FieldVector,
)
    x_view = field_vector_view(x)
    b_view = field_vector_view(b)
    keys(x_view) == keys(b_view) || error(
        "The linear system cannot be solved because x and b have incompatible \
         keys: $(set_string(keys(x_view))) vs. $(set_string(keys(b_view)))",
    )
    check_field_matrix_solver(solver.alg, solver.cache, A, b_view)
    field_matrix_solve!(solver.alg, solver.cache, x_view, A, b_view)
    return x
end

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

function partition_blocks(names₁, A, b, x = nothing)
    keys₁ = FieldVectorKeys(names₁, keys(b).name_tree)
    keys₂ = set_complement(keys₁)
    A₁₁ = A[cartesian_product(keys₁, keys₁)]
    A₁₂ = A[cartesian_product(keys₁, keys₂)]
    A₂₁ = A[cartesian_product(keys₂, keys₁)]
    A₂₂ = A[cartesian_product(keys₂, keys₂)]
    return isnothing(x) ? (A₁₁, A₁₂, A₂₁, A₂₂, b[keys₁], b[keys₂]) :
           (A₁₁, A₁₂, A₂₁, A₂₂, b[keys₁], b[keys₂], x[keys₁], x[keys₂])
end

################################################################################

"""
    BlockDiagonalSolve()

A `FieldMatrixSolverAlgorithm` for a block diagonal matrix `A`, which solves
each block's equation `Aᵢᵢ * xᵢ = bᵢ` in sequence. The equation for `xᵢ` is
solved as follows:
- If `Aᵢᵢ = λᵢ * I`, the equation is solved by setting `xᵢ .= inv(λᵢ) .* bᵢ`.
- If `Aᵢᵢ = Dᵢ`, where `Dᵢ` is a diagonal matrix, the equation is solved by
  making a single pass over the data, setting each `xᵢ[n] = inv(Dᵢ[n]) * bᵢ[n]`.
- If `Aᵢᵢ = Lᵢ * Dᵢ * Uᵢ`, where `Dᵢ` is a diagonal matrix and where `Lᵢ` and
  `Uᵢ` are unit lower and upper triangular matrices, respectively, the equation
  is solved using Gauss-Jordan elimination, which makes two passes over the
  data. The first pass multiplies both sides of the equation by `inv(Lᵢ * Dᵢ)`,
  replacing `Aᵢᵢ` with `Uᵢ` and `bᵢ` with `Uᵢxᵢ`, which is also referred to as
  putting `Aᵢᵢ` into "reduced row echelon form". The second pass solves
  `Uᵢ * xᵢ = Uᵢxᵢ` for `xᵢ` using a unit upper triangular matrix solver, which
  is also referred to as "back substitution". Only tri-diagonal and
  penta-diagonal matrices `Aᵢᵢ` are currently supported.
- The general case of `Aᵢᵢ = inv(Pᵢ) * Lᵢ * Uᵢ`, where `Pᵢ` is a row permutation
  matrix (i.e., LU factorization with partial pivoting), is not currently
  supported.
"""
struct BlockDiagonalSolve <: FieldMatrixSolverAlgorithm end

function field_matrix_solver_cache(::BlockDiagonalSolve, A, b)
    caches = map(matrix_row_keys(keys(A))) do name
        single_field_solver_cache(A[(name, name)], b[name])
    end
    return FieldNameDict{FieldName}(matrix_row_keys(keys(A)), caches)
end

function check_field_matrix_solver(::BlockDiagonalSolve, _, A, b)
    check_block_diagonal_matrix(
        A,
        "BlockDiagonalSolve cannot be used because A",
    )
    check_block_diagonal_matrix_has_no_missing_blocks(A, b)
    foreach(matrix_row_keys(keys(A))) do name
        check_single_field_solver(A[(name, name)], b[name])
    end
end

field_matrix_solve!(::BlockDiagonalSolve, cache, x, A, b) =
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[(name, name)], b[name])
    end

"""
    BlockLowerTriangularSolve(names₁...; [alg₁], [alg₂])

A `FieldMatrixSolverAlgorithm` for a block lower triangular matrix `A`, which
solves for `x` by executing the following steps:
1. Partition the entries in `A`, `x`, and `b` into the blocks `A₁₁`, `A₁₂`,
   `A₂₁`, `A₂₂`, `x₁`, `x₂`, `b₁`, and `b₂`, based on the `FieldName`s in
   `names₁`. In this notation, the subscript `₁` corresponds to `FieldName`s
   that are covered by `names₁`, while the subscript `₂` corresponds to all
   other `FieldNames`. A subscript in the first position refers to `FieldName`s
   that are used as row indices, while a subscript in the second position refers
   to column indices. This algorithm requires that the upper triangular block
   `A₁₂` be empty. (Any upper triangular solve can also be expressed as a lower
   triangular solve by swapping the subscripts `₁` and `₂`.)
2. Solve `A₁₁ * x₁ = b₁` for `x₁` using the algorithm `alg₁`, which is set to
   `BlockDiagonalSolve()` by default.
3. Solve `A₂₂ * x₂ = b₂ - A₂₁ * x₁` for `x₂` using the algorithm `alg₂`, which
   is set to `BlockDiagonalSolve()` by default.
"""
struct BlockLowerTriangularSolve{
    V <: NTuple{<:Any, FieldName},
    A1 <: FieldMatrixSolverAlgorithm,
    A2 <: FieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    names₁::V
    alg₁::A1
    alg₂::A2
end
BlockLowerTriangularSolve(
    names₁::FieldName...;
    alg₁ = BlockDiagonalSolve(),
    alg₂ = BlockDiagonalSolve(),
) = BlockLowerTriangularSolve(names₁, alg₁, alg₂)

function field_matrix_solver_cache(alg::BlockLowerTriangularSolve, A, b)
    A₁₁, _, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    cache₁ = field_matrix_solver_cache(alg.alg₁, A₁₁, b₁)
    b₂′ = similar(b₂)
    cache₂ = field_matrix_solver_cache(alg.alg₂, A₂₂, b₂′)
    return (; cache₁, b₂′, cache₂)
end

function check_field_matrix_solver(alg::BlockLowerTriangularSolve, cache, A, b)
    A₁₁, A₁₂, _, A₂₂, b₁, _ = partition_blocks(alg.names₁, A, b)
    isempty(keys(A₁₂)) || error(
        "BlockLowerTriangularSolve cannot be used because A has entries at the \
         following upper triangular keys: $(set_string(keys(A₁₂)))",
    )
    check_field_matrix_solver(alg.alg₁, cache.cache₁, A₁₁, b₁)
    check_field_matrix_solver(alg.alg₂, cache.cache₂, A₂₂, cache.b₂′)
end

function field_matrix_solve!(alg::BlockLowerTriangularSolve, cache, x, A, b)
    A₁₁, _, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    field_matrix_solve!(alg.alg₁, cache.cache₁, x₁, A₁₁, b₁)
    @. cache.b₂′ = b₂ - A₂₁ * x₁
    field_matrix_solve!(alg.alg₂, cache.cache₂, x₂, A₂₂, cache.b₂′)
end

"""
    SchurComplementSolve(names₁...; [alg₁])

A `FieldMatrixSolverAlgorithm` for a block matrix `A`, which solves for `x` by
executing the following steps:
1. Partition the entries in `A`, `x`, and `b` into the blocks `A₁₁`, `A₁₂`,
   `A₂₁`, `A₂₂`, `x₁`, `x₂`, `b₁`, and `b₂`, based on the `FieldName`s in
   `names₁`. In this notation, the subscript `₁` corresponds to `FieldName`s
   that are covered by `names₁`, while the subscript `₂` corresponds to all
   other `FieldNames`. A subscript in the first position refers to `FieldName`s
   that are used as row indices, while a subscript in the second position refers
   to column indices. This algorithm requires that the block `A₂₂` be a diagonal
   matrix, which allows it to assume that `inv(A₂₂)` can be computed on the fly.
2. Solve `(A₁₁ - A₁₂ * inv(A₂₂) * A₂₁) * x₁ = b₁ - A₁₂ * inv(A₂₂) * b₂` for `x₁`
   using the algorithm `alg₁`, which is set to `BlockDiagonalSolve()` by
   default. The matrix `A₁₁ - A₁₂ * inv(A₂₂) * A₂₁` is called the "Schur
   complement" of `A₂₂` in `A`.
3. Set `x₂` to `inv(A₂₂) * (b₂ - A₂₁ * x₁)`.
"""
struct SchurComplementSolve{
    V <: NTuple{<:Any, FieldName},
    A <: FieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    names₁::V
    alg₁::A
end
SchurComplementSolve(names₁::FieldName...; alg₁ = BlockDiagonalSolve()) =
    SchurComplementSolve(names₁, alg₁)

function field_matrix_solver_cache(alg::SchurComplementSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    A₁₁′ = @. A₁₁ - A₁₂ * inv(A₂₂) * A₂₁ # A₁₁′ could have more blocks than A₁₁
    b₁′ = similar(b₁)
    cache₁ = field_matrix_solver_cache(alg.alg₁, A₁₁′, b₁′)
    return (; A₁₁′, b₁′, cache₁)
end

function check_field_matrix_solver(alg::SchurComplementSolve, cache, A, b)
    _, _, _, A₂₂, _, b₂ = partition_blocks(alg.names₁, A, b)
    check_diagonal_matrix(A₂₂, "SchurComplementSolve cannot be used because A")
    check_block_diagonal_matrix_has_no_missing_blocks(A₂₂, b₂)
    check_field_matrix_solver(alg.alg₁, cache.cache₁, cache.A₁₁′, cache.b₁′)
end

function field_matrix_solve!(alg::SchurComplementSolve, cache, x, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    @. cache.A₁₁′ = A₁₁ - A₁₂ * inv(A₂₂) * A₂₁
    @. cache.b₁′ = b₁ - A₁₂ * inv(A₂₂) * b₂
    field_matrix_solve!(alg.alg₁, cache.cache₁, x₁, cache.A₁₁′, cache.b₁′)
    @. x₂ = inv(A₂₂) * (b₂ - A₂₁ * x₁)
end

"""
    ApproximateFactorizationSolve(name_pairs₁...; [alg₁], [alg₂])

A `FieldMatrixSolverAlgorithm` for a block matrix `A`, which (approximately)
solves for `x` by executing the following steps:
1. Use the entries in `A = M + I = M₁ + M₂ + I` to compute `A₁ = M₁ + I` and
   `A₂ = M₂ + I`, based on the pairs of `FieldName`s in `name_pairs₁`. In this
   notation, the subscript `₁` refers to pairs of `FieldName`s that are covered
   by `name_pairs₁`, while the subscript `₂` refers to all other pairs of 
   `FieldNames`s. This algorithm approximates the matrix `A` as the product
   `A₁ * A₂`, which introduces an error that scales roughly with the norm of
   `A₁ * A₂ - A = M₁ * M₂`. (More precisely, the error introduced by this
   algorithm is `x_exact - x_approx = inv(A) * b - inv(A₁ * A₂) * b`.)
2. Solve `A₁ * A₂x = b` for `A₂x` using the algorithm `alg₁`, which is set to
   `BlockDiagonalSolve()` by default.
3. Solve `A₂ * x = A₂x` for `x` using the algorithm `alg₂`, which is set to
   `BlockDiagonalSolve()` by default.
"""
struct ApproximateFactorizationSolve{
    V <: NTuple{<:Any, FieldNamePair},
    A1 <: FieldMatrixSolverAlgorithm,
    A2 <: FieldMatrixSolverAlgorithm,
} <: FieldMatrixSolverAlgorithm
    name_pairs₁::V
    alg₁::A1
    alg₂::A2
end
ApproximateFactorizationSolve(
    name_pairs₁::FieldNamePair...;
    alg₁ = BlockDiagonalSolve(),
    alg₂ = BlockDiagonalSolve(),
) = ApproximateFactorizationSolve(name_pairs₁, alg₁, alg₂)
# Note: This algorithm assumes that x is `similar` to b. In other words, it
# assumes that typeof(x) == typeof(b), rather than just keys(x) == keys(b).

function approximate_factors(name_pairs₁, A, b)
    keys₁ = FieldMatrixKeys(name_pairs₁, keys(b).name_tree)
    keys₂ = set_complement(keys₁)
    A₁ = A[keys₁] .+ one(A)[keys₂] # `one` can be used because x is similar to b
    A₂ = A[keys₂] .+ one(A)[keys₁]
    return A₁, A₂
end

function field_matrix_solver_cache(alg::ApproximateFactorizationSolve, A, b)
    A₁, A₂ = approximate_factors(alg.name_pairs₁, A, b)
    cache₁ = field_matrix_solver_cache(alg.alg₁, A₁, b)
    A₂x = @. A₂ * b # x can be replaced with b because they are similar
    cache₂ = field_matrix_solver_cache(alg.alg₂, A₂, A₂x)
    return (; cache₁, A₂x, cache₂)
end

function check_field_matrix_solver(
    alg::ApproximateFactorizationSolve,
    cache,
    A,
    b,
)
    A₁, A₂ = approximate_factors(alg.name_pairs₁, A, b)
    check_field_matrix_solver(alg.alg₁, cache.cache₁, A₁, b)
    check_field_matrix_solver(alg.alg₂, cache.cache₂, A₂, cache.A₂x)
end

function field_matrix_solve!(alg::ApproximateFactorizationSolve, cache, x, A, b)
    A₁, A₂ = approximate_factors(alg.name_pairs₁, A, b)
    field_matrix_solve!(alg.alg₁, cache.cache₁, cache.A₂x, A₁, b)
    field_matrix_solve!(alg.alg₂, cache.cache₂, x, A₂, cache.A₂x)
end
