"""
    FieldMatrixSolverAlgorithm

Description of how to solve an equation of the form `A * x = b` for `x`, where
`A` is a `FieldMatrix` and where `x` and `b` are both `FieldVector`s. Different
algorithms can be nested inside each other, enabling the construction of
specialized linear solvers that fully utilize the sparsity pattern of `A`.

# Interface

Every subtype of `FieldMatrixSolverAlgorithm` must implement methods for the
following functions:
- [`field_matrix_solver_cache`](@ref)
- [`check_field_matrix_solver`](@ref)
- [`run_field_matrix_solver!`](@ref)
"""
abstract type FieldMatrixSolverAlgorithm end

"""
    field_matrix_solver_cache(alg, A, b)

Allocates the cache required by the `FieldMatrixSolverAlgorithm` `alg` to solve
the equation `A * x = b`.
"""
function field_matrix_solver_cache end

"""
    check_field_matrix_solver(alg, cache, A, b)

Checks that the sparsity structure of `A` is supported by the
`FieldMatrixSolverAlgorithm` `alg`, and that `A` is compatible with `b` in the
equation `A * x = b`.
"""
function check_field_matrix_solver end

"""
    run_field_matrix_solver!(alg, cache, x, A, b)

Sets `x` to the value that solves the equation `A * x = b` using the
`FieldMatrixSolverAlgorithm` `alg`.
"""
function run_field_matrix_solver! end

"""
    FieldMatrixSolver(alg, A, b)

Combination of a `FieldMatrixSolverAlgorithm` `alg` and the cache it requires to
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
    A_with_tree = replace_name_tree(A, keys(b_view).name_tree)
    cache = field_matrix_solver_cache(alg, A_with_tree, b_view)
    check_field_matrix_solver(alg, cache, A_with_tree, b_view)
    return FieldMatrixSolver(alg, cache)
end

"""
    field_matrix_solve!(solver, x, A, b)

Solves the equation `A * x = b` for `x` using the `FieldMatrixSolver` `solver`.
"""
NVTX.@annotate function field_matrix_solve!(
    solver::FieldMatrixSolver,
    x::Fields.FieldVector,
    A::FieldMatrix,
    b::Fields.FieldVector,
)
    (; alg, cache) = solver
    x_view = field_vector_view(x)
    b_view = field_vector_view(b)
    keys(x_view) == keys(b_view) || error(
        "The linear system cannot be solved because x and b have incompatible \
         keys: $(set_string(keys(x_view))) vs. $(set_string(keys(b_view)))",
    )
    A_with_tree = replace_name_tree(A, keys(b_view).name_tree)
    check_field_matrix_solver(alg, cache, A_with_tree, b_view)
    run_field_matrix_solver!(alg, cache, x_view, A_with_tree, b_view)
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

function partition_blocks(names₁, A, b = nothing, x = nothing)
    keys₁ = FieldVectorKeys(names₁, keys(A).name_tree)
    keys₂ = set_complement(keys₁)
    A₁₁ = A[cartesian_product(keys₁, keys₁)]
    A₁₂ = A[cartesian_product(keys₁, keys₂)]
    A₂₁ = A[cartesian_product(keys₂, keys₁)]
    A₂₂ = A[cartesian_product(keys₂, keys₂)]
    b_blocks = isnothing(b) ? () : (b[keys₁], b[keys₂])
    x_blocks = isnothing(x) ? () : (x[keys₁], x[keys₂])
    return (A₁₁, A₁₂, A₂₁, A₂₂, b_blocks..., x_blocks...)
end

function similar_to_x(A, b)
    entries = map(matrix_row_keys(keys(A))) do name
        similar(b[name], x_eltype(A[name, name], b[name]))
    end
    return FieldNameDict(matrix_row_keys(keys(A)), entries)
end

################################################################################

# Lazy (i.e., as matrix-free as possible) operations for FieldMatrix and
# analogues of FieldMatrix

lazy_inv(A) = Base.Broadcast.broadcasted(inv, A)
lazy_add(As...) = Base.Broadcast.broadcasted(+, As...)
lazy_sub(As...) = Base.Broadcast.broadcasted(-, As...)

"""
    lazy_mul(A, args...)

Constructs a lazy `FieldMatrix` that represents the product `@. *(A, args...)`.
This involves regular broadcasting when `A` is a `FieldMatrix`, but it has more
complex behavior for other objects like the [`LazySchurComplement`](@ref).
"""
lazy_mul(A, args...) = Base.Broadcast.broadcasted(*, A, args...)

"""
    LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, [alg₁, cache₁, A₁₂_x₂, invA₁₁_A₁₂_x₂])

An analogue of a `FieldMatrix` that represents the Schur complement of `A₁₁` in
`A`, `A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`. Since `inv(A₁₁)` will generally be a dense
matrix, it would not be efficient to directly compute the Schur complement. So,
this object only supports the "lazy" functions [`lazy_mul`](@ref), which allows
it to be multiplied by the vector `x₂`, and [`lazy_preconditioner`](@ref), which
allows it to be approximated with a `FieldMatrix`.

The values `alg₁`, `cache₁`, `A₁₂_x₂`, and `invA₁₁_A₁₂_x₂` need to be specified
in order for `lazy_mul` to be able to compute `inv(A₁₁) * A₁₂ * x₂`. When a
`LazySchurComplement` is not passed to `lazy_mul`, these values can be omitted.
"""
struct LazySchurComplement{M11, M12, M21, M22, A1, C1, V1, V2}
    A₁₁::M11
    A₁₂::M12
    A₂₁::M21
    A₂₂::M22
    alg₁::A1
    cache₁::C1
    A₁₂_x₂::V1
    invA₁₁_A₁₂_x₂::V2
end
LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂) =
    LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, nothing, nothing, nothing, nothing)

NVTX.@annotate function lazy_mul(A₂₂′::LazySchurComplement, x₂)
    (; A₁₁, A₁₂, A₂₁, A₂₂, alg₁, cache₁, A₁₂_x₂, invA₁₁_A₁₂_x₂) = A₂₂′
    zero_rows = setdiff(keys(A₁₂_x₂), matrix_row_keys(keys(A₁₂)))
    @. A₁₂_x₂ = A₁₂ * x₂ + zero(A₁₂_x₂[zero_rows])
    run_field_matrix_solver!(alg₁, cache₁, invA₁₁_A₁₂_x₂, A₁₁, A₁₂_x₂)
    return lazy_sub(lazy_mul(A₂₂, x₂), lazy_mul(A₂₁, invA₁₁_A₁₂_x₂))
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
This algorithm solves the `N` block equations `Aₙₙ * xₙ = bₙ` in sequence (though
we might want to parallelize it in the future).

If `Aₙₙ` is a diagonal matrix, the equation `Aₙₙ * xₙ = bₙ` is solved by making a
single pass over the data, setting each `xₙ[i]` to `inv(Aₙₙ[i, i]) * bₙ[i]`.

Otherwise, the equation `Aₙₙ * xₙ = bₙ` is solved using Gaussian elimination
(without pivoting), which makes two passes over the data. This is currently only
implemented for tri-diagonal and penta-diagonal matrices `Aₙₙ`. In Gaussian
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
"""
struct BlockDiagonalSolve <: FieldMatrixSolverAlgorithm end

function field_matrix_solver_cache(::BlockDiagonalSolve, A, b)
    caches = map(matrix_row_keys(keys(A))) do name
        single_field_solver_cache(A[name, name], b[name])
    end
    return FieldNameDict(matrix_row_keys(keys(A)), caches)
end

function check_field_matrix_solver(::BlockDiagonalSolve, _, A, b)
    check_block_diagonal_matrix(
        A,
        "BlockDiagonalSolve cannot be used because A",
    )
    check_block_diagonal_matrix_has_no_missing_blocks(A, b)
    foreach(matrix_row_keys(keys(A))) do name
        check_single_field_solver(A[name, name], b[name])
    end
end

# multiple_field_solve! seems to use too many registers, possibly from branch divergence.
# NVTX.@annotate run_field_matrix_solver!(::BlockDiagonalSolve, cache, x, A, b) =
#     multiple_field_solve!(cache, x, A, b)

NVTX.@annotate run_field_matrix_solver!(::BlockDiagonalSolve, cache, x, A, b) =
    foreach(matrix_row_keys(keys(A))) do name
        single_field_solve!(cache[name], x[name], A[name, name], b[name])
    end

"""
    BlockLowerTriangularSolve(names₁...; [alg₁], [alg₂])

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

function field_matrix_solver_cache(alg::BlockLowerTriangularSolve, A, b)
    A₁₁, _, _, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
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

NVTX.@annotate function run_field_matrix_solver!(
    alg::BlockLowerTriangularSolve,
    cache,
    x,
    A,
    b,
)
    A₁₁, _, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    run_field_matrix_solver!(alg.alg₁, cache.cache₁, x₁, A₁₁, b₁)
    @. cache.b₂′ = b₂ - A₂₁ * x₁
    run_field_matrix_solver!(alg.alg₂, cache.cache₂, x₂, A₂₂, cache.b₂′)
end

"""
    BlockArrowheadSolve(names₁...; [alg₂])

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

Since `A₁₁` is a diagonal matrix, `inv(A₁₁)` is easy to compute, which means
that the Schur complement of `A₁₁` in `A`, `A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`, as well
as the vectors `b₂ - A₂₁ * inv(A₁₁) * b₁` and `inv(A₁₁) * (b₁ - A₁₂ * x₂)`, are
also easy to compute.

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

function field_matrix_solver_cache(alg::BlockArrowheadSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, _, b₂ = partition_blocks(alg.names₁, A, b)
    A₂₂′ = @. A₂₂ - A₂₁ * inv(A₁₁) * A₁₂
    b₂′ = similar(b₂)
    cache₂ = field_matrix_solver_cache(alg.alg₂, A₂₂′, b₂′)
    return (; A₂₂′, b₂′, cache₂)
end

function check_field_matrix_solver(alg::BlockArrowheadSolve, cache, A, b)
    A₁₁, _, _, _, b₁, _ = partition_blocks(alg.names₁, A, b)
    check_diagonal_matrix(A₁₁, "BlockArrowheadSolve cannot be used because A")
    check_block_diagonal_matrix_has_no_missing_blocks(A₁₁, b₁)
    check_field_matrix_solver(alg.alg₂, cache.cache₂, cache.A₂₂′, cache.b₂′)
end

NVTX.@annotate function run_field_matrix_solver!(
    alg::BlockArrowheadSolve,
    cache,
    x,
    A,
    b,
)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    @. cache.A₂₂′ = A₂₂ - A₂₁ * inv(A₁₁) * A₁₂
    @. cache.b₂′ = b₂ - A₂₁ * inv(A₁₁) * b₁
    run_field_matrix_solver!(alg.alg₂, cache.cache₂, x₂, cache.A₂₂′, cache.b₂′)
    @. x₁ = inv(A₁₁) * (b₁ - A₁₂ * x₂)
end

"""
    SchurComplementReductionSolve(names₁...; [alg₁], alg₂)

A `FieldMatrixSolverAlgorithm` for any 2×2 block matrix:
```math
A = \\begin{bmatrix} A_{11} & A_{12} \\\\ A_{21} & A_{22} \\end{bmatrix}
```
The `FieldName`s in `names₁` correspond to the subscript `₁`, while all other
`FieldName`s correspond to the subscript `₂`. This algorithm has 3 steps:
1. Solve `A₁₁ * x₁′ = b₁` for `x₁′` using the algorithm `alg₁`, which is set to
   a [`BlockDiagonalSolve`](@ref) by default.
2. Solve `(A₂₂ - A₂₁ * inv(A₁₁) * A₁₂) * x₂ = b₂ - A₂₁ * x₁′` for `x₂`
   using the algorithm `alg₂`.
3. Solve `A₁₁ * x₁ = b₁ - A₁₂ * x₂` for `x₁` using the algorithm `alg₁`.

Since `A₁₁` is not necessarily a diagonal matrix, `inv(A₁₁)` will generally be a
dense matrix, which means that the Schur complement of `A₁₁` in `A`,
`A₂₂ - A₂₁ * inv(A₁₁) * A₁₂`, cannot be computed efficiently. So, `alg₂` must be
set to a `LazyFieldMatrixSolverAlgorithm`, which can evaluate the matrix-vector
product `(A₂₂ - A₂₁ * inv(A₁₁) * A₁₂) * x₂` without actually computing the Schur
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

function field_matrix_solver_cache(alg::SchurComplementReductionSolve, A, b)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂ = partition_blocks(alg.names₁, A, b)
    b₁′ = similar(b₁)
    cache₁ = field_matrix_solver_cache(alg.alg₁, A₁₁, b₁)
    A₂₂′ = LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂)
    b₂′ = similar(b₂)
    cache₂ = field_matrix_solver_cache(alg.alg₂, A₂₂′, b₂′)
    return (; b₁′, cache₁, b₂′, cache₂)
end

function check_field_matrix_solver(
    alg::SchurComplementReductionSolve,
    cache,
    A,
    b,
)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, _ = partition_blocks(alg.names₁, A, b)
    check_field_matrix_solver(alg.alg₁, cache.cache₁, A₁₁, b₁)
    A₂₂′ = LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂)
    check_field_matrix_solver(alg.alg₂, cache.cache₂, A₂₂′, cache.b₂′)
end

NVTX.@annotate function run_field_matrix_solver!(
    alg::SchurComplementReductionSolve,
    cache,
    x,
    A,
    b,
)
    A₁₁, A₁₂, A₂₁, A₂₂, b₁, b₂, x₁, x₂ = partition_blocks(alg.names₁, A, b, x)
    x₁′ = x₁ # Use x₁ as temporary storage to avoid additional allocations.
    schur_complement_args = (alg.alg₁, cache.cache₁, cache.b₁′, x₁′)
    A₂₂′ = LazySchurComplement(A₁₁, A₁₂, A₂₁, A₂₂, schur_complement_args...)
    run_field_matrix_solver!(alg.alg₁, cache.cache₁, x₁′, A₁₁, b₁)
    @. cache.b₂′ = b₂ - A₂₁ * x₁′
    run_field_matrix_solver!(alg.alg₂, cache.cache₂, x₂, A₂₂′, cache.b₂′)
    @. cache.b₁′ = b₁ - A₁₂ * x₂
    run_field_matrix_solver!(alg.alg₁, cache.cache₁, x₁, A₁₁, cache.b₁′)
end
