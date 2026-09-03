"""
    FieldMatrixWithSolver(A, b, alg = BlockDiagonalSolve())

Wrapper that combines a `FieldMatrix` `A` with a [`FieldMatrixSolver`](@ref) that
can be used to solve the equation `A * x = b` for `x`, where `x` and `b` are both
`FieldVector`s. Like a `LinearAlgebra.Factorization`, this wrapper can be passed to
`ldiv!`, whereas a regular `FieldMatrix` cannot.

By default, the [`FieldMatrixSolverAlgorithm`](@ref) `alg` is a
[`BlockDiagonalSolve`](@ref), so a custom `alg` must be specified when `A` is not
a block diagonal matrix.
"""
struct FieldMatrixWithSolver{M, S} <: AbstractDict{FieldNamePair, Any}
    matrix::M
    solver::S
end
FieldMatrixWithSolver(
    A::FieldMatrix,
    b::Fields.FieldVector,
    alg::FieldMatrixSolverAlgorithm = BlockDiagonalSolve(),
) = FieldMatrixWithSolver(A, FieldMatrixSolver(alg, A, b))

Base.keys(A::FieldMatrixWithSolver) = keys(A.matrix)

Base.values(A::FieldMatrixWithSolver) = values(A.matrix)

Base.pairs(A::FieldMatrixWithSolver) = pairs(A.matrix)

Base.length(A::FieldMatrixWithSolver) = length(A.matrix)

Base.iterate(A::FieldMatrixWithSolver, index = 1) = iterate(A.matrix, index)

Base.getindex(A::FieldMatrixWithSolver, key) = getindex(A.matrix, key)

Base.:(==)(A1::FieldMatrixWithSolver, A2::FieldMatrixWithSolver) =
    A1.matrix == A2.matrix && A1.solver.alg == A2.solver.alg

Base.similar(A::FieldMatrixWithSolver) =
    FieldMatrixWithSolver(similar(A.matrix), A.solver)

# Since zero(::FieldMatrix) retains the sparsity pattern of the original matrix
# while zeroing out all mutable entries, its linear solver is unchanged.
Base.zero(A::FieldMatrixWithSolver) =
    FieldMatrixWithSolver(zero(A.matrix), A.solver)

# Since one(::FieldMatrix) is an identity matrix, it does not require a linear
# solver. The equation I * x == b can be solved directly, without calling ldiv.
# TODO: Construct a linear solver for the identity matrix.
Base.one(A::FieldMatrixWithSolver) =
    FieldMatrixWithSolver(one(A.matrix), nothing)

Base.Broadcast.broadcastable(A::FieldMatrixWithSolver) = A.matrix

Base.Broadcast.materialize!(A::FieldMatrixWithSolver, matrix::FieldMatrix) =
    Base.Broadcast.materialize!(A.matrix, matrix)

ldiv!(x::Fields.FieldVector, A::FieldMatrixWithSolver, b::Fields.FieldVector) =
    isnothing(A.solver) ? error("FieldMatrixSolver is unavailable") :
    field_matrix_solve!(A.solver, x, A.matrix, b)

mul!(b::Fields.FieldVector, A::FieldMatrixWithSolver, x::Fields.FieldVector) =
    @. b = A.matrix * x
