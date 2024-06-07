"""
    FieldMatrixWithSolver(A, b, [alg])

A wrapper that combines a `FieldMatrix` `A` with a `FieldMatrixSolver` that can
be used to solve the equation `A * x = b` for `x`, where `x` and `b` are both
`FieldVector`s. Similar to a `LinearAlgebra.Factorization`, this wrapper can be
passed to `ldiv!`, whereas a regular `FieldMatrix` cannot be passed to `ldiv!`.

By default, the `FieldMatrixSolverAlgorithm` `alg` is set to a
[`BlockDiagonalSolve`](@ref), so a custom `alg` must be specified when `A` is
not a block diagonal matrix.
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

# TODO: Find a simple way to make b an optional argument and add a method for
# Base.one(::FieldMatrixWithSolver).

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

ldiv!(x::Fields.FieldVector, A::FieldMatrixWithSolver, b::Fields.FieldVector) =
    field_matrix_solve!(A.solver, x, A.matrix, b)

mul!(b::Fields.FieldVector, A::FieldMatrixWithSolver, x::Fields.FieldVector) =
    @. b = A.matrix * x
