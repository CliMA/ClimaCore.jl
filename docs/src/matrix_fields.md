# MatrixFields

```@meta
CurrentModule = ClimaCore.MatrixFields
```

```@docs
MatrixFields
```

## Matrix Field Element Type

```@docs
BandMatrixRow
```

## Matrix Field Multiplication

```@docs
MultiplyColumnwiseBandMatrixField
```

## Operator Matrices

```@docs
operator_matrix
```

## Vectors and Matrices of Fields

```@docs
FieldNameDict
identity_field_matrix
field_vector_view
concrete_field_vector
```

## Linear Solvers

```@docs
FieldMatrixSolverAlgorithm
FieldMatrixSolver
FieldMatrixWithSolver
field_matrix_solve!
BlockDiagonalSolve
BlockLowerTriangularSolve
BlockArrowheadSolve
SchurComplementReductionSolve
LazyFieldMatrixSolverAlgorithm
StationaryIterativeSolve
ApproximateBlockArrowheadIterativeSolve
```

## Preconditioners

```@docs
PreconditionerAlgorithm
MainDiagonalPreconditioner
BlockDiagonalPreconditioner
BlockArrowheadPreconditioner
BlockArrowheadSchurComplementPreconditioner
WeightedPreconditioner
CustomPreconditioner
```

## Internals

```@docs
outer_diagonals
band_matrix_row_type
matrix_shape
column_axes
AbstractLazyOperator
replace_lazy_operator
FieldName
@name
FieldNameTree
FieldNameSet
is_lazy
lazy_main_diagonal
lazy_mul
LazySchurComplement
field_matrix_solver_cache
check_field_matrix_solver
run_field_matrix_solver!
solver_algorithm
lazy_preconditioner
preconditioner_cache
check_preconditioner
lazy_or_concrete_preconditioner
apply_preconditioner
```

## Utilities

```@docs
column_field2array
column_field2array_view
field2arrays
field2arrays_view
```
