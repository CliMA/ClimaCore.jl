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
get_scalar_keys
get_field_first_index_offset
broadcasted_get_field_type
inner_type_ignore_adjoint
```

## Utilities

```@docs
column_field2array
column_field2array_view
field2arrays
field2arrays_view
scalar_fieldmatrix
```

## Indexing a FieldMatrix

A FieldMatrix entry can be:

- An `UniformScaling`, which contains a `Number`
- A `DiagonalMatrixRow`, which can contain aything
- A `ColumnwiseBandMatrixField`, where each row is a [`BandMatrixRow`](@ref) where the band element type is representable with the space's base number type.

If an entry contains a composite type, the fields of that type can be extracted.
This is also true for nested composite types.

For example:

```@example 1
using ClimaCore.CommonSpaces # hide
import ClimaCore: MatrixFields, Quadratures # hide
import ClimaCore.MatrixFields: @name # hide
space = Box3DSpace(; # hide
           z_elem = 3, # hide
           x_min = 0, # hide
           x_max = 1, # hide
           y_min = 0, # hide
           y_max = 1, # hide
           z_min = 0, # hide
           z_max = 10, # hide
           periodic_x = false, # hide
           periodic_y = false, # hide
           n_quad_points = 1, # hide
           quad = Quadratures.GL{1}(), # hide
           x_elem = 1, # hide
           y_elem = 2, # hide
           staggering = CellCenter() # hide
       ) # hide
nt_entry_field = fill(MatrixFields.DiagonalMatrixRow((; foo = 1.0, bar = 2.0)), space)
nt_fieldmatrix = MatrixFields.FieldMatrix((@name(a), @name(b)) => nt_entry_field)
nt_fieldmatrix[(@name(a), @name(b))]
```

The internal values of the named tuples can be extracted with

```@example 1
nt_fieldmatrix[(@name(a.foo), @name(b))]
```

and

```@example 1
nt_fieldmatrix[(@name(a.bar), @name(b))]
```

If the key `(@name(name1), @name(name2))` corresponds to an entry, then
`(@name(foo.bar.buz), @name(biz.bop.fud))` would be the internal key for the key
`(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))`.

Currently, internal values cannot be extracted in all situations. Extracting interal values
works when:

- The second name in the internal key is empty, and the first name in the internal key accesses internal values for the type of element contained in each row of the entry. This does not work when the element type of each row is a 2d tensor.

- The first name in the internal key is empty, and the type of element contained in each row of the entry is an `AxisVector` or the adjoint of an `AxisVector`. In this case, the second name must access inernal values for the type of `AxisVector` contained in each row.

- The element type of each row in the entry is a 2d tensor, and the internal key is of the form `(@name(components.data.:(1)), @name(components.data.:(2)))`, but possibly with different numbers to index into the 2d tensor

- The element type of each row in the entry is some number of nested `Tuple`s and `NamedTuple`s, and the first name in the internal key accesses an `AxisVector` or the adjoint of an `AxisVector` from the outer `Tuple`/`NamedTuple`, and the second name in the inernal key accesses a component of the `AxisVector`

If the `FieldMatrix` represents a Jacobian, then extracting internal values works when an entry represents:

- The partial derrivative of an `AxisVector`, `Tuple`, or `NamedTuple` with respect to a scalar.

- The partial derrivative of a scalar with respect to an `AxisVector`.

- The partial derrivative of a `Tuple`, or `NamedTuple` with respect to an `AxisVector`. In this case, the first name of the internal key must index into the tuple and result in a scalar.

- The partial derrivative of an `AxisVector` with respect to an `AxisVector`. In this case, the partial derrivative of a component of the first `AxisVector` with respect to a component of the second `AxisVector` can be extracted, but not an entire `AxisVector` with respect to a component, or a component with respect to an entire `AxisVector`
