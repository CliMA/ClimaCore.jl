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
field_offset_and_type
```

## Utilities

```@docs
column_field2array
column_field2array_view
field2arrays
field2arrays_view
scalar_field_matrix
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

### Further Indexing Details

Let key `(@name(name1), @name(name2))` correspond to entry `sample_entry` in `FieldMatrix` `A`.
An example of this is:

```julia
 A = MatrixFields.FieldMatrix((@name(name1), @name(name2)) => sample_entry)
```

Now consider what happens indexing `A` with the key `(@name(name1.foo.bar.buz), @name(name2.biz.bop.fud))`.

First, a function searches the keys of `A` for a key that `(@name(foo.bar.buz), @name(biz.bop.fud))`
is a child of. In this example, `(@name(foo.bar.buz), @name(biz.bop.fud))` is a child of
the key `(@name(name1), @name(name2))`, and
`(@name(foo.bar.buz), @name(biz.bop.fud))` is referred to as the internal key.

Next, the entry that `(@name(name1), @name(name2))` is paired with is recursively indexed
by the internal key.

The recursive indexing of an internal entry given some entry `entry` and internal_key `internal_name_pair`
works as follows:

1. If the  `internal_name_pair` is blank, return `entry`
2. If the element type of each band of `entry` is an `Axis2Tensor`, and `internal_name_pair` is of the form
`(@name(components.data.1...), @name(components.data.2...))` (potentially with different numbers),
then extract the specified component, and recurse on it with the remaining `internal_name_pair`.
3. If the element type of each band of `entry` is a `Geometry.AdjointAxisVector`, then recurse on the parent of the adjoint.
4. If `internal_name_pair[1]` is not empty, and the first name in it is a field of the element type of each band of `entry`,
extract that field from `entry`, and recurse on the it with the remaining names of `internal_name_pair[1]` and all of `internal_name_pair[2]`
5. If `internal_name_pair[2]` is not empty, and the first name in it is a field of the element type of each row of `entry`,
extract that field from `entry`, and recurse on the it with all of `internal_name_pair[1]` and the remaining names of `internal_name_pair[2]`
6. At this point, if none of the previous cases are true, both `internal_name_pair[1]` and `internal_name_pair[2]` should be
non-empty, and it is assumed that `entry` is being used to implicitly represent some tensor structure. If the first name in
`internal_name_pair[1]` is equivalent to `internal_name_pair[2]`, then both the first names are dropped, and entry is recursed onto.
If the first names are different, both the first names are dropped, and the zero of entry is recursed onto.

When the entry is a `ColumnWiseBandMatrixField`, indexing it will return a broadcasted object in
the following situations:

1. The internal key indexes to a type different than the basetype of the entry
2. The internal key indexes to a zero-ed value
